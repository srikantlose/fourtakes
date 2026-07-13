"""Local demo web UI for FourTakes: drag-and-drop a clip, watch the four
takes generate live.

Not part of the judged submission — the Dockerfile never COPYs this
directory, and requirements-web.txt (just Flask) is separate from the
submission image's requirements.txt. Wraps the existing pipeline pieces
(FrameExtractor, Captioner, FireworksClient) exactly as-is; no pipeline
behavior changes here.

Run: python -m webapp.server
"""

import json
import logging
import queue
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.captioner import Captioner, STYLE_KEYS  # noqa: E402
from src.config import load_config, load_prompts  # noqa: E402
from src.fireworks_client import FireworksClient  # noqa: E402
from src.frame_extractor import FrameExtractor  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402
from src.pipeline import VIDEO_EXTENSIONS, downsample_frames  # noqa: E402

logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300 MB

# job_id -> queue.Queue of SSE event dicts. A single-user local dev tool,
# so an in-memory dict is enough; no persistence needed.
JOBS: dict = {}


def _emit(q: "queue.Queue", event_type: str, **data) -> None:
    q.put({"type": event_type, **data})


def _process_job(job_id: str, video_path: str, temp_root: Path, mock: bool) -> None:
    q = JOBS[job_id]
    started = time.time()
    try:
        config = load_config()
        if mock:
            config["mock_mode"] = True
        prompts = load_prompts(config["prompts_path"])

        client = FireworksClient(
            api_key=config["fireworks_api_key"],
            model=config["fireworks_caption_model"],
            base_url=config["fireworks_base_url"],
            mock_mode=config["mock_mode"],
            timeout=config.get("api_timeout", 60),
        )
        captioner = Captioner(client, prompts)
        extractor = FrameExtractor(
            frame_interval=config["frame_interval_seconds"],
            frame_scale_width=config.get("frame_scale_width", 512),
        )

        _emit(q, "frames_extracting")
        frame_paths, _meta = extractor.extract_frames(
            video_path, output_dir=str(temp_root / "frames")
        )
        if not frame_paths:
            raise RuntimeError("No frames extracted from this clip")

        sampled = downsample_frames(frame_paths, config["max_frames"])
        _emit(q, "frames_done", extracted=len(frame_paths), sent=len(sampled))

        _emit(q, "base_captioning")
        base_caption = captioner.generate_base_caption(sampled)
        _emit(q, "base_done", text=base_caption, mock=client.mock_mode)

        def _one_style(style: str) -> tuple:
            try:
                return style, captioner.generate_styled_caption(style, base_caption), False
            except Exception as exc:
                logger.warning("Style '%s' failed in web demo: %s", style, exc)
                return style, base_caption, True

        with ThreadPoolExecutor(max_workers=len(STYLE_KEYS)) as pool:
            futures = [pool.submit(_one_style, s) for s in STYLE_KEYS]
            for future in as_completed(futures):
                style, text, fallback = future.result()
                _emit(q, "style_done", style=style, text=text, fallback=fallback)

        _emit(
            q, "done",
            seconds=round(time.time() - started, 1),
            calls=client.usage_summary()["calls"],
            mock=client.mock_mode,
        )
    except Exception as exc:
        logger.error("Web demo job %s failed: %s", job_id, exc)
        _emit(q, "error", message=str(exc))
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
        q.put(None)  # sentinel: end of stream


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/caption", methods=["POST"])
def start_caption():
    upload = request.files.get("video")
    if upload is None or not upload.filename:
        return jsonify({"error": "No video file provided"}), 400

    suffix = Path(secure_filename(upload.filename)).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        suffix = ".mp4"

    mock = request.form.get("mock", "false").lower() == "true"

    job_id = uuid.uuid4().hex
    temp_root = PROJECT_ROOT / ".temp" / "webapp" / job_id
    temp_root.mkdir(parents=True, exist_ok=True)
    video_path = str(temp_root / f"upload{suffix}")
    upload.save(video_path)

    JOBS[job_id] = queue.Queue()
    thread = threading.Thread(
        target=_process_job, args=(job_id, video_path, temp_root, mock), daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/events/<job_id>")
def events(job_id: str):
    q = JOBS.get(job_id)
    if q is None:
        return jsonify({"error": "Unknown job_id"}), 404

    def stream():
        try:
            while True:
                event = q.get()
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            JOBS.pop(job_id, None)

    return Response(stream(), mimetype="text/event-stream")


def main() -> None:
    setup_logging()
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)


if __name__ == "__main__":
    main()
