"""FourTakes pipeline orchestrator.

Video file → frames (downsampled) → optional transcript → base caption
→ four styled captions → JSON result.

One video failing never aborts a batch run; the error is recorded in
that video's result entry instead.
"""

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .captioner import Captioner
from .fireworks_client import FireworksClient
from .frame_extractor import FrameExtractor
from .transcriber import Transcriber

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def downsample_frames(frame_paths: List[str], max_frames: int) -> List[str]:
    """Evenly sample at most max_frames from the extracted frames.

    Keeps first and last frames so the beginning and end of the clip are
    always represented. Protects against vision-model image limits and
    runaway token costs on longer clips.
    """
    if len(frame_paths) <= max_frames:
        return frame_paths
    step = (len(frame_paths) - 1) / (max_frames - 1)
    indices = [round(i * step) for i in range(max_frames)]
    return [frame_paths[i] for i in indices]


class FourTakesPipeline:
    """End-to-end captioning pipeline for one or more videos."""

    def __init__(self, config: dict, prompts: dict):
        self.config = config
        self.client = FireworksClient(
            api_key=config["fireworks_api_key"],
            model=config["fireworks_caption_model"],
            mock_mode=config["mock_mode"],
        )
        self.transcriber = Transcriber(
            api_key=config["fireworks_api_key"],
            model=config["fireworks_transcription_model"],
            mock_mode=config["mock_mode"] or not config["fireworks_api_key"],
        )
        self.captioner = Captioner(self.client, prompts)
        self.extractor = FrameExtractor(
            frame_interval=config["frame_interval_seconds"]
        )

    def process_video(self, video_path: str) -> dict:
        """Run the full pipeline on a single video. Never raises."""
        video_path = str(video_path)
        video_id = Path(video_path).stem
        started = time.time()
        temp_root = Path(self.config["temp_dir"]) / video_id
        result = {
            "video_id": video_id,
            "video_path": video_path,
            "status": "ok",
            "base_caption": None,
            "captions": {},
            "metadata": {},
        }

        try:
            # 1. Extract frames
            frame_paths, frame_meta = self.extractor.extract_frames(
                video_path, output_dir=str(temp_root / "frames")
            )
            if not frame_paths:
                raise RuntimeError("No frames extracted")

            sampled = downsample_frames(frame_paths, self.config["max_frames"])
            logger.info(
                "Video %s: %d frames extracted, %d sent to model",
                video_id, len(frame_paths), len(sampled),
            )

            # 2. Optional audio transcription (best-effort)
            transcript = None
            if self.config["enable_audio_transcription"]:
                try:
                    audio_path, _ = self.extractor.extract_audio(
                        video_path, output_dir=str(temp_root / "audio")
                    )
                    transcript = self.transcriber.transcribe(audio_path)
                except Exception as exc:
                    logger.warning(
                        "Audio step failed for %s (continuing without): %s",
                        video_id, exc,
                    )

            # 3. Base caption (generated once, reused by all styles)
            base_caption = self.captioner.generate_base_caption(
                sampled, transcript=transcript
            )
            result["base_caption"] = base_caption

            # 4. Four styled captions
            result["captions"] = self.captioner.generate_all_styles(base_caption)

            result["metadata"] = {
                "model_used": self.config["fireworks_caption_model"],
                "mock_mode": self.client.mock_mode,
                "duration_seconds": frame_meta.get("duration_seconds"),
                "frames_extracted": len(frame_paths),
                "frames_sent_to_model": len(sampled),
                "frame_interval_seconds": self.config["frame_interval_seconds"],
                "audio_transcribed": transcript is not None,
                "processing_seconds": round(time.time() - started, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as exc:
            logger.error("Pipeline failed for %s: %s", video_path, exc)
            result["status"] = "error"
            result["error"] = str(exc)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

        return result

    def process_path(self, input_path: str) -> List[dict]:
        """Process a single video file or every video in a directory."""
        path = Path(input_path)
        if path.is_dir():
            videos = sorted(
                p for p in path.iterdir()
                if p.suffix.lower() in VIDEO_EXTENSIONS
            )
            if not videos:
                raise FileNotFoundError(
                    f"No video files found in directory: {input_path}"
                )
            logger.info("Processing %d videos from %s", len(videos), input_path)
            return [self.process_video(v) for v in videos]

        if path.is_file():
            return [self.process_video(path)]

        raise FileNotFoundError(f"Input path not found: {input_path}")

    def write_results(self, results: List[dict], output_dir: str) -> List[str]:
        """Write one JSON file per video plus a combined results file."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        written = []

        for result in results:
            per_video = out / f"{result['video_id']}.json"
            with open(per_video, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            written.append(str(per_video))

        combined = out / "all_results.json"
        payload = {
            "results": results,
            "api_usage": self.client.usage_summary(),
        }
        with open(combined, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        written.append(str(combined))

        logger.info("Wrote %d result files to %s", len(written), output_dir)
        return written
