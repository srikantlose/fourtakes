"""Submission-mode runner for the Track 2 judging harness.

Contract (from the participant guide):
  - Read tasks from /input/tasks.json:
      [{"task_id": "v1", "video_url": "https://...", "styles": [...]}, ...]
  - Write results to /output/results.json before exiting:
      [{"task_id": "v1", "captions": {"formal": "...", ...}}, ...]
  - Exit 0 on success. Every requested style must have a caption
    (missing styles score zero; malformed JSON scores the whole run zero).

Design rules:
  - One bad task never breaks the others (per-task isolation).
  - Fallback ladder per style: styled call fails -> base caption;
    whole video fails -> generic neutral caption per style.
  - Tasks run concurrently to fit the 10-minute runtime budget.
"""

import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import requests

from .captioner import STYLE_KEYS
from .pipeline import FourTakesPipeline

logger = logging.getLogger(__name__)

# Last-resort captions when the video could not be processed at all
# (download failed, undecodable, vision call dead after retries).
# Deliberately content-free error handling — not cached answers.
GENERIC_FALLBACK_CAPTION = (
    "A short video clip showing a scene that unfolds over the course of "
    "the recording."
)


def load_tasks(tasks_path: str) -> List[dict]:
    """Parse tasks.json; tolerate a missing styles list (default all four).

    utf-8-sig transparently handles files with or without a UTF-8 BOM.
    """
    with open(tasks_path, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)

    tasks = []
    for i, entry in enumerate(raw):
        task_id = str(entry.get("task_id") or f"task_{i}")
        video_url = entry.get("video_url", "")
        styles = entry.get("styles") or list(STYLE_KEYS)
        tasks.append({"task_id": task_id, "video_url": video_url, "styles": styles})
    return tasks


def download_video(
    url: str,
    dest_path: str,
    timeout: int = 120,
    max_retries: int = 2,
) -> str:
    """Stream a video URL to disk with retries. Returns the local path."""
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 2):  # initial try + retries
        start = time.time()
        try:
            with requests.get(url, stream=True, timeout=(10, timeout)) as resp:
                resp.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            size_mb = Path(dest_path).stat().st_size / (1024 * 1024)
            logger.info(
                "Downloaded %s (%.1f MB in %.1fs)",
                url, size_mb, time.time() - start,
            )
            return dest_path
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            logger.warning(
                "Download failed (attempt %d) for %s: %s", attempt, url, exc
            )
            if attempt <= max_retries:
                time.sleep(2 * attempt)

    raise RuntimeError(f"Could not download video after retries: {last_error}")


class TaskRunner:
    """Runs the caption pipeline over a tasks.json batch."""

    def __init__(self, config: dict, prompts: dict):
        self.config = config
        self.pipeline = FourTakesPipeline(config, prompts)

    def _video_dest(self, task: dict) -> str:
        """Unique local path per task (task_id keys the temp tree, so
        concurrent tasks never collide)."""
        suffix = Path(urlparse(task["video_url"]).path).suffix or ".mp4"
        downloads = Path(self.config["temp_dir"]) / "downloads"
        return str(downloads / f"{task['task_id']}{suffix}")

    def run_task(self, task: dict) -> dict:
        """Process one task. Never raises; always returns a full captions
        dict covering every requested style."""
        task_id = task["task_id"]
        styles = task["styles"]
        video_path = self._video_dest(task)

        try:
            download_video(
                task["video_url"],
                video_path,
                timeout=self.config["download_timeout"],
            )
            result = self.pipeline.process_video(video_path, styles=styles)
            if result["status"] != "ok":
                raise RuntimeError(result.get("error", "pipeline error"))
            captions = result["captions"]
        except Exception as exc:
            logger.error("Task %s failed entirely: %s — using fallbacks", task_id, exc)
            captions = {}
        finally:
            Path(video_path).unlink(missing_ok=True)

        # Guarantee: a caption for every requested style, no error strings.
        final_captions = {}
        for style in styles:
            text = captions.get(style)
            final_captions[style] = text if text else GENERIC_FALLBACK_CAPTION

        return {"task_id": task_id, "captions": final_captions}

    def run_all(self, tasks: List[dict]) -> List[dict]:
        """Run all tasks concurrently, preserving input order in results."""
        workers = max(1, min(self.config["max_concurrent_tasks"], len(tasks)))
        logger.info("Running %d task(s) with %d worker(s)", len(tasks), workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self.run_task, tasks))


def write_results(results: List[dict], results_path: str) -> None:
    """Write results.json atomically (write temp file, then replace)."""
    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    tmp.replace(path)
    logger.info("Wrote %d result(s) to %s", len(results), results_path)


def run(config: dict, prompts: dict) -> int:
    """Entry point for submission mode. Returns the process exit code."""
    tasks_path = config["tasks_path"]
    results_path = config["results_path"]
    started = time.time()

    try:
        tasks = load_tasks(tasks_path)
    except Exception as exc:
        logger.error("Could not read tasks file %s: %s", tasks_path, exc)
        try:
            write_results([], results_path)
        except Exception:
            pass
        return 1

    runner = TaskRunner(config, prompts)
    results = runner.run_all(tasks)

    try:
        write_results(results, results_path)
    except Exception as exc:
        logger.error("Could not write results to %s: %s", results_path, exc)
        return 1
    finally:
        shutil.rmtree(
            Path(config["temp_dir"]) / "downloads", ignore_errors=True
        )

    logger.info(
        "Completed %d task(s) in %.1fs — %s",
        len(results), time.time() - started,
        runner.pipeline.client.usage_summary(),
    )
    return 0
