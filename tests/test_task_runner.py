"""Tests for submission (task) mode: /input/tasks.json -> /output/results.json.

No ffmpeg, no network, no API key required: downloads and extraction are
patched out, and the Fireworks client runs in mock mode.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import task_runner
from src.captioner import STYLE_KEYS
from src.config import load_prompts
from src.task_runner import GENERIC_FALLBACK_CAPTION, TaskRunner, load_tasks

ALL_STYLES = list(STYLE_KEYS)

FAKE_FRAMES = [f"frame_{i:04d}.jpg" for i in range(10)]
FAKE_META = {"duration_seconds": 45.0}


def make_config(tmpdir: str, **overrides) -> dict:
    config = {
        "fireworks_api_key": "",
        "fireworks_caption_model": "test/model-from-config",
        "fireworks_transcription_model": "whisper-v3",
        "mock_mode": True,
        "frame_interval_seconds": 1.5,
        "max_frames": 16,
        "frame_scale_width": 512,
        "enable_audio_transcription": False,
        "tasks_path": str(Path(tmpdir) / "input" / "tasks.json"),
        "results_path": str(Path(tmpdir) / "output" / "results.json"),
        "max_concurrent_tasks": 3,
        "download_timeout": 10,
        "api_timeout": 10,
        "prompts_path": str(PROJECT_ROOT / "config" / "prompts.json"),
        "log_level": "WARNING",
        "log_file": str(Path(tmpdir) / "test.log"),
        "output_dir": str(Path(tmpdir) / "output"),
        "temp_dir": str(Path(tmpdir) / "temp"),
    }
    config.update(overrides)
    return config


def write_tasks(config: dict, tasks: list) -> None:
    path = Path(config["tasks_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tasks, f)


def fake_download(url, dest_path, **kwargs):
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(dest_path).write_bytes(b"fake video bytes")
    return dest_path


class TestLoadTasks(unittest.TestCase):
    def test_missing_styles_defaults_to_all_four(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(tmpdir)
            write_tasks(config, [{"task_id": "v1", "video_url": "http://x/v.mp4"}])
            tasks = load_tasks(config["tasks_path"])
        self.assertEqual(tasks[0]["styles"], ALL_STYLES)

    def test_missing_task_id_gets_positional_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(tmpdir)
            write_tasks(config, [{"video_url": "http://x/v.mp4"}])
            tasks = load_tasks(config["tasks_path"])
        self.assertEqual(tasks[0]["task_id"], "task_0")


class TestTaskRunnerEndToEnd(unittest.TestCase):
    """Full submission-mode run with downloads and ffmpeg patched out."""

    def _run(self, config, tasks):
        write_tasks(config, tasks)
        prompts = load_prompts()
        with patch.object(task_runner, "download_video", side_effect=fake_download), \
             patch(
                 "src.frame_extractor.FrameExtractor.extract_frames",
                 return_value=(FAKE_FRAMES, FAKE_META),
             ):
            exit_code = task_runner.run(config, prompts)
        with open(config["results_path"], encoding="utf-8") as f:
            return exit_code, json.load(f)

    def test_results_match_guide_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(tmpdir)
            exit_code, results = self._run(config, [
                {
                    "task_id": "v1",
                    "video_url": "https://example.com/clip1.mp4",
                    "styles": ALL_STYLES,
                },
            ])

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(results), 1)
        self.assertEqual(set(results[0].keys()), {"task_id", "captions"})
        self.assertEqual(results[0]["task_id"], "v1")
        self.assertEqual(set(results[0]["captions"].keys()), set(ALL_STYLES))
        self.assertIn("humorous_non_tech", results[0]["captions"])
        for text in results[0]["captions"].values():
            self.assertTrue(text.strip())

    def test_multiple_tasks_preserve_order_and_style_subsets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(tmpdir)
            exit_code, results = self._run(config, [
                {
                    "task_id": "v1",
                    "video_url": "https://example.com/a.mp4",
                    "styles": ALL_STYLES,
                },
                {
                    "task_id": "v2",
                    "video_url": "https://example.com/b.mp4",
                    "styles": ["formal", "sarcastic"],
                },
            ])

        self.assertEqual(exit_code, 0)
        self.assertEqual([r["task_id"] for r in results], ["v1", "v2"])
        self.assertEqual(
            set(results[1]["captions"].keys()), {"formal", "sarcastic"}
        )

    def test_unknown_style_still_gets_a_caption(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(tmpdir)
            exit_code, results = self._run(config, [
                {
                    "task_id": "v1",
                    "video_url": "https://example.com/a.mp4",
                    "styles": ["formal", "dramatic"],
                },
            ])

        self.assertEqual(exit_code, 0)
        captions = results[0]["captions"]
        self.assertEqual(set(captions.keys()), {"formal", "dramatic"})
        self.assertTrue(captions["dramatic"].strip())


class TestFallbacks(unittest.TestCase):
    def test_download_failure_yields_generic_captions_and_exit_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(tmpdir)
            write_tasks(config, [
                {
                    "task_id": "v1",
                    "video_url": "https://example.com/broken.mp4",
                    "styles": ALL_STYLES,
                },
            ])
            with patch.object(
                task_runner, "download_video",
                side_effect=RuntimeError("network down"),
            ):
                exit_code = task_runner.run(config, load_prompts())
            with open(config["results_path"], encoding="utf-8") as f:
                results = json.load(f)

        self.assertEqual(exit_code, 0)
        self.assertEqual(set(results[0]["captions"].keys()), set(ALL_STYLES))
        for text in results[0]["captions"].values():
            self.assertEqual(text, GENERIC_FALLBACK_CAPTION)

    def test_missing_tasks_file_writes_empty_results_and_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(tmpdir)  # tasks.json never written
            exit_code = task_runner.run(config, load_prompts())
            with open(config["results_path"], encoding="utf-8") as f:
                results = json.load(f)

        self.assertEqual(exit_code, 1)
        self.assertEqual(results, [])

    def test_unique_download_paths_per_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(tmpdir)
            runner = TaskRunner(config, load_prompts())
            dest_a = runner._video_dest(
                {"task_id": "v1", "video_url": "https://x/clip.mp4"}
            )
            dest_b = runner._video_dest(
                {"task_id": "v2", "video_url": "https://x/clip.mp4"}
            )
        self.assertNotEqual(dest_a, dest_b)
        self.assertTrue(dest_a.endswith("v1.mp4"))


if __name__ == "__main__":
    unittest.main()
