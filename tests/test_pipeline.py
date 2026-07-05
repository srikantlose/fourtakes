"""Integration tests for the FourTakes pipeline using mock mode.

No ffmpeg, no network, no API key required: the Fireworks client runs in
mock mode and frame/audio extraction is patched out.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.captioner import Captioner, STYLE_KEYS
from src.config import load_prompts
from src.fireworks_client import FireworksClient, MOCK_BASE_CAPTION
from src.pipeline import FourTakesPipeline, downsample_frames


def make_test_config(**overrides) -> dict:
    config = {
        "fireworks_api_key": "",
        "fireworks_caption_model": "test/model-from-config",
        "fireworks_transcription_model": "whisper-v3",
        "mock_mode": True,
        "frame_interval_seconds": 1.5,
        "max_frames": 16,
        "enable_audio_transcription": True,
        "prompts_path": str(PROJECT_ROOT / "config" / "prompts.json"),
        "log_level": "WARNING",
        "log_file": "test.log",
        "output_dir": "output",
        "temp_dir": tempfile.mkdtemp(prefix="fourtakes_test_"),
    }
    config.update(overrides)
    return config


class TestDownsampleFrames(unittest.TestCase):
    def test_no_downsampling_needed(self):
        frames = [f"f{i}.jpg" for i in range(10)]
        self.assertEqual(downsample_frames(frames, 16), frames)

    def test_downsamples_to_max(self):
        frames = [f"f{i}.jpg" for i in range(80)]
        sampled = downsample_frames(frames, 16)
        self.assertEqual(len(sampled), 16)

    def test_keeps_first_and_last_frame(self):
        frames = [f"f{i}.jpg" for i in range(80)]
        sampled = downsample_frames(frames, 16)
        self.assertEqual(sampled[0], frames[0])
        self.assertEqual(sampled[-1], frames[-1])


class TestPrompts(unittest.TestCase):
    def test_prompts_file_valid(self):
        prompts = load_prompts()
        self.assertIn("base_caption", prompts)
        for style in STYLE_KEYS:
            self.assertIn(style, prompts["styles"])
            self.assertIn("{description}", prompts["styles"][style]["prompt"])


class TestMockClient(unittest.TestCase):
    def test_mock_mode_when_no_api_key(self):
        client = FireworksClient(api_key="", model="test/model")
        self.assertTrue(client.mock_mode)

    def test_mock_completion_and_logging(self):
        client = FireworksClient(api_key="", model="test/model")
        text = client.chat_completion([{"role": "user", "content": "hi"}])
        self.assertEqual(text, MOCK_BASE_CAPTION)
        self.assertEqual(len(client.call_log), 1)
        self.assertEqual(client.call_log[0]["model"], "test/model")

    def test_usage_summary(self):
        client = FireworksClient(api_key="", model="test/model")
        client.chat_completion([{"role": "user", "content": "hi"}])
        summary = client.usage_summary()
        self.assertEqual(summary["calls"], 1)
        self.assertTrue(summary["mock_mode"])


class TestCaptioner(unittest.TestCase):
    def setUp(self):
        self.client = FireworksClient(api_key="", model="test/model")
        self.captioner = Captioner(self.client, load_prompts())

    def test_all_styles_generated_and_distinct(self):
        captions = self.captioner.generate_all_styles(MOCK_BASE_CAPTION)
        self.assertEqual(set(captions.keys()), set(STYLE_KEYS))
        # All four mock captions should differ from each other
        self.assertEqual(len(set(captions.values())), 4)

    def test_missing_style_prompt_raises(self):
        broken = {"base_caption": {"prompt": "x"}, "styles": {"formal": {"prompt": "y"}}}
        with self.assertRaises(ValueError):
            Captioner(self.client, broken)


class TestPipelineEndToEnd(unittest.TestCase):
    """Full pipeline run with extraction patched out and mock API."""

    def _run_pipeline(self, config):
        prompts = load_prompts()
        pipeline = FourTakesPipeline(config, prompts)

        fake_frames = [f"frame_{i:04d}.jpg" for i in range(40)]
        fake_meta = {"duration_seconds": 60.0, "frames_extracted": 40}

        with patch.object(
            pipeline.extractor, "extract_frames",
            return_value=(fake_frames, fake_meta),
        ), patch.object(
            pipeline.extractor, "extract_audio",
            return_value=("audio.wav", {}),
        ), patch.object(
            Path, "is_file", return_value=True
        ), patch.object(
            Path, "is_dir", return_value=False
        ):
            return pipeline.process_path("fake_video.mp4")

    def test_end_to_end_mock_run(self):
        results = self._run_pipeline(make_test_config())
        self.assertEqual(len(results), 1)
        result = results[0]

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["video_id"], "fake_video")
        self.assertTrue(result["base_caption"])
        self.assertEqual(set(result["captions"].keys()), set(STYLE_KEYS))

        meta = result["metadata"]
        self.assertEqual(meta["model_used"], "test/model-from-config")
        self.assertTrue(meta["mock_mode"])
        self.assertEqual(meta["frames_extracted"], 40)
        self.assertEqual(meta["frames_sent_to_model"], 16)
        self.assertTrue(meta["audio_transcribed"])

    def test_audio_disabled(self):
        config = make_test_config(enable_audio_transcription=False)
        results = self._run_pipeline(config)
        self.assertFalse(results[0]["metadata"]["audio_transcribed"])

    def test_write_results(self):
        config = make_test_config()
        results = self._run_pipeline(config)

        pipeline = FourTakesPipeline(config, load_prompts())
        with tempfile.TemporaryDirectory() as tmpdir:
            written = pipeline.write_results(results, tmpdir)
            self.assertEqual(len(written), 2)  # per-video + combined

            with open(written[0], encoding="utf-8") as f:
                per_video = json.load(f)
            self.assertEqual(per_video["video_id"], "fake_video")

            with open(written[1], encoding="utf-8") as f:
                combined = json.load(f)
            self.assertIn("api_usage", combined)

    def test_extraction_failure_is_contained(self):
        config = make_test_config()
        pipeline = FourTakesPipeline(config, load_prompts())
        with patch.object(
            pipeline.extractor, "extract_frames",
            side_effect=RuntimeError("ffmpeg exploded"),
        ):
            result = pipeline.process_video("broken.mp4")
        self.assertEqual(result["status"], "error")
        self.assertIn("ffmpeg exploded", result["error"])


if __name__ == "__main__":
    unittest.main()
