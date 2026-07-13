"""Tests for the local demo web UI (webapp/). Skipped entirely if Flask
isn't installed — the core submission pipeline has no dependency on it.
"""

import io
import json
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import flask  # noqa: F401
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


@unittest.skipUnless(FLASK_AVAILABLE, "flask not installed (webapp is a dev-only extra)")
class TestWebappCaptionFlow(unittest.TestCase):
    def setUp(self):
        from webapp import server

        self.server = server
        self.server.app.config["TESTING"] = True
        self.client = self.server.app.test_client()

    def _drain_events(self, job_id, timeout=10):
        """Poll the in-memory job queue directly instead of parsing the
        streamed HTTP response, since the werkzeug test client doesn't
        stream a live generator the way a real browser would."""
        events = []
        deadline = time.time() + timeout
        q = self.server.JOBS.get(job_id)
        self.assertIsNotNone(q, "job queue missing right after submission")
        while time.time() < deadline:
            try:
                event = q.get(timeout=0.5)
            except Exception:
                continue
            if event is None:
                break
            events.append(event)
        return events

    def test_mock_run_produces_all_four_styles(self):
        fake_frames = [f"frame_{i:04d}.jpg" for i in range(10)]
        fake_meta = {"duration_seconds": 12.0}

        # The patch must stay active until the background worker thread
        # (spawned by the POST handler) actually calls extract_frames —
        # exiting the `with` block right after the POST returns would
        # unpatch before that thread gets scheduled.
        with patch.object(
            self.server.FrameExtractor, "extract_frames",
            return_value=(fake_frames, fake_meta),
        ):
            data = {
                "video": (io.BytesIO(b"fake video bytes"), "clip.mp4"),
                "mock": "true",
            }
            resp = self.client.post(
                "/api/caption", data=data, content_type="multipart/form-data"
            )
            self.assertEqual(resp.status_code, 200)
            job_id = resp.get_json()["job_id"]
            events = self._drain_events(job_id)

        types = [e["type"] for e in events]

        self.assertIn("frames_done", types)
        self.assertIn("base_done", types)

        style_events = [e for e in events if e["type"] == "style_done"]
        self.assertEqual(
            {e["style"] for e in style_events},
            {"formal", "sarcastic", "humorous_tech", "humorous_non_tech"},
        )
        for e in style_events:
            self.assertTrue(e["text"].strip())

        self.assertIn("done", types)
        self.assertNotIn("error", types)

    def test_missing_file_returns_400(self):
        resp = self.client.post("/api/caption", data={}, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)

    def test_unknown_job_id_returns_404(self):
        resp = self.client.get("/api/events/does-not-exist")
        self.assertEqual(resp.status_code, 404)

    def test_extraction_failure_emits_error_event(self):
        with patch.object(
            self.server.FrameExtractor, "extract_frames",
            side_effect=RuntimeError("ffmpeg exploded"),
        ):
            data = {
                "video": (io.BytesIO(b"fake video bytes"), "clip.mp4"),
                "mock": "true",
            }
            resp = self.client.post(
                "/api/caption", data=data, content_type="multipart/form-data"
            )
            job_id = resp.get_json()["job_id"]
            events = self._drain_events(job_id)

        error_events = [e for e in events if e["type"] == "error"]
        self.assertEqual(len(error_events), 1)
        self.assertIn("ffmpeg exploded", error_events[0]["message"])


if __name__ == "__main__":
    unittest.main()
