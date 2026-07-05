import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from frame_extractor import FrameExtractor


class TestFrameExtractor(unittest.TestCase):
    """Tests for FrameExtractor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = FrameExtractor(frame_interval=1.5)

    def test_initialization(self):
        """Test FrameExtractor initialization."""
        self.assertEqual(self.extractor.frame_interval, 1.5)

    def test_get_video_duration_not_found(self):
        """Test get_video_duration with non-existent file."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="File not found")
            duration = self.extractor.get_video_duration("nonexistent.mp4")
            self.assertEqual(duration, 0.0)

    @patch("subprocess.run")
    def test_get_video_duration_success(self, mock_run):
        """Test successful video duration retrieval."""
        mock_run.return_value = MagicMock(returncode=0, stdout="120.5\n")
        duration = self.extractor.get_video_duration("test_video.mp4")
        self.assertEqual(duration, 120.5)

    @patch("subprocess.run")
    def test_get_video_info(self, mock_run):
        """Test video info retrieval."""
        test_info = {
            "format": {"duration": "120.5"},
            "streams": [{"width": 1920, "height": 1080}]
        }
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(test_info)
        )
        info = self.extractor.get_video_info("test_video.mp4")
        self.assertEqual(info["format"]["duration"], "120.5")

    def test_extract_frames_file_not_found(self):
        """Test extract_frames with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.extractor.extract_frames("nonexistent.mp4")

    @patch("subprocess.run")
    def test_extract_frames_duration_zero(self, mock_run):
        """Test extract_frames when duration is zero."""
        mock_run.return_value = MagicMock(returncode=1)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_file = f.name

        try:
            with self.assertRaises(ValueError):
                self.extractor.extract_frames(temp_file)
        finally:
            Path(temp_file).unlink(missing_ok=True)

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_extract_frames_success(self, mock_exists, mock_run):
        """Test successful frame extraction."""
        mock_exists.return_value = True

        # Mock ffprobe (duration)
        # Mock ffmpeg (frame extraction)
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="60.0\n"),  # ffprobe duration
            MagicMock(returncode=0, stderr=""),        # ffmpeg extraction
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                temp_video = f.name

            try:
                # Create mock frame files
                output_dir = tempfile.mkdtemp()
                for i in range(40):
                    Path(output_dir, f"frame_{i+1:04d}.jpg").touch()

                with patch("tempfile.mkdtemp", return_value=output_dir):
                    with patch("pathlib.Path.glob") as mock_glob:
                        frame_files = [
                            Path(output_dir, f"frame_{i+1:04d}.jpg")
                            for i in range(40)
                        ]
                        mock_glob.return_value = frame_files

                        frames, metadata = self.extractor.extract_frames(temp_video)

                        self.assertEqual(len(frames), 40)
                        self.assertEqual(metadata["duration_seconds"], 60.0)
                        self.assertEqual(metadata["frame_interval"], 1.5)
                        self.assertIn("frames_extracted", metadata)
            finally:
                Path(temp_video).unlink(missing_ok=True)

    @patch("subprocess.run")
    def test_extract_audio_file_not_found(self, mock_run):
        """Test extract_audio with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.extractor.extract_audio("nonexistent.mp4")

    @patch("subprocess.run")
    def test_extract_audio_timeout(self, mock_run):
        """Test extract_audio with timeout."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=300)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_file = f.name

        try:
            with self.assertRaises(RuntimeError):
                self.extractor.extract_audio(temp_file)
        finally:
            Path(temp_file).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
