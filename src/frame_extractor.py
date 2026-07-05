import subprocess
import json
import tempfile
from pathlib import Path
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extract frames from video files using ffmpeg."""

    def __init__(self, frame_interval: float = 1.5):
        """
        Initialize frame extractor.

        Args:
            frame_interval: Seconds between extracted frames
        """
        self.frame_interval = frame_interval

    def get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds using ffprobe."""
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1:nokey=1",
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception as e:
            logger.error(f"Error getting video duration: {e}")
        return 0.0

    def get_video_info(self, video_path: str) -> dict:
        """Get video metadata (duration, fps, resolution)."""
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_format",
                "-show_streams",
                "-of", "json",
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return json.loads(result.stdout)
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
        return {}

    def extract_frames(self, video_path: str, output_dir: str = None) -> Tuple[List[str], dict]:
        """
        Extract frames from video at specified interval.

        Args:
            video_path: Path to input video file
            output_dir: Directory to save frames (temp if None)

        Returns:
            Tuple of (list of frame paths, metadata dict)
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        duration = self.get_video_duration(video_path)
        if duration == 0:
            raise ValueError(f"Could not determine video duration: {video_path}")

        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="fourtakes_frames_")
        else:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Extracting frames from {video_path} (duration: {duration:.1f}s)")

        frame_pattern = str(Path(output_dir) / "frame_%04d.jpg")
        fps = 1.0 / self.frame_interval

        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"fps={fps}",
            "-q:v", "2",
            frame_pattern
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg error: {result.stderr}")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Frame extraction timed out for {video_path}")

        frame_files = sorted(Path(output_dir).glob("frame_*.jpg"))
        frame_paths = [str(f) for f in frame_files]

        metadata = {
            "video_path": video_path,
            "duration_seconds": duration,
            "frame_interval": self.frame_interval,
            "frames_extracted": len(frame_paths),
            "output_dir": output_dir,
        }

        logger.info(
            f"Extracted {len(frame_paths)} frames from {video_path} "
            f"(interval: {self.frame_interval}s)"
        )

        return frame_paths, metadata

    def extract_audio(self, video_path: str, output_dir: str = None) -> Tuple[str, dict]:
        """
        Extract audio from video using ffmpeg.

        Args:
            video_path: Path to input video file
            output_dir: Directory to save audio (temp if None)

        Returns:
            Tuple of (audio file path, metadata dict)
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="fourtakes_audio_")
        else:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        audio_path = str(Path(output_dir) / "audio.wav")

        logger.info(f"Extracting audio from {video_path}")

        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-q:a", "9",
            "-n",
            audio_path
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0 and "File already exists" not in result.stderr:
                raise RuntimeError(f"ffmpeg error: {result.stderr}")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Audio extraction timed out for {video_path}")

        metadata = {
            "video_path": video_path,
            "audio_path": audio_path,
        }

        logger.info(f"Extracted audio to {audio_path}")

        return audio_path, metadata
