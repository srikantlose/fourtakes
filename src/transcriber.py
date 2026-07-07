"""Audio transcription via the Fireworks-hosted Whisper endpoint.

Uses the same Fireworks API key as the caption model, so no separate
OpenAI account is needed. The transcription model is configurable via
FIREWORKS_TRANSCRIPTION_MODEL (default: whisper-v3).

Transcription is best-effort: any failure returns None so the caption
pipeline continues on frames alone rather than crashing.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import requests

from .fireworks_client import DEFAULT_BASE_URL

logger = logging.getLogger(__name__)

MOCK_TRANSCRIPT = (
    "Good catch, buddy! Go get it! [dog barking] That's a good boy."
)


class Transcriber:
    """Transcribe audio files using Fireworks-hosted Whisper.

    Args:
        api_key: Fireworks API key. Empty string enables mock mode.
        model: Transcription model name from config.
        base_url: OpenAI-compatible API base URL (see FireworksClient).
        mock_mode: Force mock transcripts even if an API key is set.
        max_retries: Retry attempts per call.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "whisper-v3",
        base_url: str = DEFAULT_BASE_URL,
        mock_mode: bool = False,
        max_retries: int = 2,
        timeout: int = 120,
    ):
        self.api_key = api_key
        self.model = model
        self.transcription_url = f"{base_url.rstrip('/')}/audio/transcriptions"
        self.mock_mode = mock_mode or not api_key
        self.max_retries = max_retries
        self.timeout = timeout

        if self.mock_mode:
            logger.warning("Transcriber running in MOCK MODE")

    def transcribe(self, audio_path: str) -> Optional[str]:
        """Transcribe an audio file. Returns None on failure (best-effort).

        Args:
            audio_path: Path to a .wav/.mp3 audio file.

        Returns:
            Transcript text, empty string for silent clips, or None if
            transcription failed entirely.
        """
        if self.mock_mode:
            logger.info("Mock transcription for %s", audio_path)
            return MOCK_TRANSCRIPT

        if not Path(audio_path).exists():
            logger.error("Audio file not found: %s", audio_path)
            return None

        headers = {"Authorization": f"Bearer {self.api_key}"}

        for attempt in range(1, self.max_retries + 1):
            try:
                with open(audio_path, "rb") as f:
                    response = requests.post(
                        self.transcription_url,
                        headers=headers,
                        files={"file": f},
                        data={"model": self.model},
                        timeout=self.timeout,
                    )
                if response.status_code == 200:
                    text = response.json().get("text", "").strip()
                    logger.info(
                        "Transcribed %s (%d chars)", audio_path, len(text)
                    )
                    return text

                logger.warning(
                    "Transcription attempt %d failed: HTTP %d: %s",
                    attempt,
                    response.status_code,
                    response.text[:300],
                )
                if response.status_code not in (429, 500, 502, 503, 504):
                    break
            except requests.RequestException as exc:
                logger.warning(
                    "Transcription attempt %d raised: %s", attempt, exc
                )

            if attempt < self.max_retries:
                time.sleep(2 ** attempt)

        logger.error(
            "Transcription failed for %s — continuing without audio context",
            audio_path,
        )
        return None
