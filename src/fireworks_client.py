"""Fireworks AI API client with retry logic, call logging, and a mock mode.

The model name is NEVER hardcoded here — it is passed in from config
(FIREWORKS_CAPTION_MODEL env var) so launch-day model swaps require no
code changes.

Mock mode (MOCK_MODE=true or missing API key) returns realistic canned
responses so the full pipeline can be built and tested without spending
API credits.
"""

import base64
import logging
import time
from pathlib import Path
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"

MOCK_BASE_CAPTION = (
    "A golden retriever runs across a grassy park chasing a red frisbee, "
    "while two people watch from a wooden bench under a large oak tree. "
    "The dog catches the frisbee mid-air and trots back toward the people."
)

MOCK_STYLE_RESPONSES = {
    "formal": (
        "A canine of the golden retriever breed traverses a public green space "
        "in pursuit of a recreational disc, successfully intercepting the object "
        "mid-flight before returning to its accompanying party."
    ),
    "sarcastic": (
        "Ah yes, a dog fetching a frisbee. Groundbreaking footage. Truly, no one "
        "could have predicted the dog would... bring it back. Riveting stuff."
    ),
    "humorous_tech": (
        "Dog executes fetch() in a single thread, zero latency, catches the "
        "frisbee mid-air with 100% uptime, then returns the payload to the "
        "clients on the bench. Ship it."
    ),
    "humorous_non_tech": (
        "This good boy just won gold in the Backyard Olympics! One leap, one "
        "catch, and a victory lap back to his adoring fans on the bench. "
        "Ten out of ten, would fetch again."
    ),
}


class FireworksAPIError(Exception):
    """Raised when the Fireworks API fails after all retries."""


class FireworksClient:
    """Thin wrapper over the Fireworks OpenAI-compatible chat API.

    Args:
        api_key: Fireworks API key. Empty string enables mock mode.
        model: Model identifier from config (never hardcoded).
        base_url: OpenAI-compatible API base URL. The judging harness
            injects its own via FIREWORKS_BASE_URL; defaults to the
            public Fireworks endpoint.
        mock_mode: Force mock responses even if an API key is set.
        max_retries: Retry attempts per call.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        mock_mode: bool = False,
        max_retries: int = 3,
        timeout: int = 120,
    ):
        self.api_key = api_key
        self.model = model
        self.chat_url = f"{base_url.rstrip('/')}/chat/completions"
        self.mock_mode = mock_mode or not api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self.call_log: List[dict] = []

        if self.mock_mode:
            logger.warning(
                "FireworksClient running in MOCK MODE — no real API calls "
                "will be made (api_key %s, mock_mode=%s)",
                "set" if api_key else "missing",
                mock_mode,
            )

    @staticmethod
    def encode_image(image_path: str) -> str:
        """Encode an image file as a base64 data URI for vision prompts."""
        data = Path(image_path).read_bytes()
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def build_vision_messages(
        self,
        prompt: str,
        image_paths: List[str],
        transcript: Optional[str] = None,
    ) -> List[dict]:
        """Build a chat message list containing text + multiple frames."""
        text = prompt
        if transcript:
            text += (
                "\n\nAudio transcript of the clip (may add useful context):\n"
                f"{transcript}"
            )

        content = [{"type": "text", "text": text}]
        for path in image_paths:
            # In mock mode, skip reading/encoding files so the pipeline can
            # run without real frames on disk (and without wasted work).
            url = f"mock://{path}" if self.mock_mode else self.encode_image(path)
            content.append(
                {"type": "image_url", "image_url": {"url": url}}
            )
        return [{"role": "user", "content": content}]

    def chat_completion(
        self,
        messages: List[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        mock_response: Optional[str] = None,
    ) -> str:
        """Send a chat completion request with retries.

        Args:
            messages: OpenAI-style message list.
            max_tokens: Response token cap.
            temperature: Sampling temperature.
            mock_response: Response to return in mock mode.

        Returns:
            The assistant's text response.

        Raises:
            FireworksAPIError: after all retries fail.
        """
        if self.mock_mode:
            return self._mock_completion(messages, mock_response)

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            start = time.time()
            try:
                response = requests.post(
                    self.chat_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                latency = time.time() - start

                if response.status_code == 200:
                    data = response.json()
                    text = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    self._log_call(
                        status="success",
                        latency=latency,
                        attempt=attempt,
                        prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"),
                        total_tokens=usage.get("total_tokens"),
                    )
                    return text

                # Retry on rate limits and server errors; fail fast otherwise
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                self._log_call(
                    status="http_error",
                    latency=latency,
                    attempt=attempt,
                    error=last_error,
                )
                if response.status_code not in (429, 500, 502, 503, 504):
                    break

            except requests.RequestException as exc:
                latency = time.time() - start
                last_error = str(exc)
                self._log_call(
                    status="exception",
                    latency=latency,
                    attempt=attempt,
                    error=last_error,
                )

            if attempt < self.max_retries:
                backoff = 2 ** attempt  # 2s, 4s, 8s...
                logger.warning(
                    "Fireworks call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt,
                    self.max_retries,
                    last_error,
                    backoff,
                )
                time.sleep(backoff)

        raise FireworksAPIError(
            f"Fireworks API failed after {self.max_retries} attempts: {last_error}"
        )

    def _mock_completion(
        self, messages: List[dict], mock_response: Optional[str]
    ) -> str:
        """Return a canned response and log the call like a real one."""
        self._log_call(status="mock", latency=0.0, attempt=1)
        if mock_response is not None:
            return mock_response
        return MOCK_BASE_CAPTION

    def _log_call(self, **fields) -> None:
        """Record one API call for debugging and demo reporting."""
        entry = {
            "model": self.model,
            "mock": self.mock_mode,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **fields,
        }
        self.call_log.append(entry)
        logger.info("Fireworks call: %s", entry)

    def usage_summary(self) -> dict:
        """Aggregate token usage across all calls in this session."""
        total = sum(e.get("total_tokens") or 0 for e in self.call_log)
        return {
            "model": self.model,
            "mock_mode": self.mock_mode,
            "calls": len(self.call_log),
            "total_tokens": total,
            "errors": sum(
                1 for e in self.call_log if e.get("status") not in ("success", "mock")
            ),
        }
