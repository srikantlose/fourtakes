"""Caption generation: one neutral base caption per video, then four
style transformations built from that single base caption.

The base caption is generated ONCE and reused for all styles so the four
outputs stay factually consistent and vision-call costs stay low.
"""

import logging
from typing import Dict, List, Optional

from .fireworks_client import FireworksClient, MOCK_STYLE_RESPONSES

logger = logging.getLogger(__name__)

STYLE_KEYS = ["formal", "sarcastic", "humorous_tech", "humorous_nontech"]


class Captioner:
    """Generates the base caption and the four styled captions.

    Args:
        client: Configured FireworksClient (real or mock).
        prompts: Parsed prompts.json dict with 'base_caption' and 'styles'.
    """

    def __init__(self, client: FireworksClient, prompts: dict):
        self.client = client
        self.prompts = prompts

        missing = [k for k in STYLE_KEYS if k not in prompts.get("styles", {})]
        if missing:
            raise ValueError(f"prompts.json missing style prompts: {missing}")

    def generate_base_caption(
        self,
        frame_paths: List[str],
        transcript: Optional[str] = None,
    ) -> str:
        """Generate the neutral factual description from video frames.

        Args:
            frame_paths: Chronologically ordered frame image paths
                (already downsampled to the configured max).
            transcript: Optional audio transcript for extra context.

        Returns:
            Neutral description text.
        """
        prompt = self.prompts["base_caption"]["prompt"]
        messages = self.client.build_vision_messages(
            prompt=prompt,
            image_paths=frame_paths,
            transcript=transcript,
        )
        logger.info(
            "Generating base caption from %d frames (transcript: %s)",
            len(frame_paths),
            "yes" if transcript else "no",
        )
        caption = self.client.chat_completion(messages, max_tokens=400)
        return caption.strip()

    def generate_styled_caption(self, style: str, base_caption: str) -> str:
        """Transform the base caption into one of the four styles."""
        style_config = self.prompts["styles"][style]
        prompt = style_config["prompt"].format(description=base_caption)
        messages = [{"role": "user", "content": prompt}]

        logger.info("Generating '%s' caption", style)
        caption = self.client.chat_completion(
            messages,
            max_tokens=300,
            temperature=0.9,
            mock_response=MOCK_STYLE_RESPONSES.get(style),
        )
        return caption.strip()

    def generate_all_styles(self, base_caption: str) -> Dict[str, str]:
        """Generate all four styled captions from one base caption.

        A failure in one style records an error string for that style
        rather than aborting the other three.
        """
        results = {}
        for style in STYLE_KEYS:
            try:
                results[style] = self.generate_styled_caption(style, base_caption)
            except Exception as exc:
                logger.error("Style '%s' failed: %s", style, exc)
                results[style] = f"[ERROR: caption generation failed: {exc}]"
        return results
