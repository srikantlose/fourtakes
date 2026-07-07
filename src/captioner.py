"""Caption generation: one neutral base caption per video, then style
transformations built from that single base caption.

The base caption is generated ONCE and reused for all styles so the
outputs stay factually consistent and vision-call costs stay low.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from .fireworks_client import FireworksClient, MOCK_STYLE_RESPONSES

logger = logging.getLogger(__name__)

# Exact style keys from the Track 2 submission guide.
STYLE_KEYS = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]


class Captioner:
    """Generates the base caption and the styled captions.

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
        """Transform the base caption into the requested style.

        Styles without a dedicated prompt fall back to the generic_style
        template so unexpected style names still get a real attempt.
        """
        style_config = self.prompts["styles"].get(style)
        if style_config is not None:
            prompt = style_config["prompt"].format(description=base_caption)
        else:
            generic = self.prompts.get("generic_style")
            if not generic:
                raise ValueError(
                    f"No prompt for style '{style}' and no generic_style template"
                )
            logger.warning("Unknown style '%s' — using generic template", style)
            prompt = generic["prompt"].format(
                style_name=style.replace("_", " "),
                description=base_caption,
            )
        messages = [{"role": "user", "content": prompt}]

        logger.info("Generating '%s' caption", style)
        caption = self.client.chat_completion(
            messages,
            max_tokens=300,
            temperature=0.9,
            mock_response=MOCK_STYLE_RESPONSES.get(style),
        )
        return caption.strip()

    def generate_all_styles(
        self,
        base_caption: str,
        styles: Optional[List[str]] = None,
    ) -> Dict[str, Optional[str]]:
        """Generate styled captions for the requested styles in parallel.

        The style calls are independent text transforms, so running them
        concurrently cuts per-clip latency to roughly one call's worth.

        A failed style maps to None; the caller decides the fallback
        (the pipeline substitutes the base caption).
        """
        styles = list(styles) if styles else list(STYLE_KEYS)
        results: Dict[str, Optional[str]] = {s: None for s in styles}

        def _one(style: str) -> None:
            try:
                results[style] = self.generate_styled_caption(style, base_caption)
            except Exception as exc:
                logger.error("Style '%s' failed: %s", style, exc)

        with ThreadPoolExecutor(max_workers=min(4, len(styles))) as pool:
            list(pool.map(_one, styles))
        return results
