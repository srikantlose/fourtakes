import json
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_PROMPTS_PATH = PROJECT_ROOT / "config" / "prompts.json"


def load_config():
    """Load configuration from .env file and environment variables.

    Model names are always read from the environment — never hardcoded —
    so launch-day model swaps require only an env var change.
    """
    env_path = PROJECT_ROOT / ".env.local"
    if not env_path.exists():
        env_path = PROJECT_ROOT / ".env"

    if env_path.exists():
        load_dotenv(env_path)

    return {
        "fireworks_api_key": os.getenv("FIREWORKS_API_KEY", ""),
        "fireworks_caption_model": os.getenv(
            "FIREWORKS_CAPTION_MODEL",
            "accounts/fireworks/models/phi-4-vision-128k"
        ),
        "fireworks_transcription_model": os.getenv(
            "FIREWORKS_TRANSCRIPTION_MODEL", "whisper-v3"
        ),
        "mock_mode": os.getenv("MOCK_MODE", "false").lower() == "true",
        "frame_interval_seconds": float(
            os.getenv("FRAME_INTERVAL_SECONDS", "1.5")
        ),
        "max_frames": int(os.getenv("MAX_FRAMES", "16")),
        "enable_audio_transcription": os.getenv(
            "ENABLE_AUDIO_TRANSCRIPTION", "true"
        ).lower() == "true",
        "prompts_path": os.getenv("PROMPTS_PATH", str(DEFAULT_PROMPTS_PATH)),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "log_file": os.getenv("LOG_FILE", "fourtakes.log"),
        "output_dir": os.getenv("OUTPUT_DIR", "output"),
        "temp_dir": os.getenv("TEMP_DIR", ".temp"),
    }


def load_prompts(prompts_path: str = None) -> dict:
    """Load the base-caption and style prompts from the prompts JSON file."""
    path = Path(prompts_path or DEFAULT_PROMPTS_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    config = load_config()
    for key, value in config.items():
        if "key" not in key.lower():
            print(f"{key}: {value}")
        else:
            print(f"{key}: {'***' if value else '(not set)'}")
