import os
from pathlib import Path
from dotenv import load_dotenv


def load_config():
    """Load configuration from .env file and environment variables."""
    env_path = Path(__file__).parent.parent / ".env.local"
    if not env_path.exists():
        env_path = Path(__file__).parent.parent / ".env"

    if env_path.exists():
        load_dotenv(env_path)

    return {
        "fireworks_api_key": os.getenv("FIREWORKS_API_KEY", ""),
        "fireworks_caption_model": os.getenv(
            "FIREWORKS_CAPTION_MODEL",
            "accounts/fireworks/models/phi-4-vision-128k"
        ),
        "frame_interval_seconds": float(
            os.getenv("FRAME_INTERVAL_SECONDS", "1.5")
        ),
        "enable_audio_transcription": os.getenv(
            "ENABLE_AUDIO_TRANSCRIPTION", "true"
        ).lower() == "true",
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "log_file": os.getenv("LOG_FILE", "fourtakes.log"),
        "output_dir": os.getenv("OUTPUT_DIR", "output"),
        "temp_dir": os.getenv("TEMP_DIR", ".temp"),
    }


if __name__ == "__main__":
    config = load_config()
    for key, value in config.items():
        if "key" not in key.lower():
            print(f"{key}: {value}")
        else:
            print(f"{key}: {'***' if value else '(not set)'}")
