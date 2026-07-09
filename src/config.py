import json
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_PROMPTS_PATH = PROJECT_ROOT / "config" / "prompts.json"


def load_config():
    """Load configuration from .env file and environment variables.

    Model names are always read from the environment — never hardcoded in
    pipeline logic — so swapping models is only a config change.

    Empty-string env values fall through to the defaults (`or` pattern), so
    a Docker build without --build-arg overrides still gets sane defaults.
    """
    env_path = PROJECT_ROOT / ".env.local"
    if not env_path.exists():
        env_path = PROJECT_ROOT / ".env"

    if env_path.exists():
        # override=True so .env.local always wins over a stale exported
        # shell var during local dev. Harmless in the judging container:
        # .env.local is gitignored and never copied into the image, so
        # the harness's injected env vars are the only source there.
        load_dotenv(env_path, override=True)

    return {
        "fireworks_api_key": os.getenv("FIREWORKS_API_KEY", ""),
        # The judging harness injects FIREWORKS_BASE_URL at runtime (its
        # token-recording proxy). Empty/unset falls through to the public
        # Fireworks endpoint for local development.
        "fireworks_base_url": (
            os.getenv("FIREWORKS_BASE_URL")
            or "https://api.fireworks.ai/inference/v1"
        ).rstrip("/"),
        "fireworks_caption_model": os.getenv("FIREWORKS_CAPTION_MODEL")
        or "accounts/fireworks/models/qwen3p7-plus",
        "fireworks_transcription_model": os.getenv("FIREWORKS_TRANSCRIPTION_MODEL")
        or "whisper-v3",
        "mock_mode": os.getenv("MOCK_MODE", "false").lower() == "true",
        "frame_interval_seconds": float(
            os.getenv("FRAME_INTERVAL_SECONDS", "1.5")
        ),
        "max_frames": int(os.getenv("MAX_FRAMES", "16")),
        "frame_scale_width": int(os.getenv("FRAME_SCALE_WIDTH", "512")),
        "enable_audio_transcription": os.getenv(
            "ENABLE_AUDIO_TRANSCRIPTION", "true"
        ).lower() == "true",
        # Submission (task) mode: the judging harness mounts these paths.
        "tasks_path": os.getenv("TASKS_PATH", "/input/tasks.json"),
        "results_path": os.getenv("RESULTS_PATH", "/output/results.json"),
        "max_concurrent_tasks": int(os.getenv("MAX_CONCURRENT_TASKS", "3")),
        "download_timeout": int(os.getenv("DOWNLOAD_TIMEOUT", "120")),
        "api_timeout": int(os.getenv("API_TIMEOUT", "60")),
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
