# FourTakes: Video Captioning Agent

Generates captions in four distinct voices — **formal**, **sarcastic**, **humorous_tech**, and **humorous_non_tech** — for short video clips (30 sec–2 min), using vision models via the Fireworks AI API.

Built for AMD Developer Hackathon: ACT II, Track 2 (Video Captioning).

## How It Works

```
video (URL or local file)
  → download (submission mode) or read local file
  → extract frames with ffmpeg (every 1.5s, downscaled to ≤512px wide)
  → downsample to MAX_FRAMES (evenly, keeping first + last frame)
  → optional: extract audio + transcribe via Fireworks-hosted Whisper
    (off by default — see note below)
  → ONE vision-model call: neutral factual base caption
  → parallel text-model calls: transform base caption into each requested style
  → results JSON
```

The base caption is generated **once** per video and reused for all styles, so the outputs stay factually consistent and vision-call costs stay low.

## Submission Mode (the judged path)

The container follows the Track 2 harness contract:

1. Reads `/input/tasks.json`:
```json
[
  {
    "task_id": "v1",
    "video_url": "https://storage.example.com/clips/clip1.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  }
]
```
2. Writes `/output/results.json` before exiting (exit code 0):
```json
[
  {
    "task_id": "v1",
    "captions": {
      "formal": "...",
      "sarcastic": "...",
      "humorous_tech": "...",
      "humorous_non_tech": "..."
    }
  }
]
```

Robustness guarantees:
- **Every requested style always gets a caption.** A failed style call falls back to the base caption; a fully failed video falls back to a generic caption. Missing styles score zero, so nothing is ever left out.
- **One bad clip never breaks the batch** — tasks are isolated and run concurrently (`MAX_CONCURRENT_TASKS`) to stay inside the 10-minute runtime limit.
- **Unknown style names still work** via a generic style-prompt template.
- `results.json` is always valid JSON, written atomically.

## Key Design Rules

- **No hardcoded model names.** The caption model comes from `FIREWORKS_CAPTION_MODEL` (env var, build arg, or `--model` flag). Swapping models is a config change, not a code change.
- **Mock mode** (`MOCK_MODE=true` or `--mock`): the entire pipeline runs with canned API responses — no key, no credits, no network. Used for development and CI.
- **UHD-safe**: frames are downscaled to `FRAME_SCALE_WIDTH` (512px) before encoding, and at most `MAX_FRAMES` (16) are sent per video.
- **Best-effort audio**: transcription failures never break captioning. Off by default everywhere — Fireworks has deprecated hosted audio inference platform-wide, so `/audio/transcriptions` currently returns 401 for every key. The code path is kept in case Fireworks reinstates it.
- **Every API call is logged** with model name, tokens, latency, and status.

## Project Structure

```
fourtakes/
├── src/
│   ├── config.py            # env-based config + prompts loader
│   ├── frame_extractor.py   # ffmpeg frame/audio extraction (with downscale)
│   ├── fireworks_client.py  # Fireworks API wrapper (retries, logging, mock mode)
│   ├── transcriber.py       # Fireworks-hosted Whisper transcription
│   ├── captioner.py         # base caption + parallel style transforms
│   ├── pipeline.py          # per-video orchestrator + style fallbacks
│   ├── task_runner.py       # submission mode: tasks.json → results.json
│   ├── logging_config.py    # file + console logging
│   └── main.py              # CLI entrypoint (submission + dev modes)
├── config/
│   └── prompts.json         # ALL prompts (base + styles + generic) — edit here
├── tests/                   # 49 tests, all runnable offline
├── Dockerfile
├── .env.template            # copy to .env.local and fill in
└── requirements.txt
```

## Setup

Prerequisites: Python 3.9+, ffmpeg/ffprobe on PATH.

```bash
# 1. Install dependencies
python -m venv venv
venv\Scripts\activate          # Windows  (source venv/bin/activate on mac/linux)
pip install -r requirements.txt

# 2. Configure
copy .env.template .env.local   # then edit: set FIREWORKS_API_KEY
```

## Usage

```bash
# Submission mode against a local tasks file
python -m src.main --tasks input/tasks.json --results output/results.json

# Local dev: single video or directory
python -m src.main path/to/clip.mp4
python -m src.main path/to/videos/ --output-dir results

# No API key yet? Mock mode runs the full pipeline with canned captions
python -m src.main path/to/clip.mp4 --mock

# Override the model for one run
python -m src.main clip.mp4 --model accounts/fireworks/models/some-model
```

## Docker

The judging harness injects `FIREWORKS_API_KEY` and `FIREWORKS_BASE_URL` (its API proxy) as **environment variables at container runtime**. The code reads both from the environment at startup, so the submission image needs no credentials baked in:

```bash
docker build -t fourtakes .

# Simulate the judging harness locally (runtime env injection):
docker run \
  -e FIREWORKS_API_KEY=your_key \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  fourtakes

# Mock-mode smoke test (no key anywhere => mock mode automatically):
docker run -v /path/to/input:/input -v /path/to/output:/output fourtakes
```

As a fallback (e.g. running the image somewhere that injects nothing), the same variables can optionally be baked at build time with `--build-arg FIREWORKS_API_KEY=... --build-arg FIREWORKS_CAPTION_MODEL=...`; runtime env vars always take precedence over baked values. If you do bake a key, remember the image is public once pushed — revoke the key after the event.

The judging VM runs `linux/amd64`; standard builds on Intel/AMD machines produce that by default. On Apple Silicon add `--platform linux/amd64`.

## Submission

Track 2 deliverables, all linked from this repository:

- **Code:** this repo — https://github.com/srikantlose/fourtakes
- **Docker image:** https://hub.docker.com/r/srikantlose/fourtakes — `docker pull srikantlose/fourtakes:latest`
- **Demo video:** _TODO_
- **Slide deck:** _TODO_

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

49 tests, all runnable offline — no API key, credits, or ffmpeg required (extraction, downloads, and API calls are mocked).

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `FIREWORKS_API_KEY` | (empty → mock mode) | Fireworks API key (injected at judging time) |
| `FIREWORKS_BASE_URL` | `https://api.fireworks.ai/inference/v1` | API base URL (judging harness injects its proxy) |
| `FIREWORKS_CAPTION_MODEL` | `accounts/fireworks/models/qwen3p7-plus` | Vision model (config-only, never hardcoded; must be serverless) |
| `FIREWORKS_TRANSCRIPTION_MODEL` | `whisper-v3` | Fireworks-hosted Whisper model |
| `MOCK_MODE` | `false` | Force canned responses, no API calls |
| `TASKS_PATH` | `/input/tasks.json` | Task list (submission mode) |
| `RESULTS_PATH` | `/output/results.json` | Results file (submission mode) |
| `MAX_CONCURRENT_TASKS` | `3` | Videos processed in parallel |
| `FRAME_INTERVAL_SECONDS` | `1.5` | Seconds between extracted frames |
| `MAX_FRAMES` | `16` | Cap on frames sent to the model (evenly sampled) |
| `FRAME_SCALE_WIDTH` | `512` | Downscale frames wider than this (0 = off) |
| `ENABLE_AUDIO_TRANSCRIPTION` | `false` | Transcribe audio for extra context (Fireworks has deprecated this endpoint — see below) |
| `DOWNLOAD_TIMEOUT` / `API_TIMEOUT` | `120` / `60` | Timeouts in seconds |
| `PROMPTS_PATH` | `config/prompts.json` | Prompt definitions file |
| `LOG_LEVEL` / `LOG_FILE` | `INFO` / `fourtakes.log` | Logging |
| `OUTPUT_DIR` / `TEMP_DIR` | `output` / `.temp` | Directories (dev mode) |

## Editing the Caption Styles

All prompts (base caption, the four styles, and the generic fallback template) live in [config/prompts.json](config/prompts.json). Edit that file to iterate on caption voice — no code changes needed.

## License

MIT
