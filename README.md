# FourTakes: Video Captioning Pipeline

Generate four distinctly-voiced captions — **Formal**, **Sarcastic**, **Humorous-Tech**, and **Humorous-Non-Tech** — for short video clips (30 sec–2 min) using vision models via the Fireworks AI API.

Built for AMD Developer Hackathon: ACT II, Track 2 (Video Captioning).

## How It Works

```
video clip
  → extract frames with ffmpeg (every 1.5s, configurable)
  → downsample to MAX_FRAMES (evenly, keeping first + last frame)
  → extract audio + transcribe via Fireworks-hosted Whisper (optional, best-effort)
  → ONE vision-model call: neutral factual base caption
  → FOUR text-model calls: transform base caption into each style
  → JSON result per video + combined results file
```

The base caption is generated **once** per video and reused for all four styles, so the outputs stay factually consistent and vision-call costs stay low.

## Key Design Rules

- **No hardcoded model names.** The caption model comes from `FIREWORKS_CAPTION_MODEL` (env var or `--model` flag). Launch-day model swaps are a config change, not a code change.
- **Mock mode** (`MOCK_MODE=true` or `--mock`): the entire pipeline runs with canned API responses — no key, no credits, no network. Used for development and CI.
- **Best-effort audio**: transcription failures never break captioning; the pipeline continues on frames alone.
- **Batch-safe**: one broken video in a directory never aborts the others; its error is recorded in its own result entry.
- **Every API call is logged** with model name, tokens, latency, and status.

## Project Structure

```
fourtakes/
├── src/
│   ├── config.py            # env-based config + prompts loader
│   ├── frame_extractor.py   # ffmpeg frame/audio extraction
│   ├── fireworks_client.py  # Fireworks API wrapper (retries, logging, mock mode)
│   ├── transcriber.py       # Fireworks-hosted Whisper transcription
│   ├── captioner.py         # base caption + 4 style transforms
│   ├── pipeline.py          # end-to-end orchestrator + JSON output
│   ├── logging_config.py    # file + console logging
│   └── main.py              # CLI entrypoint
├── config/
│   └── prompts.json         # ALL prompts (base + 4 styles) — edit here to iterate
├── tests/                   # unit + integration tests (no API key or ffmpeg needed)
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
# Single video
python -m src.main path/to/clip.mp4

# Directory of videos
python -m src.main path/to/videos/ --output-dir results

# No API key yet? Run in mock mode (full pipeline, canned captions)
python -m src.main path/to/clip.mp4 --mock

# Override the model for one run (e.g., launch day)
python -m src.main clip.mp4 --model accounts/fireworks/models/launch-day-model

# Skip audio transcription
python -m src.main clip.mp4 --no-audio
```

Output per video (`output/<video_id>.json`):

```json
{
  "video_id": "clip",
  "status": "ok",
  "base_caption": "...neutral factual description...",
  "captions": {
    "formal": "...",
    "sarcastic": "...",
    "humorous_tech": "...",
    "humorous_nontech": "..."
  },
  "metadata": {
    "model_used": "...", "frames_extracted": 23, "frames_sent_to_model": 16,
    "audio_transcribed": true, "duration_seconds": 35.0, "...": "..."
  }
}
```

`output/all_results.json` additionally includes an `api_usage` summary (total calls, tokens, errors).

## Docker

```bash
docker build -t fourtakes .

docker run \
  -e FIREWORKS_API_KEY=your_key \
  -e FIREWORKS_CAPTION_MODEL=accounts/fireworks/models/whatever-is-revealed \
  -v /path/to/videos:/videos \
  -v /path/to/output:/app/output \
  fourtakes /videos

# Mock-mode smoke test (no key needed):
docker run -v /path/to/videos:/videos -v /path/to/output:/app/output fourtakes /videos --mock
```

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

22 tests, all runnable offline — no API key, credits, or ffmpeg required (extraction and API calls are mocked).

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `FIREWORKS_API_KEY` | (empty → mock mode) | Fireworks API key |
| `FIREWORKS_CAPTION_MODEL` | `accounts/fireworks/models/phi-4-vision-128k` | Vision model — swap on launch day |
| `FIREWORKS_TRANSCRIPTION_MODEL` | `whisper-v3` | Fireworks-hosted Whisper model |
| `MOCK_MODE` | `false` | Force canned responses, no API calls |
| `FRAME_INTERVAL_SECONDS` | `1.5` | Seconds between extracted frames |
| `MAX_FRAMES` | `16` | Cap on frames sent to the model (evenly sampled) |
| `ENABLE_AUDIO_TRANSCRIPTION` | `true` | Transcribe audio for extra caption context |
| `PROMPTS_PATH` | `config/prompts.json` | Prompt definitions file |
| `LOG_LEVEL` / `LOG_FILE` | `INFO` / `fourtakes.log` | Logging |
| `OUTPUT_DIR` / `TEMP_DIR` | `output` / `.temp` | Directories |

## Editing the Caption Styles

All five prompts (base + four styles) live in [config/prompts.json](config/prompts.json). Edit that file to iterate on caption voice — no code changes needed.

## License

MIT
