# FourTakes: Video Captioning Pipeline

Generate four distinctly-voiced captions (Formal, Sarcastic, Humorous-Tech, Humorous-Non-Tech) for short video clips using vision models and the Fireworks AI API.

## Project Structure

```
fourtakes/
├── src/                    # Core pipeline code
│   ├── __init__.py
│   ├── config.py          # Configuration loader (env vars)
│   ├── frame_extractor.py # Video frame extraction
│   ├── whisper_client.py  # Audio transcription (Phase 2)
│   ├── fireworks_client.py # Fireworks API wrapper (Phase 2)
│   ├── captioner.py       # Base caption generation (Phase 2)
│   └── main.py            # CLI entrypoint (Phase 3+)
├── tests/                  # Unit and integration tests
│   ├── __init__.py
│   └── test_frame_extractor.py
├── docker/                 # Docker configuration
│   └── Dockerfile         # (Phase 5)
├── config/                 # Configuration files
│   └── prompts.json       # Style prompts (Phase 3)
├── .env.template          # Environment variable template
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Setup (Phase 1)

### Prerequisites
- **Python 3.9+**
- **ffmpeg** and **ffprobe** (for frame extraction)

#### Install ffmpeg

**Windows (PowerShell with Chocolatey):**
```powershell
choco install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install ffmpeg
```

Verify installation:
```bash
ffmpeg -version
ffprobe -version
```

### Configuration

1. **Copy environment template:**
   ```bash
   cp .env.template .env.local
   ```

2. **Edit `.env.local` with your settings:**
   ```
   FIREWORKS_API_KEY=your_key_here
   FIREWORKS_CAPTION_MODEL=accounts/fireworks/models/phi-4-vision-128k
   FRAME_INTERVAL_SECONDS=1.5
   ENABLE_AUDIO_TRANSCRIPTION=true
   OPENAI_API_KEY=your_openai_key_for_whisper
   ```

3. **Create Python virtual environment:**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Phase 1 Status: ✓ COMPLETE

- [x] Project directory structure created
- [x] Python requirements.txt configured
- [x] Config loader implemented (reads .env files)
- [x] Frame extraction module (ffmpeg wrapper)
- [x] Unit tests for frame extraction
- [x] Documentation

### What's Working
- **Frame Extraction:** Extract frames from video files at configurable intervals
- **Video Metadata:** Retrieve video duration, FPS, resolution
- **Audio Extraction:** Extract audio tracks from videos (for Whisper transcription)
- **Configuration:** Load settings from .env files with sensible defaults

### How to Test Frame Extraction

```python
from src.frame_extractor import FrameExtractor

extractor = FrameExtractor(frame_interval=1.5)  # 1 frame every 1.5 seconds

# Extract frames
frames, metadata = extractor.extract_frames("sample_video.mp4")
print(f"Extracted {len(frames)} frames")
print(f"Frame paths: {frames}")
print(f"Metadata: {metadata}")

# Get video duration
duration = extractor.get_video_duration("sample_video.mp4")
print(f"Video duration: {duration} seconds")
```

### Run Unit Tests

```bash
python -m pytest tests/test_frame_extractor.py -v
# Or
python -m unittest tests.test_frame_extractor -v
```

## Next Phase: Phase 2 (Fireworks API & Base Captions)

Planned for Days 3–5:
- [ ] Implement Fireworks API client (mocked first, then real)
- [ ] Integrate Whisper API for audio transcription
- [ ] Implement base caption generation
- [ ] Add retry logic and error handling
- [ ] Integration tests with mocked API

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FIREWORKS_API_KEY` | (required) | Fireworks API key for model access |
| `FIREWORKS_CAPTION_MODEL` | `phi-4-vision-128k` | Vision model to use for captions |
| `FRAME_INTERVAL_SECONDS` | `1.5` | Seconds between extracted frames |
| `ENABLE_AUDIO_TRANSCRIPTION` | `true` | Enable audio transcription |
| `OPENAI_API_KEY` | (optional) | OpenAI key for Whisper (if using OpenAI) |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `LOG_FILE` | `fourtakes.log` | Path to log file |
| `OUTPUT_DIR` | `output` | Directory for JSON outputs |
| `TEMP_DIR` | `.temp` | Directory for temporary files |

## License

MIT License

## Project Timeline

- **Phase 1** (Days 1–2): ✓ DONE
- **Phase 2** (Days 3–5): API integration & base captions
- **Phase 3** (Days 5–6): Style transformation prompts
- **Phase 4** (Days 7–8): Polish and optimization
- **Phase 5** (Days 9–10): Containerization
- **Phase 6** (Day 10–11): Launch-day readiness
- **Phase 7** (Optional): Fine-tuning with ActivityNet Captions
