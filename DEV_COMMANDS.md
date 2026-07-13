# Development Commands

Quick reference for common development tasks.

## Setup

### First-time setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Activate virtual environment (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify ffmpeg is installed
ffmpeg -version
ffprobe -version
```

## Testing

### Run all tests
```bash
python -m pytest tests/ -v
```

### Run specific test file
```bash
python -m pytest tests/test_frame_extractor.py -v
```

### Run with coverage
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

### Run tests with unittest (alternative)
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Configuration

### Check current configuration
```bash
# Verify .env.local exists
cat .env.local

# Or print configuration to console (without API keys)
python -c "from src.config import load_config; import pprint; config = load_config(); pprint.pprint({k: v for k, v in config.items() if 'key' not in k.lower()})"
```

### Update configuration
```bash
# Copy template if not exists
cp .env.template .env.local

# Edit with your text editor
# Windows
notepad .env.local
# macOS/Linux
nano .env.local
```

## Linting & Formatting

### Check code style (requires black and flake8)
```bash
pip install black flake8

# Format with black
black src/ tests/

# Check with flake8
flake8 src/ tests/ --max-line-length=100
```

## Logging

### View recent logs
```bash
# Last 50 lines
tail -50 fourtakes.log

# Follow logs in real-time (requires `tail` on Windows)
tail -f fourtakes.log
```

### Clear logs
```bash
rm fourtakes.log
# or on Windows
del fourtakes.log
```

## Submission (Task) Mode Testing

### Create a local tasks file with the official example clips
```json
// save as input/tasks.json
[
  {
    "task_id": "v1",
    "video_url": "https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  },
  {
    "task_id": "v2",
    "video_url": "https://storage.googleapis.com/amd-hackathon-clips/13825391-uhd_3840_2160_30fps.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  },
  {
    "task_id": "v3",
    "video_url": "https://storage.googleapis.com/amd-hackathon-clips/3044693-uhd_3840_2160_24fps.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  }
]
```

### Run submission mode locally
```bash
# Real API (needs FIREWORKS_API_KEY in .env.local; downloads the clips)
python -m src.main --tasks input/tasks.json --results output/results.json

# Mock mode (still downloads + extracts frames, but no API calls)
MOCK_MODE=true python -m src.main --tasks input/tasks.json --results output/results.json
```

### Inspect the results
```bash
python -m json.tool output/results.json
```

## Local Demo Web UI

Not part of the submission — dev-only, separate `requirements-web.txt` (Flask), never copied into the Docker image.

### Install and run
```bash
pip install -r requirements-web.txt
python -m webapp.server
# open http://127.0.0.1:5000
```

### Run the webapp tests
```bash
# Skips cleanly if flask isn't installed
python -m unittest tests.test_webapp -v
```

### Exercise the API directly (no browser)
```bash
# Submit a clip in mock mode
curl -s -X POST http://127.0.0.1:5000/api/caption -F "video=@sample.mp4" -F "mock=true"
# -> {"job_id": "..."}

# Stream the live events for that job
curl -s -N http://127.0.0.1:5000/api/events/<job_id>
```

### Troubleshooting
- Port 5000 already in use: another instance is still running — find it with `netstat -ano | grep :5000` (Windows) and stop it, or edit the port in `webapp/server.py`'s `main()`.
- Upload succeeds but the job immediately errors on "Could not determine video duration": the file isn't a real video ffprobe can read (e.g. a text file with a `.mp4` name) — this is expected, not a bug.
- Nothing happens after clicking "Generate captions": open the browser devtools Network tab and check the `/api/events/<job_id>` request stayed open (EventSource); if the server log shows the job completed but the UI didn't update, it's a frontend issue in `webapp/static/app.js`, not the pipeline.

## Frame Extraction Testing

### Extract frames from a video (Python script)
```python
from src.frame_extractor import FrameExtractor
from src.logging_config import setup_logging

setup_logging()

extractor = FrameExtractor(frame_interval=1.5)
frames, metadata = extractor.extract_frames("path/to/video.mp4")

print(f"Extracted {len(frames)} frames:")
for frame in frames[:5]:
    print(f"  - {frame}")
```

### Quick test
```bash
python -c "
from src.frame_extractor import FrameExtractor
extractor = FrameExtractor()
try:
    duration = extractor.get_video_duration('sample.mp4')
    print(f'Video duration: {duration}s')
except Exception as e:
    print(f'Error: {e}')
"
```

## Cleanup

### Remove temporary files
```bash
# Remove cached Python files
find . -type d -name __pycache__ -exec rm -rf {} +
find . -name "*.pyc" -delete

# Remove temp directories
rm -rf .temp/
rm -rf temp/
rm -rf frames_*/

# On Windows PowerShell
Get-ChildItem -Path . -Include __pycache__ -Recurse -Force | Remove-Item -Recurse -Force
```

### Clean virtual environment
```bash
# Remove virtual environment (and reinstall if needed)
rm -rf venv
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Docker

### Build (submission image — no credentials baked; judging injects env vars at runtime)
```bash
docker build -t fourtakes:latest .
```

### Simulate the judging harness (runtime env injection)
```bash
docker run \
  -e FIREWORKS_API_KEY=your_key \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -v "$(pwd)/input:/input" \
  -v "$(pwd)/output:/output" \
  fourtakes:latest
```

### Optional: bake fallback credentials at build time
```bash
# Runtime env vars always override baked values. Only needed if the image
# must also work somewhere that injects nothing.
docker build \
  --build-arg FIREWORKS_API_KEY=your_key \
  --build-arg FIREWORKS_CAPTION_MODEL=accounts/fireworks/models/qwen3p7-plus \
  -t fourtakes:latest .
```

### Mock-mode smoke test (no key anywhere => mock mode)
```bash
docker run -v "$(pwd)/input:/input" -v "$(pwd)/output:/output" fourtakes:latest
```

### Local dev run against a mounted video
```bash
docker run -v /path/to/videos:/videos fourtakes:latest /videos/sample.mp4
```

### Debug Docker image
```bash
docker run -it --entrypoint /bin/bash fourtakes:latest
```

### Push for submission (public registry, linux/amd64)
```bash
docker build -t srikantlose/fourtakes:latest .
docker push srikantlose/fourtakes:latest
```
Published: https://hub.docker.com/r/srikantlose/fourtakes

## Debugging Tips

### Enable debug logging
Edit `.env.local`:
```
LOG_LEVEL=DEBUG
```

### Trace frame extraction
```python
from src.frame_extractor import FrameExtractor
import logging

logging.basicConfig(level=logging.DEBUG)
extractor = FrameExtractor(frame_interval=0.5)  # More frames
frames, metadata = extractor.extract_frames("video.mp4")
print(metadata)
```

### Inspect extracted frames
```bash
# List extracted frames
ls -la .temp/fourtakes_frames_*/

# Or on Windows
dir ".temp\fourtakes_frames_*"
```
