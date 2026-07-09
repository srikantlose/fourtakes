FROM python:3.11-slim

# ffmpeg + ffprobe for frame/audio extraction
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY config/ config/

# The judging harness injects FIREWORKS_API_KEY and FIREWORKS_BASE_URL as
# environment variables at container runtime; the code reads both from the
# environment at startup, so nothing needs to be baked in for submission.
# Runtime env vars always override the image ENV values below.
#
# The build args remain as an OPTIONAL fallback for local dev or in case
# the harness injects nothing:
#   docker build \
#     --build-arg FIREWORKS_API_KEY=your_key \
#     --build-arg FIREWORKS_CAPTION_MODEL=accounts/fireworks/models/... \
#     -t fourtakes .
# Empty values fall through to config defaults (empty key => mock mode,
# empty base URL => public Fireworks endpoint).
ARG FIREWORKS_API_KEY=""
ARG FIREWORKS_BASE_URL=""
ARG FIREWORKS_CAPTION_MODEL=""
ENV FIREWORKS_API_KEY=${FIREWORKS_API_KEY} \
    FIREWORKS_BASE_URL=${FIREWORKS_BASE_URL} \
    FIREWORKS_CAPTION_MODEL=${FIREWORKS_CAPTION_MODEL}

# Submission-mode defaults: harness mounts /input and /output.
# Audio transcription is off: Fireworks has deprecated hosted audio
# inference platform-wide, so /audio/transcriptions returns 401 for
# every key right now, regardless of this flag.
ENV TASKS_PATH=/input/tasks.json \
    RESULTS_PATH=/output/results.json \
    ENABLE_AUDIO_TRANSCRIPTION=false \
    OUTPUT_DIR=/app/output \
    TEMP_DIR=/tmp/fourtakes \
    PROMPTS_PATH=/app/config/prompts.json

# No args => submission mode (reads TASKS_PATH, writes RESULTS_PATH).
# Pass a video path or flags for local dev usage instead.
ENTRYPOINT ["python", "-m", "src.main"]
CMD []
