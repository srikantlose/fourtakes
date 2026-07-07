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

# Track 2 judging injects NO environment variables — credentials and model
# ship inside the image. Supply them at build time:
#   docker build \
#     --build-arg FIREWORKS_API_KEY=your_key \
#     --build-arg FIREWORKS_CAPTION_MODEL=accounts/fireworks/models/... \
#     -t fourtakes .
# Empty values fall through to config defaults (empty key => mock mode).
ARG FIREWORKS_API_KEY=""
ARG FIREWORKS_CAPTION_MODEL=""
ENV FIREWORKS_API_KEY=${FIREWORKS_API_KEY} \
    FIREWORKS_CAPTION_MODEL=${FIREWORKS_CAPTION_MODEL}

# Submission-mode defaults: harness mounts /input and /output.
# Audio transcription is off in the container to protect the 10-minute
# runtime budget (re-enable with -e ENABLE_AUDIO_TRANSCRIPTION=true).
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
