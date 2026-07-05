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
COPY .env.template .

# Model + API key are supplied at runtime via env vars, never baked in:
#   docker run -e FIREWORKS_API_KEY=... -e FIREWORKS_CAPTION_MODEL=... \
#     -v /path/to/videos:/videos -v /path/to/output:/app/output \
#     fourtakes /videos
ENV OUTPUT_DIR=/app/output \
    TEMP_DIR=/tmp/fourtakes \
    PROMPTS_PATH=/app/config/prompts.json

ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--help"]
