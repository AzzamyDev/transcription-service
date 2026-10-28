FROM python:3.12-slim

WORKDIR /app

# FFmpeg ships inside the PyAV wheel, so no system packages are needed.
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/models && \
    chown -R appuser:appuser /app/models
USER appuser

ENV PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models

EXPOSE 8000

# /ping answers while the model is still loading; /health turns 200 when ready.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://localhost:8000/health', timeout=5).status != 200)"

# Workers, threads and timeouts live in gunicorn.py; flags here would
# override them (a sync worker blocks every request during a transcription).
CMD ["gunicorn", "--config", "gunicorn.py", "wsgi:app"]
