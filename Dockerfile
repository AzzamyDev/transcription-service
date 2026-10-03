# ============================================================================
# SIMTELPAS AI
# ============================================================================
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    ffmpeg git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126

COPY . .

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p /app/outputs /app/temp && \
    chown -R appuser:appuser /app/outputs /app/temp

USER appuser

ENV PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    FLASK_ENV=production \
    CUDA_LAUNCH_BLOCKING=0 \
    TORCH_ALLOW_TF32=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Workers, threads and timeouts live in gunicorn.py; flags here would
# override them (a sync worker blocks every request during a transcription).
CMD ["gunicorn", "--config", "gunicorn.py", "wsgi:app"]