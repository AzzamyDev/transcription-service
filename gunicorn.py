"""Gunicorn Config - 1 Worker with Real-time Progress"""
bind = "0.0.0.0:8000"
workers = 1
worker_class = "gthread"
threads = 2
worker_tmp_dir = "/dev/shm"
timeout = 900
max_requests = 100
max_requests_jitter = 10
accesslog = "-"
errorlog = "-"
loglevel = "info"

def post_worker_init(worker):
    """Initialize pipeline AFTER fork"""
    import sys
    import src.api.api as api_module
    from src.utils.pipeline import AudioTranscriptionPipeline

    print(f"\n{'='*80}", flush=True)
    print(f"[Worker {worker.pid}] Initializing pipeline...", flush=True)
    print(f"{'='*80}\n", flush=True)

    # Force unbuffered output for real-time progress
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    pipeline = AudioTranscriptionPipeline()
    api_module.pipeline = pipeline

    print(f"\n{'='*80}", flush=True)
    print(f"[Worker {worker.pid}] ✓ Ready!", flush=True)
    print(f"{'='*80}\n", flush=True)