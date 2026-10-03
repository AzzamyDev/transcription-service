"""Gunicorn config: one process owns the GPU models; threads keep the API
responsive while job worker threads run transcriptions."""
bind = "0.0.0.0:8000"
workers = 1
worker_class = "gthread"
threads = 8
worker_tmp_dir = "/dev/shm"
# Loading the models in post_worker_init takes minutes.
timeout = 900
# No max_requests: recycling the worker would drop queued jobs and reload
# the models; polling alone reaches any request limit quickly.
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

    api_module.init(AudioTranscriptionPipeline())

    print(f"\n{'='*80}", flush=True)
    print(f"[Worker {worker.pid}] ✓ Ready!", flush=True)
    print(f"{'='*80}\n", flush=True)
