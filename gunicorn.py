"""Gunicorn config: one process owns the model; threads keep the API
responsive while the job worker thread transcribes. Run several containers
for more throughput, not more workers."""
bind = "0.0.0.0:8000"
workers = 1
worker_class = "gthread"
threads = 8
worker_tmp_dir = "/dev/shm"
# Loading the model in post_worker_init can take minutes the first time.
timeout = 900
# No max_requests: recycling the worker would drop queued jobs and reload
# the model; polling alone reaches any request limit quickly.
accesslog = "-"
errorlog = "-"
loglevel = "info"


def post_worker_init(worker):
    """Load the model after the fork"""
    import src.api.api as api_module
    from src.utils.pipeline import TranscriptionPipeline

    api_module.init(TranscriptionPipeline())
    print(f"[Worker {worker.pid}] ready", flush=True)
