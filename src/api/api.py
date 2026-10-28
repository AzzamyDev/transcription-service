"""
REST API of the speech-to-text service. Requests are queued and processed by
one worker thread (the model is not thread-safe); scale out with replicas.
"""
import hmac
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint

from src.core.config import Config
from src.services.jobs import JobQueue, QueueFullError

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_AUDIO_SIZE_MB * 1024 * 1024
CORS(app)

SWAGGER_URL = '/docs'
API_URL = '/api/swagger.json'
app.register_blueprint(
    get_swaggerui_blueprint(
        SWAGGER_URL, API_URL,
        config={'app_name': 'Speech-to-text API', 'defaultModelsExpandDepth': -1, 'docExpansion': 'none'},
    ),
    url_prefix=SWAGGER_URL,
)
SWAGGER_SPEC = Path(__file__).with_name('swagger.json')

# Probes and docs stay open when API_KEY is set.
OPEN_PATHS = ('/ping', '/health', API_URL)

# Set by init(): wsgi.py leaves them empty until the model is loaded in the
# worker, run.py loads it before serving.
pipeline = None
jobs = None
# Leaves request threads free for /ping, /health and /jobs.
sync_waiters = threading.BoundedSemaphore(Config.SYNC_MAX_WAITERS)
LANGUAGE_CODE = re.compile(r'^[a-z]{2,3}$')


def init(loaded_pipeline):
    """Attach the loaded pipeline and start the job worker"""
    global pipeline, jobs
    pipeline = loaded_pipeline
    jobs = JobQueue(
        _run_job,
        max_pending=Config.ASR_MAX_PENDING,
        job_timeout_seconds=Config.ASR_JOB_TIMEOUT_MINUTES * 60,
    )


def _run_job(payload):
    """Runs on the job worker thread: the only place the model is used"""
    return pipeline.process(payload['tracks'], language=payload['language'])


def _job_view(job):
    view = {
        'job_id': job.id,
        'job_key': job.key,
        'status': job.status,
        'position': jobs.position(job),
        'created_at': job.created_at,
        'started_at': job.started_at,
        'finished_at': job.finished_at,
    }
    if job.status == 'done':
        view['result'] = job.result
    if job.status == 'failed':
        view['error'] = job.error
    return view


def _error(message, status):
    return jsonify({'success': False, 'message': message, 'data': None}), status


@app.before_request
def check_api_key():
    if not Config.API_KEY or request.path in OPEN_PATHS \
            or request.path.startswith(SWAGGER_URL):
        return None
    sent = request.headers.get('X-Api-Key', '')
    if not hmac.compare_digest(sent.encode(), Config.API_KEY.encode()):
        return _error('Missing or wrong X-Api-Key', 401)
    return None


def _read_audio_request():
    """
    The payload of a transcription request, or an error response. Either
    several `tracks` with matching `labels` and `offsets` (seconds from the
    start of the common timeline), or one `file` with an optional `label`.
    `language` overrides the default for this request.
    """
    language = request.form.get('language') or None
    if language and not LANGUAGE_CODE.match(language):
        return None, _error('language must be a code such as id or en', 400)

    tracks = request.files.getlist('tracks')
    if tracks:
        labels = request.form.getlist('labels')
        offsets = request.form.getlist('offsets', type=float)
        if len(labels) != len(tracks) or len(offsets) != len(tracks):
            return None, _error('Every track needs a label and an offset', 400)
        if any(not label for label in labels) or any(o < 0 for o in offsets):
            return None, _error('Labels must be set and offsets not negative', 400)
        entries = [(labels[i], offsets[i], tracks[i]) for i in range(len(tracks))]
    else:
        file = request.files.get('file')
        if file is None or file.filename == '':
            return None, _error('No file provided', 400)
        entries = [(request.form.get('label') or None, 0.0, file)]

    audio = [(label, offset, upload.read()) for label, offset, upload in entries]
    if any(len(data) == 0 for _, _, data in audio):
        return None, _error('Empty audio file', 400)
    return {
        'language': language,
        'tracks': [{'label': label, 'offset': offset, 'audio': data} for label, offset, data in audio],
    }, None


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.route(API_URL)
def swagger_spec():
    return jsonify(json.loads(SWAGGER_SPEC.read_text()))


@app.route('/ping', methods=['GET'])
def ping():
    """Liveness probe, never blocked by transcription"""
    return jsonify({'status': 'ok'}), 200


@app.route('/health', methods=['GET'])
def health():
    """Ready once the model is loaded"""
    body = {
        'service': 'speech-to-text',
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    if pipeline is None:
        return jsonify({**body, 'status': 'loading'}), 503
    return jsonify({
        **body,
        'status': 'healthy',
        'model': pipeline.transcriber.model_size,
        'compute_type': Config.WHISPER_COMPUTE_TYPE,
        'cpu_threads': Config.WHISPER_CPU_THREADS,
        'language': Config.LANGUAGE,
        'jobs': jobs.stats(),
    }), 200


@app.route('/jobs', methods=['POST'])
def create_job():
    """
    Queue a transcription and answer at once (202). `job_key` makes a resend
    return the job already queued, running or done for it.
    """
    if pipeline is None:
        return _error('Model is still loading', 503)
    payload, error = _read_audio_request()
    if error:
        return error
    try:
        job = jobs.submit(payload, request.form.get('job_key') or None)
    except QueueFullError as e:
        return _error(f'Queue full: {e}', 503)
    return jsonify({'success': True, 'message': 'Queued', 'data': _job_view(job)}), 202


@app.route('/jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    job = jobs.get(job_id) if jobs else None
    if job is None:
        return _error('Job not found', 404)
    return jsonify({'success': True, 'message': job.status, 'data': _job_view(job)}), 200


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Transcription that answers when done. It goes through the job queue too,
    so the model still runs one job at a time; long audio belongs in /jobs.
    """
    if pipeline is None:
        return _error('Model is still loading', 503)
    started = time.time()
    payload, error = _read_audio_request()
    if error:
        return error
    if not sync_waiters.acquire(blocking=False):
        return _error('Too many waiting requests; use POST /jobs', 503)
    try:
        return _transcribe_and_wait(payload, started)
    finally:
        sync_waiters.release()


def _transcribe_and_wait(payload, started):
    try:
        job = jobs.submit(payload)
    except QueueFullError as e:
        return _error(f'Queue full: {e}', 503)

    deadline = time.time() + Config.SYNC_MAX_WAIT_SECONDS
    while job.status in ('queued', 'processing'):
        if time.time() > deadline:
            return _error(f'Still processing; poll GET /jobs/{job.id}', 504)
        time.sleep(0.5)
    if job.status == 'failed':
        return _error(f'Transcription failed: {job.error}', 500)

    data = dict(job.result)
    data['timings'] = {**data['timings'], 'request_total': round(time.time() - started, 2)}
    return jsonify({'success': True, 'message': 'Transcription completed', 'data': data}), 200


@app.errorhandler(413)
def too_large(e):
    return _error(f'File too large. Maximum size: {Config.MAX_AUDIO_SIZE_MB}MB', 413)


@app.errorhandler(500)
def internal_error(e):
    return _error('Internal server error', 500)


def main():
    """Flask development server, used by run.py"""
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8000))
    debug = os.getenv('API_DEBUG', 'false').lower() == 'true'
    print(f"Swagger UI: http://127.0.0.1:{port}/docs", flush=True)
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    main()
