"""
REST API for Audio Transcription Service - PRODUCTION VERSION
Pure in-memory processing with preloaded pipeline
"""
from flask import Flask, request, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS
import os
import time
from pathlib import Path
from io import BytesIO
import threading

from src.core.config import Config
from src.core.exceptions import *
from src.services.jobs import JobQueue, QueueFullError


# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_AUDIO_SIZE_MB * 1024 * 1024

# Enable CORS
CORS(app)

# Swagger UI setup
SWAGGER_URL = '/docs'
API_URL = '/api/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "SIMTELPAS AI - Audio Transcription API",
        'defaultModelsExpandDepth': -1,
        'docExpansion': 'none'
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# ============================================================================
# PIPELINE INITIALIZATION
# ============================================================================
# Pipeline will be injected by wsgi.py (production) or initialized by run.py (dev)
# DO NOT create pipeline here - it will be set externally
pipeline = None
jobs = None
# Leaves request threads free for /ping, /health and /jobs.
sync_waiters = threading.BoundedSemaphore(Config.SYNC_MAX_WAITERS)


def init(loaded_pipeline):
    """Attach the loaded pipeline and start the job workers"""
    global pipeline, jobs
    pipeline = loaded_pipeline
    jobs = JobQueue(
        _run_job,
        max_pending=Config.ASR_MAX_PENDING,
        job_timeout_seconds=Config.ASR_JOB_TIMEOUT_MINUTES * 60,
    )


def _run_job(payload):
    """Runs on a job worker thread: the only place the GPU is used"""
    pipe = get_pipeline()
    if payload['kind'] == 'tracks':
        result = pipe.process_tracks(
            payload['tracks'], include_summary=payload['include_summary']
        )
    else:
        result = pipe.process(
            BytesIO(payload['audio']),
            num_speakers=payload['num_speakers'],
            min_speakers=payload['min_speakers'],
            max_speakers=payload['max_speakers'],
            include_summary=payload['include_summary'],
        )
    return _format_result(result)


def _format_result(result):
    """The response data shared by /transcribe and finished jobs"""
    format_start = time.time()
    transcript_lines = []
    current_speaker = None
    for seg in result['segments']:
        if seg['speaker'] == current_speaker and transcript_lines:
            transcript_lines[-1] += f" {seg['text']}"
        else:
            transcript_lines.append(f"{seg['speaker']}: {seg['text']}")
            current_speaker = seg['speaker']

    return {
        'transcript': "\n".join(transcript_lines),
        'summary': result['summary'],
        'summary_status': result.get('summary_status'),
        'sentiment': result['sentiment'],
        'flagged_phrases': result['flagged_phrases'],
        'segments': result['segments'],
        'audio_info': {
            'duration_seconds': result['audio_info']['duration_seconds'],
            'duration_minutes': result['audio_info']['duration_minutes'],
            'sample_rate': result['audio_info']['sample_rate']
        },
        'statistics': {
            'total_duration': result['statistics']['total_duration'],
            'total_words': result['statistics']['total_words'],
            'num_speakers': result['statistics']['num_speakers'],
            'speakers': {
                speaker: {
                    'duration': round(data['duration'], 2),
                    'duration_minutes': round(data['duration'] / 60, 2),
                    'words': data['words'],
                    'turns': data['turns']
                }
                for speaker, data in result['statistics']['speakers'].items()
            }
        },
        'timings': {
            'validation': round(result['timings'].get('validation', 0), 2),
            'preparation': round(result['timings'].get('preparation', 0), 2),
            'diarization': round(result['timings'].get('diarization', 0), 2),
            'transcription': round(result['timings'].get('transcription', 0), 2),
            'summarization': round(result['timings'].get('summarization', 0), 2),
            'formatting': round(time.time() - format_start, 2),
            'processing_total': round(result['timings'].get('total', 0), 2),
        }
    }


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


def get_pipeline():
    """
    Get pipeline instance

    Pipeline must be initialized externally:
    - In production: wsgi.py injects preloaded pipeline
    - In development: run.py initializes and injects pipeline

    Returns:
        AudioTranscriptionPipeline instance

    Raises:
        RuntimeError: If pipeline not initialized
    """
    global pipeline
    if pipeline is None:
        raise RuntimeError(
            "Pipeline not initialized! "
            "Run via 'gunicorn wsgi:app' (production) or 'python run.py' (development)"
        )
    return pipeline


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/swagger.json')
def swagger_spec():
    """Serve Swagger specification - embedded"""
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "SIMTELPAS AI - Audio Transcription API",
            "description": "API untuk transcribe audio dengan speaker diarization, AI summary, dan detailed timing info. Pure in-memory processing tanpa file saving.",
            "version": "2.0.0",
            "contact": {
                "name": "SIMTELPAS AI"
            }
        },
        "servers": [
            {
                "url": "/",
                "description": "Current server"
            }
        ],
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health Check",
                    "description": "Check API status dan dependencies",
                    "tags": ["System"],
                    "responses": {
                        "200": {
                            "description": "Service is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string", "example": "healthy"},
                                            "service": {"type": "string", "example": "SIMTELPAS AI"},
                                            "version": {"type": "string", "example": "2.0.0"},
                                            "timestamp": {"type": "string", "format": "date-time"},
                                            "dependencies": {
                                                "type": "object",
                                                "properties": {
                                                    "whisper": {"type": "string", "example": "loaded"},
                                                    "diarizer": {"type": "string", "example": "loaded"},
                                                    "gpu": {"type": "string", "example": "available"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/transcribe": {
                "post": {
                    "summary": "Transcribe Audio",
                    "description": "Upload audio file untuk di-transcribe dengan speaker diarization dan AI summary. Pure in-memory processing dengan detailed timing info.",
                    "tags": ["Transcription"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "required": ["file"],
                                    "properties": {
                                        "file": {
                                            "type": "string",
                                            "format": "binary",
                                            "description": "Audio file (mp3, wav, m4a, flac, ogg, opus, webm)"
                                        },
                                        "num_speakers": {
                                            "type": "integer",
                                            "description": "Exact number of speakers (optional)",
                                            "example": 2
                                        },
                                        "include_summary": {
                                            "type": "boolean",
                                            "description": "Generate AI summary (default: true)",
                                            "default": True,
                                            "example": True
                                        },
                                        "detailed_segments": {
                                            "type": "boolean",
                                            "description": "Include detailed segments array (default: false)",
                                            "default": False,
                                            "example": False
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Transcription successful",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "success": {"type": "boolean", "example": True},
                                            "message": {"type": "string", "example": "Transcription completed successfully"},
                                            "data": {
                                                "type": "object",
                                                "properties": {
                                                    "transcript": {
                                                        "type": "string",
                                                        "example": "SPEAKER_1: Hello\nSPEAKER_2: Hi there"
                                                    },
                                                    "summary": {
                                                        "type": "string",
                                                        "example": "Brief conversation summary..."
                                                    },
                                                    "sentiment": {
                                                        "type": "string",
                                                        "enum": ["POSITIVE", "NEGATIVE", "NEUTRAL"],
                                                        "example": "NEUTRAL",
                                                        "description": "Sentiment dari summary: POSITIVE | NEGATIVE | NEUTRAL"
                                                    },
                                                    "flagged_phrases": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                        "example": ["SPEAKER_1: transfer rekening ke nomor ini"],
                                                        "description": "Frasa mencurigakan dari transcript (maks 10, prioritas keyword TINGGI)"
                                                    },
                                                    "timings": {
                                                        "type": "object",
                                                        "properties": {
                                                            "upload": {"type": "number", "example": 0.45},
                                                            "validation": {"type": "number", "example": 0.12},
                                                            "preparation": {"type": "number", "example": 0.18},
                                                            "diarization": {"type": "number", "example": 12.34},
                                                            "transcription": {"type": "number", "example": 8.67},
                                                            "summarization": {"type": "number", "example": 2.15},
                                                            "formatting": {"type": "number", "example": 0.03},
                                                            "processing_total": {"type": "number", "example": 23.49},
                                                            "request_total": {"type": "number", "example": 23.97}
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request"
                        },
                        "413": {
                            "description": "File too large"
                        },
                        "500": {
                            "description": "Internal server error"
                        }
                    }
                }
            }
        }
    }
    return jsonify(spec)


@app.route('/ping', methods=['GET'])
def ping():
    """Lightweight liveness probe — never blocked by inference"""
    return jsonify({'status': 'ok'}), 200


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    import torch
    from datetime import datetime

    try:
        # Check if pipeline is initialized
        pipe = get_pipeline()

        health_data = {
            'status': 'healthy',
            'service': 'SIMTELPAS AI - Audio Transcription',
            'version': '2.0.0',
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'dependencies': {
                'pipeline': 'loaded',
                'whisper': 'loaded',
                'diarizer': 'loaded',
                'gpu': 'available' if torch.cuda.is_available() else 'unavailable',
                'summarizer': (
                    pipe.summarizer.last_known_status() if pipe.summarizer
                    else 'disabled'
                ),
            },
            'jobs': jobs.stats() if jobs else None
        }

        return jsonify(health_data), 200

    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'SIMTELPAS AI - Audio Transcription',
            'version': '2.0.0',
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'error': str(e)
        }), 500


@app.route('/jobs', methods=['POST'])
def create_job():
    """
    Queue a transcription and answer at once (202). Either several
    `tracks` (one per speaker) with matching `labels`, `names` and `offsets`
    (seconds from the call start), or one mixed `file` that is diarized.
    `job_key` makes a resend return the job already queued for it.
    """
    include_summary = request.form.get('include_summary', 'true').lower() == 'true'
    key = request.form.get('job_key') or None
    tracks = request.files.getlist('tracks')

    if tracks:
        labels = request.form.getlist('labels')
        names = request.form.getlist('names')
        offsets = request.form.getlist('offsets', type=float)
        if len(labels) != len(tracks) or len(offsets) != len(tracks):
            return _error('Every track needs a label and an offset', 400)
        if names and len(names) != len(tracks):
            return _error('Names must match the tracks', 400)
        if any(not label for label in labels) or any(o < 0 for o in offsets):
            return _error('Labels must be set and offsets not negative', 400)
        audios = [track.read() for track in tracks]
        if any(len(audio) == 0 for audio in audios):
            return _error('Empty track', 400)
        payload = {
            'kind': 'tracks',
            'include_summary': include_summary,
            'tracks': [
                {
                    'label': labels[i],
                    'name': names[i] if names else labels[i],
                    'offset': offsets[i],
                    'audio': audios[i],
                }
                for i in range(len(tracks))
            ],
        }
    else:
        file = request.files.get('file')
        if file is None or file.filename == '':
            return _error('No file provided', 400)
        audio = file.read()
        if len(audio) == 0:
            return _error('Empty file', 400)
        payload = {
            'kind': 'mixed',
            'include_summary': include_summary,
            'audio': audio,
            'num_speakers': request.form.get('num_speakers', type=int),
            'min_speakers': request.form.get('min_speakers', type=int),
            'max_speakers': request.form.get('max_speakers', type=int),
        }

    try:
        job = jobs.submit(payload, key)
    except QueueFullError as e:
        return _error(f'Queue full: {e}', 503)
    return jsonify({'success': True, 'message': 'Queued', 'data': _job_view(job)}), 202


@app.route('/jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    job = jobs.get(job_id)
    if job is None:
        return _error('Job not found', 404)
    return jsonify({'success': True, 'message': job.status, 'data': _job_view(job)}), 200


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Synchronous transcription, kept for older clients. It goes through the
    job queue too, so the GPU still runs one job at a time.
    """
    request_start = time.time()
    file = request.files.get('file')
    if file is None or file.filename == '':
        return _error('No file provided', 400)
    audio = file.read()
    if len(audio) == 0:
        return _error('Empty file', 400)
    if not sync_waiters.acquire(blocking=False):
        return _error('Too many waiting requests; use POST /jobs', 503)
    try:
        return _transcribe_and_wait(audio, request_start)
    finally:
        sync_waiters.release()


def _transcribe_and_wait(audio, request_start):
    try:
        job = jobs.submit({
            'kind': 'mixed',
            'include_summary': request.form.get('include_summary', 'true').lower() == 'true',
            'audio': audio,
            'num_speakers': request.form.get('num_speakers', type=int),
            'min_speakers': None,
            'max_speakers': None,
        })
    except QueueFullError as e:
        return _error(f'Queue full: {e}', 503)

    deadline = time.time() + Config.SYNC_MAX_WAIT_SECONDS
    while job.status in ('queued', 'processing'):
        if time.time() > deadline:
            return _error(f'Still processing; poll GET /jobs/{job.id}', 504)
        time.sleep(1)
    if job.status == 'failed':
        return _error(f'Transcription failed: {job.error}', 500)

    data = dict(job.result)
    if request.form.get('detailed_segments', 'false').lower() != 'true':
        data.pop('segments', None)
    data['timings'] = {**data['timings'], 'request_total': round(time.time() - request_start, 2)}
    return jsonify({
        'success': True,
        'message': 'Transcription completed successfully',
        'data': data
    }), 200


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'message': f'File too large. Maximum size: {Config.MAX_AUDIO_SIZE_MB}MB',
        'data': None
    }), 413


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    return jsonify({
        'success': False,
        'message': 'Internal server error',
        'data': None
    }), 500


def main():
    """
    Run Flask development server
    This is used by run.py for local development
    """
    # Validate configuration
    is_valid, errors = Config.validate()
    if not is_valid:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return

    # Run server
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8000))
    debug = os.getenv('API_DEBUG', 'false').lower() == 'true'

    print("=" * 80)
    print("SIMTELPAS AI - Audio Transcription Service")
    print("=" * 80)
    print(f"Model: {Config.WHISPER_MODEL}")
    print(f"Language: {Config.LANGUAGE}")
    print(f"Max file size: {Config.MAX_AUDIO_SIZE_MB}MB")
    print("=" * 80)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print("=" * 80)
    print(f"\n📖 Swagger UI: http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}/docs")
    print(f"❤️  Health Check: http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}/health")
    print(f"🎤 Transcribe: POST http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}/transcribe")
    print("=" * 80)

    app.run(
        host=host,
        port=port,
        debug=debug
    )


if __name__ == '__main__':
    main()