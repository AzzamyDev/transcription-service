"""
REST API for Audio Transcription Service - PRODUCTION VERSION
Pure in-memory processing with preloaded pipeline
"""
from flask import Flask, request, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS
import os
import traceback
import time
from pathlib import Path
from io import BytesIO

from src.core.config import Config
from src.core.exceptions import *


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
                'gpu': 'available' if torch.cuda.is_available() else 'unavailable'
            }
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


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Transcribe audio with speaker diarization and AI summary
    Pure in-memory processing with detailed timing info
    """
    request_start = time.time()

    try:
        # Track upload time
        upload_start = time.time()

        # Get file from request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No file provided',
                'data': None
            }), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No file selected',
                'data': None
            }), 400

        # Get parameters
        num_speakers = request.form.get('num_speakers', type=int)
        include_summary = request.form.get('include_summary', 'true').lower() == 'true'
        detailed_segments = request.form.get('detailed_segments', 'false').lower() == 'true'

        # Read file into memory
        file_content = file.read()
        upload_time = time.time() - upload_start

        # Validate file not empty
        if len(file_content) == 0:
            return jsonify({
                'success': False,
                'message': 'Empty file',
                'data': None
            }), 400

        # Validate file size
        file_size_mb = len(file_content) / (1024 * 1024)
        if file_size_mb > Config.MAX_AUDIO_SIZE_MB:
            return jsonify({
                'success': False,
                'message': f'File too large: {file_size_mb:.1f}MB. Maximum: {Config.MAX_AUDIO_SIZE_MB}MB',
                'data': None
            }), 413

        # Get preloaded pipeline (NO LOADING HERE!)
        pipe = get_pipeline()

        # Process audio - PURE IN-MEMORY (NO FILE SAVING)
        audio_stream = BytesIO(file_content)

        result = pipe.process(
            audio_stream,
            num_speakers=num_speakers,
            include_summary=include_summary
        )

        # Track formatting time
        format_start = time.time()

        # Format transcript for display
        transcript_lines = []
        current_speaker = None

        for seg in result['segments']:
            speaker = seg['speaker']
            text = seg['text']

            # Merge if same speaker
            if speaker == current_speaker and transcript_lines:
                transcript_lines[-1] += f" {text}"
            else:
                transcript_lines.append(f"{speaker}: {text}")
                current_speaker = speaker

        formatted_transcript = "\n".join(transcript_lines)

        format_time = time.time() - format_start

        # Calculate total request time
        total_request_time = time.time() - request_start

        # Prepare response data
        response_data = {
            'transcript': formatted_transcript,
            'summary': result['summary'],
            'sentiment': result['sentiment'],
            'flagged_phrases': result['flagged_phrases'],
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
                'upload': round(upload_time, 2),
                'validation': round(result['timings']['validation'], 2),
                'preparation': round(result['timings']['preparation'], 2),
                'diarization': round(result['timings']['diarization'], 2),
                'transcription': round(result['timings']['transcription'], 2),
                'summarization': round(result['timings']['summarization'], 2),
                'formatting': round(format_time, 2),
                'processing_total': round(result['timings']['total'], 2),
                'request_total': round(total_request_time, 2)
            }
        }

        # Add detailed segments if requested (optional)
        if detailed_segments:
            response_data['segments'] = result['segments']

        # Return response
        response = {
            'success': True,
            'message': 'Transcription completed successfully',
            'data': response_data
        }

        return jsonify(response), 200

    except AudioFileError as e:
        return jsonify({
            'success': False,
            'message': str(e),
            'data': None
        }), 400

    except AudioFormatError as e:
        return jsonify({
            'success': False,
            'message': str(e),
            'data': None
        }), 400

    except AudioSizeError as e:
        return jsonify({
            'success': False,
            'message': str(e),
            'data': None
        }), 413

    except DiarizationError as e:
        return jsonify({
            'success': False,
            'message': f'Speaker diarization failed: {str(e)}',
            'data': None
        }), 500

    except TranscriptionError as e:
        return jsonify({
            'success': False,
            'message': f'Transcription failed: {str(e)}',
            'data': None
        }), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': 'Internal server error',
            'data': {'error_details': str(e)}
        }), 500


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