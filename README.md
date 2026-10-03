# Audio Transcription System

A professional audio transcription system with speaker diarization and AI-powered summarization. Process audio files to get accurate transcripts with speaker identification and intelligent summaries.

## Features

- 🎙️ **Speaker Diarization**: Automatically identify and separate different speakers
- 📝 **Speech-to-Text**: High-quality transcription using OpenAI Whisper
- 🤖 **AI Summarization**: Intelligent summaries powered by Ollama (self-hosted)
- 🌍 **Multi-language Support**: Support for multiple languages including Indonesian and English
- 🚫 **No Raw Audio Storage**: Process audio in-memory without saving raw files
- 📊 **Rich Statistics**: Detailed analytics on speakers, duration, and word counts
- 🔄 **Multiple Output Formats**: TXT, JSON, SRT, and VTT formats
- ⚡ **REST API**: Easy integration via HTTP API
- 🛡️ **Robust Error Handling**: Comprehensive error handling with standard international responses

## Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio processing)
- Ollama (for AI summarization)

#### Install FFmpeg

**Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**macOS:**

```bash
brew install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html)

#### Install Ollama

**Linux/WSL:**

```bash
sudo apt-get install zstd -y
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download installer from [ollama.com/download](https://ollama.com/download/windows)

**Pull model yang direkomendasikan:**

```bash
# Production (V100 32GB)
ollama pull qwen2.5:14b

# Local/Testing (8GB VRAM)
ollama pull qwen2.5:7b
```

### Setup

1. Clone or download this project

2. Create virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure environment variables:

```bash
cp .env.example .env
# Edit .env with your tokens
```

## Configuration

### Required Configuration

1. **HuggingFace Token** (Required for speaker diarization)
   - Get token from: https://huggingface.co/settings/tokens
   - Accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1

2. **Ollama** (Required for AI summaries - self-hosted, gratis)
   - Install Ollama dan pull model (lihat bagian Install Ollama di atas)

### Environment Variables

```bash
# Required
HUGGINGFACE_TOKEN=your_huggingface_token_here

# Model configuration
WHISPER_MODEL=large-v3  # Options: tiny, base, small, medium, large, large-v3
LANGUAGE=id             # Default language code

# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:14b  # qwen2.5:7b untuk local testing
```

## Usage

### Command Line

#### Basic Usage

```bash
python pipeline.py audio_file.mp3
```

#### With Python Script

```python
from pipeline import AudioTranscriptionPipeline

# Initialize pipeline
pipeline = AudioTranscriptionPipeline(
    whisper_model='base',  # Model size
    language='id',         # Target language
    enable_summary=True    # Enable AI summary
)

# Process audio
result = pipeline.process(
    audio_input='recording.mp3',
    num_speakers=2,              # Optional: exact number of speakers
    include_summary=True,        # Generate AI summary
    output_formats=['txt', 'json', 'srt']  # Output formats
)

# Access results
print(f"Duration: {result['statistics']['total_duration']/60:.2f} minutes")
print(f"Speakers: {result['statistics']['num_speakers']}")
print(f"Summary: {result['summary']}")
```

### REST API

#### Start API Server

```bash
python run.py
```

The API will be available at `http://localhost:8000`

#### API Endpoints

**Health Check**

```bash
GET /health
```

**Transcribe Audio**

```bash
POST /transcribe

Form Data:
- file: Audio file (required)
- num_speakers: Number of speakers (optional)
- min_speakers: Minimum speakers (optional)
- max_speakers: Maximum speakers (optional)
- language: Language code (optional)
- include_summary: true/false (optional)
- output_formats: txt,json,srt,vtt (optional)
```

**Example with cURL:**

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@recording.mp3" \
  -F "num_speakers=2" \
  -F "include_summary=true" \
  -F "output_formats=txt,json,srt"
```

**Example with Python:**

```python
import requests

url = 'http://localhost:8000/transcribe'
files = {'file': open('recording.mp3', 'rb')}
data = {
    'num_speakers': 2,
    'include_summary': 'true',
    'output_formats': 'txt,json,srt'
}

response = requests.post(url, files=files, data=data)
result = response.json()

print(result)
```

**Download Output File**

```bash
GET /download/<filename>
```

**Get Supported Formats**

```bash
GET /formats
```

### Background jobs (recommended)

`POST /transcribe` answers only when the transcription is done, which takes
minutes for long calls. `POST /jobs` answers at once with `202` and a job id;
one worker thread runs the GPU work, one job at a time (the models are not
thread-safe). A job running longer than `ASR_JOB_TIMEOUT_MINUTES` (default
60) is taken as hung and the worker restarts; clients resend lost jobs.

**Per-speaker tracks** (no diarization; the speaker is known):

```bash
curl -X POST http://localhost:8000/jobs \
  -F job_key=<call id> \
  -F tracks=@inmate.mp3 -F labels=INMATE      -F names="Warga Binaan" -F offsets=0 \
  -F tracks=@family.mp3 -F labels=FAMILY:abc  -F names="Ibu"          -F offsets=8.28
```

`offsets` is when each track starts in the call, in seconds; segments come
back on the call's timeline, labelled with the track's label. Silence and text
Whisper invents over it are dropped (`SILENCE_DB`, default -45).

**One mixed file** (diarized): `-F file=@call.mp3 -F num_speakers=2`
(or `min_speakers` / `max_speakers`).

Sending the same `job_key` again returns the job already queued, running or
done for it. Poll the job:

```bash
curl http://localhost:8000/jobs/<job_id>
# data.status: queued (data.position = jobs ahead) | processing | done | failed
# done -> data.result: transcript, segments, summary, sentiment, flagged_phrases, statistics, timings
```

Finished jobs are kept for 24 hours; a restart drops them, so clients should
resubmit a job that answers 404. `ASR_MAX_PENDING` (default 30) caps the queue,
whose jobs hold their audio in memory. `POST /transcribe` serves at most 4
waiting callers (503 beyond) for up to `SYNC_MAX_WAIT_SECONDS` (504 after).

Tests (no GPU needed): `pip install pytest flask flask-cors flask-swagger-ui python-dotenv numpy && python -m pytest tests`

## Output Formats

### TXT Format

Human-readable transcript with timestamps and speaker labels:

```
[00:00 - 00:05] SPEAKER_1: Hello, welcome to the meeting.
[00:05 - 00:10] SPEAKER_2: Thank you for having me.
```

### JSON Format

Structured data including full transcript, segments, statistics, and summary:

```json
{
  "audio_info": {
    "duration_minutes": 5.2,
    "sample_rate": 16000
  },
  "statistics": {
    "total_duration": 312.5,
    "total_words": 450,
    "num_speakers": 2
  },
  "segments": [...],
  "summary": "..."
}
```

### SRT Format

Subtitle format for video players:

```
1
00:00:00,000 --> 00:00:05,000
SPEAKER_1: Hello, welcome to the meeting.

2
00:00:05,000 --> 00:00:10,000
SPEAKER_2: Thank you for having me.
```

### VTT Format

WebVTT subtitle format for web browsers:

```
WEBVTT

00:00:00.000 --> 00:00:05.000
SPEAKER_1: Hello, welcome to the meeting.

00:00:05.000 --> 00:00:10.000
SPEAKER_2: Thank you for having me.
```

## Supported Audio Formats

- MP3
- WAV
- M4A
- FLAC
- OGG
- OPUS
- WebM

## Error Handling

The system provides comprehensive error handling with standard international error responses:

### Error Types

- `AudioFileError`: Issues with audio file loading
- `AudioFormatError`: Unsupported audio format
- `AudioSizeError`: File exceeds size limit
- `ModelLoadError`: Failed to load AI models
- `DiarizationError`: Speaker diarization failed
- `TranscriptionError`: Transcription failed
- `SummarizationError`: AI summarization failed
- `ConfigurationError`: Configuration issues

### Example Error Response (API)

```json
{
  "success": false,
  "error": "Unsupported format: .avi. Supported formats: .mp3, .wav, .m4a, .flac, .ogg, .opus, .webm",
  "error_type": "AudioFormatError"
}
```

## Performance Considerations

### Whisper Model Selection

| Model  | Speed   | Accuracy | RAM   | Use Case      |
| ------ | ------- | -------- | ----- | ------------- |
| tiny   | Fastest | Low      | ~1GB  | Quick drafts  |
| base   | Fast    | Good     | ~1GB  | General use   |
| small  | Medium  | Better   | ~2GB  | Balanced      |
| medium | Slow    | High     | ~5GB  | High accuracy |
| large  | Slowest | Highest  | ~10GB | Best quality  |

### Ollama Model Selection

| Model       | VRAM  | Kualitas Indo | Rekomendasi       |
| ----------- | ----- | ------------- | ----------------- |
| qwen2.5:3b  | ~4GB  | ⭐⭐⭐        | Low-end GPU       |
| qwen2.5:7b  | ~8GB  | ⭐⭐⭐⭐      | Local testing     |
| qwen2.5:14b | ~16GB | ⭐⭐⭐⭐⭐    | Production (V100) |

### Processing Time

Approximate processing time (relative to audio duration):

- **Tiny/Base**: 0.1-0.2x (e.g., 5 min audio = 0.5-1 min processing)
- **Small**: 0.3-0.5x
- **Medium**: 0.5-1x
- **Large**: 1-2x

### File Size Limits

- Default maximum: 1000MB
- Configurable via `MAX_AUDIO_SIZE_MB` di `.env`
- Recommended: Keep files under 100MB for optimal performance

## Architecture

```
┌─────────────────┐
│  Audio Input    │
│ (File/Stream)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Audio Processor │  ← Validate & Prepare (No Save)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Diarizer      │  ← Identify Speakers (pyannote)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Transcriber    │  ← Speech-to-Text (Whisper)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Summarizer     │  ← AI Summary (Ollama - self-hosted)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Formatter     │  ← Generate Outputs
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output Files   │
│ (TXT/JSON/SRT)  │
└─────────────────┘
```

## Project Structure

```
audio-transcription/
├── src/
│   ├── api/
│   │   └── api.py             # REST API endpoints
│   ├── core/
│   │   ├── config.py          # Configuration settings
│   │   └── exceptions.py      # Custom exception classes
│   ├── services/
│   │   ├── diarizer.py        # Speaker diarization
│   │   ├── transcriber.py     # Speech transcription
│   │   └── summarizer.py      # AI summarization (Ollama)
│   └── utils/
│       ├── audio_processor.py # Audio processing utilities
│       ├── formatter.py       # Output formatting
│       └── pipeline.py        # Main pipeline orchestration
├── docker-compose.yml
├── Dockerfile
├── gunicorn.py
├── requirements.txt
├── run.py                     # Development server
├── wsgi.py                    # Production WSGI entry point
└── README.md
```

## Troubleshooting

### Common Issues

**1. "HUGGINGFACE_TOKEN is required"**

- Solution: Set `HUGGINGFACE_TOKEN` in `.env` file
- Get token from https://huggingface.co/settings/tokens
- Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1

**2. "Failed to load Whisper model"**

- Solution: Check internet connection (first run downloads model)
- Ensure sufficient disk space (~1-10GB depending on model)
- Try smaller model: `WHISPER_MODEL=base`

**3. "FFmpeg not found"**

- Solution: Install FFmpeg (see Installation section)

**4. "Ollama tidak bisa diakses"**

- Solution: Pastikan Ollama sudah berjalan (`ollama serve`)
- Cek `OLLAMA_BASE_URL` di `.env` sudah benar
- Pastikan model sudah di-pull: `ollama pull qwen2.5:14b`

**5. Out of memory errors**

- Solution: Use smaller Whisper model
- Gunakan Ollama model yang lebih kecil (`qwen2.5:7b`)
- Reduce audio file size

**6. Slow processing**

- Solution: Use smaller model (tiny/base)
- Enable GPU if available
- Process shorter audio segments

## Advanced Usage

### Custom Summary Types

```python
from src.services.summarizer import AISummarizer

summarizer = AISummarizer()

# Meeting summary
summary = summarizer.create_meeting_summary(
    transcript=transcript_text,
    statistics=stats,
    meeting_type='standup',  # or 'interview', 'general'
    language='id'
)
```

### Stream Processing

```python
# Process audio from bytes or file-like object
with open('recording.mp3', 'rb') as f:
    result = pipeline.process(f.read())
```

## License

This project uses several open-source components:

- OpenAI Whisper (MIT License)
- pyannote.audio (MIT License)
- Ollama (MIT License)
- qwen2.5 (Apache 2.0 License)

## Credits

- **Whisper**: OpenAI
- **Pyannote**: CNRS
- **Ollama**: Ollama Inc.
- **Qwen2.5**: Alibaba Cloud

## Support

For issues, questions, or contributions, please refer to the project repository or contact the maintainers.

---

**Note**: This system processes audio without storing raw audio files, ensuring privacy and efficient disk usage. All processing is done in-memory with only the final transcripts and outputs saved.
