# Speech-to-text service

Client-agnostic HTTP service: audio in, timestamped text out, faster-whisper on the CPU. No client names, domain vocabulary or diarization/summaries belong in it; callers own those.

## Layout

- `src/api/api.py` — routes, request parsing (`_read_audio_request` serves `/jobs` and `/transcribe`), `X-Api-Key` check; `src/api/swagger.json` is the served OpenAPI spec, edit it with the routes.
- `src/utils/pipeline.py` — `TranscriptionPipeline.process(tracks, language)`: decode, transcribe, shift by offset, sort, merge, statistics. A single `file` is one track with `label=None`.
- `src/services/transcriber.py` — `WhisperTranscriber`: VAD regions, Whisper decode, the silence/invention filters in `transcribe_speech`.
- `src/services/hallucinations.py` — stock lines and loops Whisper invents (Indonesian list; extend per language when needed).
- `src/services/jobs.py` — `JobQueue`. `src/utils/audio.py` — PyAV decoder. `src/core/config.py` — every tunable, from env.

## Runtime model

- **One worker per process.** The model is not thread-safe: all transcription runs on the single `JobQueue` thread, and request threads never touch it. Throughput comes from container replicas (`docker-compose.yml`), not threads or gunicorn workers (`gunicorn.py` stays at 1 worker, no `max_requests` — recycling drops queued jobs).
- **Model loads after fork** in `gunicorn.py` `post_worker_init`; `/health` is 503 until `api.init()` ran. `run.py` is the dev path.
- **Jobs live in memory**, 24 h, lost on restart; clients resubmit on 404. Same `job_key` returns the existing job unless it failed. A job past `ASR_JOB_TIMEOUT_MINUTES` makes the process `os._exit(1)` so Docker restarts it.
- **Audio stays in memory**; a job's payload is dropped once processed.

## Gotchas

- `faster_whisper.decode_audio` breaks on PyAV ≥ 15 (`metadata_errors`); `src/utils/audio.py` decodes with PyAV directly. Keep it that way.
- `faster-whisper` imports `requests` without declaring it; keep it in `requirements.txt`.
- The language comes from `WHISPER_LANGUAGE`, not `LANGUAGE` (the locale variable on GNU systems).
- Old clients may still send `names`, `include_summary`, `num_speakers`; the API ignores unknown fields on purpose.

## Dev and tests

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests          # no model needed
WHISPER_MODEL=tiny .venv/bin/python run.py
```

API tests use `FakePipeline`; filter tests stub `WhisperTranscriber._decode`. Check a change that touches decoding or Whisper with real audio (webm/opus and mp4), not only the stubs — the PyAV break above passed every unit test.

## Conventions

- Comments say *why* in a sentence or two; docstrings say what a function returns or guarantees.
- Tunables go in `Config` with a comment on what the number means.
- Conventional Commits with scope: `feat(asr): …`, `fix(api): …`.
