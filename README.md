# Speech-to-text service

HTTP service that turns audio into text with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), running on the CPU. Any client can use it: send a recording, poll for the transcript. Audio is decoded in memory and never stored.

- One recording, or several recordings (one per speaker) placed on a common timeline.
- Timestamps and a confidence score per line.
- Silence and text Whisper invents over it ("Terima kasih.", repeated words) are dropped.
- Jobs are queued and run one at a time per process; scale with replicas.

## Run

Requires Python 3.10+.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
cp .env.example .env            # optional
WHISPER_MODEL=tiny .venv/bin/python run.py     # http://localhost:8000/docs
```

The model is downloaded on first start into `MODEL_DIR` (`./models`).

With Docker (CPU only):

```bash
docker compose up --build
```

### Replicas

One container runs one job at a time. For more throughput run replicas on a port range and spread requests across the ports:

```bash
docker compose run --rm asr python -c "from src.services.transcriber import WhisperTranscriber as W; W()"  # download the model once
ASR_REPLICAS=6 ASR_PORTS=8001-8006 WHISPER_CPU_THREADS=4 docker compose up -d
```

Keep replicas × `WHISPER_CPU_THREADS` within the machine's cores. Jobs live in memory of the replica that took them: a client polls the replica that answered, and resubmits when it gets 404 (see below).

## API

Interactive docs at `/docs`. When `API_KEY` is set, send it as `X-Api-Key` (`/ping`, `/health` and `/docs` stay open).

### `POST /jobs` (multipart)

Answers at once with `202`. Send either

| Field | Meaning |
| --- | --- |
| `file` | One recording. Optional `label` names its speaker. |
| `tracks` + `labels` + `offsets` | One recording per speaker. `offsets[i]` is the second at which track *i* starts on the common timeline. |

and optionally `language` (`id`, `en`, ... default `WHISPER_LANGUAGE`) and `job_key`. Sending the same `job_key` again returns the job already queued, running or done for it, so a client can retry safely.

```bash
curl -F file=@answer.webm -F job_key=answer-42 localhost:8000/jobs
curl -F tracks=@agent.mp3 -F labels=AGENT -F offsets=0 \
     -F tracks=@caller.mp3 -F labels=CALLER -F offsets=3.5 localhost:8000/jobs
```

Responses use `{success, message, data}`. `503`: queue full (`ASR_MAX_PENDING`) or model still loading.

### `GET /jobs/{job_id}`

`data.status` is `queued` (`data.position` = jobs ahead) → `processing` → `done` or `failed` (`data.error`). A `done` job has `data.result`:

```json
{
  "transcript": "AGENT: Halo ...\nCALLER: Selamat pagi ...",
  "segments": [{"speaker": "AGENT", "start": 0.0, "end": 4.2, "duration": 4.2, "text": "Halo ...", "confidence": 0.91}],
  "language": "id",
  "audio_info": {"duration_seconds": 12.4, "sample_rate": 16000},
  "statistics": {"total_words": 31, "speech_seconds": 9.8, "speakers": {"AGENT": {"words": 14, "speech_seconds": 4.2, "turns": 1}}},
  "timings": {"decoding": 0.02, "transcription": 1.4, "total": 1.4}
}
```

Without labels `speaker` is `null` and `transcript` is plain text. Finished jobs are kept 24 hours and are lost when the process restarts: resubmit on `404`. A job running past `ASR_JOB_TIMEOUT_MINUTES` is taken as hung and the process exits so the container restarts.

### `POST /transcribe`

Same input; waits and returns the result directly (`data` = the result above). It uses the same queue, answers `504` with a job id after `SYNC_MAX_WAIT_SECONDS` and `503` when more than 4 callers already wait. Use it for short audio.

### `GET /ping`, `GET /health`

`/ping` is liveness. `/health` is `200` once the model is loaded (`503` while loading) and reports model, threads and job counts.

Audio an FFmpeg build can read works: mp3, wav, m4a/mp4, ogg, opus, webm, flac. A recording that cannot be decoded makes its job `failed` (or `/transcribe` answer `500`).

## Configuration

See [.env.example](.env.example). The ones that matter:

| Variable | Default | |
| --- | --- | --- |
| `WHISPER_MODEL` | `large-v3-turbo` | `tiny` … `large-v3`, `large-v3-turbo` |
| `WHISPER_COMPUTE_TYPE` | `int8` | `int8` is the fast CPU choice |
| `WHISPER_CPU_THREADS` | `4` | per process |
| `WHISPER_LANGUAGE` | `id` | empty = detect per recording |
| `WHISPER_BEAM_SIZE` | `5` | lower is faster, a little less accurate |
| `HOTWORDS` | empty | comma separated words to favour |
| `API_KEY` | empty | empty = no authentication |

Pick model and beam size by transcribing a few of your own recordings and comparing speed (`timings`, and the realtime factor in the log) against the text.

## Tests

```bash
.venv/bin/python -m pytest tests
```

No model needed.
