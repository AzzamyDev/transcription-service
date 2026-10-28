# Panduan integrasi

Untuk engineer yang menghubungkan sistem (backend) ke service speech-to-text ini. Service menerima rekaman suara dan mengembalikan teks berikut timestamp. Dokumen ini menjelaskan alur, kontrak API, dan cara klien menanganinya dengan benar. Spesifikasi lengkap ada di `/docs` (Swagger) pada service yang berjalan.

## Alur

Transkripsi lambat (puluhan detik sampai menit), jadi API bersifat asinkron:

```
klien                               service
  │  POST /jobs  (audio, job_key)      │
  │ ─────────────────────────────────▶ │  202 { job_id, status: queued }
  │                                    │  … antre, lalu diproses satu per satu …
  │  GET /jobs/{job_id}   (polling)    │
  │ ─────────────────────────────────▶ │  { status: queued | processing }
  │  GET /jobs/{job_id}                │
  │ ─────────────────────────────────▶ │  { status: done, result: { … } }
```

Tidak ada webhook; klien melakukan polling. Untuk audio pendek ada `POST /transcribe` yang menunggu sampai selesai (lihat di bawah).

## Dasar

| | |
| --- | --- |
| Base URL | `http://<host>:<port>`; satu alamat gateway yang membagi ke semua replika (lihat [Banyak replika](#banyak-replika)) |
| Autentikasi | Jika service memakai `API_KEY`, kirim header `X-Api-Key: <kunci>` di semua request kecuali `/ping`, `/health`, `/docs`. Salah atau kosong → `401`. |
| Format respons | JSON `{ "success": bool, "message": string, "data": … }`. Pada error `success` false dan `data` null. |
| Format audio | mp3, wav, m4a/mp4, ogg, opus, webm, flac (apa pun yang dibaca FFmpeg), mono atau stereo, sample rate bebas. Browser: `audio/webm;codecs=opus` dan `audio/mp4` berfungsi. |
| Ukuran | Maksimum `MAX_AUDIO_SIZE_MB` per request (default 1000) → `413`. |

## Endpoint

### `POST /jobs`

`multipart/form-data`. Jawaban datang seketika dengan `202`.

**Satu rekaman:**

| Field | Wajib | Keterangan |
| --- | --- | --- |
| `file` | ya | File audio |
| `label` | tidak | Nama pembicara; mengisi `speaker` di hasil. Tanpa label, `speaker` bernilai `null` dan `transcript` berupa teks polos. |
| `language` | tidak | Kode bahasa (`id`, `en`, …). Default mengikuti `WHISPER_LANGUAGE` service. Selain 2–3 huruf kecil → `400`. |
| `job_key` | tidak, **disarankan** | Kunci idempotensi, lihat [Idempotensi](#idempotensi-dan-retry) |

**Beberapa pembicara, satu file per pembicara** (misalnya rekaman panggilan dua sisi). Ganti `file` dengan field berulang yang urutannya sejajar:

| Field | Keterangan |
| --- | --- |
| `tracks` | File audio, satu per pembicara |
| `labels` | Label tiap track (wajib, tidak boleh kosong) |
| `offsets` | Detik ke berapa tiap track mulai pada timeline bersama (≥ 0). Pakai `0` jika semua mulai bersamaan. |

Jumlah `labels` dan `offsets` harus sama dengan jumlah `tracks`. Semua baris dari semua track digabung dan diurutkan menurut waktu, dengan `speaker` berisi label track.

**Respons `202`:**

```json
{
  "success": true,
  "message": "Queued",
  "data": {
    "job_id": "9f1c…",
    "job_key": "jawaban-42",
    "status": "queued",
    "position": 3,
    "created_at": 1767000000.1,
    "started_at": null,
    "finished_at": null
  }
}
```

`position` adalah jumlah job di depan job ini (termasuk yang sedang diproses).

### `GET /jobs/{job_id}`

`data.status`:

| Status | Arti | Tindakan klien |
| --- | --- | --- |
| `queued` | Menunggu giliran (`data.position`) | Polling lagi |
| `processing` | Sedang ditranskripsi | Polling lagi |
| `done` | Selesai; hasil di `data.result` | Ambil hasil |
| `failed` | Gagal; alasan di `data.error` | Lihat [Error](#error) |

**Hasil (`data.result`):**

```json
{
  "transcript": "PESERTA: Selamat pagi, nama saya Budi.\nPEWAWANCARA: Terima kasih, Budi.",
  "segments": [
    {"speaker": "PESERTA", "start": 0.0, "end": 10.64, "duration": 10.64,
     "text": "Selamat pagi, nama saya Budi.", "confidence": 0.83},
    {"speaker": "PEWAWANCARA", "start": 12.0, "end": 15.84, "duration": 3.84,
     "text": "Terima kasih, Budi.", "confidence": 0.8}
  ],
  "language": "id",
  "audio_info": {"duration_seconds": 15.84, "sample_rate": 16000},
  "statistics": {
    "total_words": 7,
    "speech_seconds": 14.48,
    "speakers": {"PESERTA": {"words": 5, "speech_seconds": 10.64, "turns": 1}}
  },
  "timings": {"decoding": 0.02, "transcription": 1.4, "total": 1.4}
}
```

- `start` / `end` dalam detik pada timeline bersama (sudah ditambah `offset` track).
- `confidence` 0–1 per baris. Nilai rendah (kira-kira di bawah 0,5) menandai baris yang patut dicek manusia; ambangnya silakan Anda tentukan dari data sendiri.
- Susun teks sendiri dari `segments` bila perlu format khusus; `transcript` hanya gabungan siap pakai.
- **Rekaman tanpa ucapan tetap `done`**: `segments` kosong dan `transcript` `""`. Perlakukan itu sebagai hasil sah ("tidak ada suara"), bukan error.
- `statistics.speakers` hanya berisi pembicara yang berlabel.

### `POST /transcribe`

Input sama dengan `/jobs`, tetapi koneksi ditahan sampai selesai dan `data` langsung berisi hasil di atas (plus `timings.request_total`). Memakai antrian yang sama. Cocok untuk skrip dan audio pendek; untuk sistem produksi pakai `/jobs`.

Batasan: maksimum 4 request menunggu sekaligus per replika (`503` bila lebih), dan `504` setelah `SYNC_MAX_WAIT_SECONDS` (default 30 menit). Pesan `504` memuat `job_id`; teruskan dengan polling `GET /jobs/{job_id}`.

### `GET /ping` dan `GET /health`

`/ping` selalu `200` selama proses hidup. `/health` `200` setelah model termuat (`503` saat masih memuat, yang bisa memakan beberapa menit pada start pertama). Pakai `/health` untuk readiness dan load balancer.

## Idempotensi dan retry

Kirim `job_key` yang **stabil dan unik per rekaman** (misalnya `voice-{id_jawaban}`). Aturannya:

- Mengirim ulang `job_key` yang sama mengembalikan job yang sudah `queued`, `processing`, atau `done`, tanpa memproses ulang. Aman untuk retry setelah timeout jaringan.
- Job yang `failed` tidak dipakai ulang: kirim ulang dengan key yang sama membuat job baru.
- Hasil disimpan **24 jam** setelah selesai; setelah itu key bebas dan job hilang.
- Job hanya ada di memori replika. Jika replika restart, `GET /jobs/{job_id}` menjawab `404`. Simpan audio sumber (misalnya di object storage) sampai transkrip Anda simpan sendiri, dan **kirim ulang saat menerima `404`**.

Rekomendasi sisi klien: simpan hasil transkrip di database Anda begitu `done`; jangan bergantung pada service sebagai penyimpanan.

## Banyak replika

Satu replika memproses satu job pada satu waktu; kapasitas diperbesar dengan menambah replika. Di deployment Docker Compose, replika berada di belakang gateway nginx dengan **satu alamat**, dan klien tidak perlu tahu jumlah replikanya:

- Job baru dibagi bergantian ke semua replika.
- `GET /jobs/{job_id}` otomatis diteruskan ke replika yang memegang job itu (`job_id` diawali ID replikanya, mis. `9b8f27d956df-…`). Perlakukan `job_id` sebagai string buram: jangan diurai, dan jangan membuatnya sendiri.
- **`job_key` hanya mencegah duplikat per replika.** Kirim ulang `job_key` yang sama lewat gateway bisa jatuh ke replika lain dan membuat job kedua (audio diproses dua kali, hasilnya sama). Karena itu: setelah menerima `job_id`, polling saja dan jangan kirim ulang; kirim ulang hanya bila respons `POST` hilang atau polling menjawab `404`. Simpan transkrip pertama yang selesai untuk satu `job_key`.
- Jika memakai replika tanpa gateway (tiap replika di port sendiri), simpan URL replika yang menerima job bersama `job_id` dan polling ke URL itu saja.
- `503` (antrian penuh di replika itu, atau model masih memuat): coba lagi dengan backoff.

## Error

| Status | Penyebab | Tindakan |
| --- | --- | --- |
| `400` | Field tidak valid: tanpa file, file kosong, label atau offset salah, bahasa salah | Perbaiki request; jangan retry apa adanya |
| `401` | `X-Api-Key` kosong atau salah | Periksa kunci |
| `404` | `job_id` tidak dikenal (kedaluwarsa, replika restart, atau replika salah) | Kirim ulang job |
| `413` | Melebihi `MAX_AUDIO_SIZE_MB` | Perkecil atau pecah audio |
| `500` | Gagal di `/transcribe` (misalnya audio tidak bisa didecode) | Lihat `message`; retry bila sementara |
| `503` | Antrian penuh (`ASR_MAX_PENDING`, default 30 per replika), model masih memuat, atau penunggu `/transcribe` penuh | Retry dengan backoff atau pindah replika |
| `504` | `/transcribe` melewati batas tunggu | Polling `/jobs/{job_id}` |

Job `failed` membawa `data.error`. Rekaman yang tidak bisa dibaca baru ketahuan saat job diproses, jadi muncul sebagai `failed` (dengan pesan `Could not read the audio: …`), bukan `400` di awal. Kegagalan seperti itu bersifat permanen untuk file yang sama; jangan retry tanpa batas.

## Pola polling yang disarankan

- Mulai polling 2–3 detik setelah submit; naikkan jarak secara bertahap (misalnya 2 → 5 → 10 detik, maksimum 15). Panjang antrian terlihat dari `position`.
- Beri batas total (misalnya 2× durasi antrian yang diharapkan) lalu anggap job bermasalah dan kirim ulang.
- Jika memiliki banyak job, gunakan antrian sendiri di sisi klien (misalnya BullMQ atau Celery) dan batasi job yang sedang berjalan (lihat [Banyak replika](#banyak-replika)); antrian tiap replika dibatasi `ASR_MAX_PENDING`.

## Contoh

### curl

```bash
# satu rekaman
curl -H "X-Api-Key: $KEY" -F file=@jawaban.webm -F job_key=jawaban-42 http://host:8001/jobs
curl -H "X-Api-Key: $KEY" http://host:8001/jobs/<job_id>

# dua pembicara
curl -H "X-Api-Key: $KEY" \
  -F tracks=@agen.mp3 -F labels=AGEN -F offsets=0 \
  -F tracks=@penelepon.mp3 -F labels=PENELEPON -F offsets=3.5 \
  -F job_key=panggilan-7 http://host:8001/jobs
```

### Python

```python
import time
import requests

BASE = "http://host:8001"
HEADERS = {"X-Api-Key": "kunci"}  # hapus bila service tanpa API_KEY


def transcribe(path, job_key, label=None, language=None, timeout=1800):
    data = {"job_key": job_key}
    if label:
        data["label"] = label
    if language:
        data["language"] = language
    with open(path, "rb") as audio:
        res = requests.post(f"{BASE}/jobs", headers=HEADERS, data=data,
                            files={"file": audio}, timeout=60)
    res.raise_for_status()
    job_id = res.json()["data"]["job_id"]

    deadline, wait = time.time() + timeout, 2
    while time.time() < deadline:
        time.sleep(wait)
        wait = min(wait * 1.5, 15)
        res = requests.get(f"{BASE}/jobs/{job_id}", headers=HEADERS, timeout=30)
        if res.status_code == 404:  # replika restart: kirim ulang
            return transcribe(path, job_key, label, language, timeout)
        job = res.json()["data"]
        if job["status"] == "done":
            return job["result"]
        if job["status"] == "failed":
            raise RuntimeError(job["error"])
    raise TimeoutError(f"job {job_id} belum selesai")


result = transcribe("jawaban.webm", "jawaban-42", label="PESERTA")
for seg in result["segments"]:
    print(f'{seg["start"]:7.2f}  {seg["confidence"]:.2f}  {seg["text"]}')
```

### Node.js (18+)

```js
const BASE = "http://host:8001";
const HEADERS = { "X-Api-Key": "kunci" }; // hapus bila service tanpa API_KEY

async function transcribe(path, jobKey, { label, language } = {}) {
  const { openAsBlob } = await import("node:fs");
  const form = new FormData();
  form.append("file", await openAsBlob(path), path.split("/").pop());
  form.append("job_key", jobKey);
  if (label) form.append("label", label);
  if (language) form.append("language", language);

  let res = await fetch(`${BASE}/jobs`, { method: "POST", headers: HEADERS, body: form });
  if (!res.ok) throw new Error(`submit ${res.status}: ${(await res.json()).message}`);
  const { job_id } = (await res.json()).data;

  for (let wait = 2000; ; wait = Math.min(wait * 1.5, 15000)) {
    await new Promise((r) => setTimeout(r, wait));
    res = await fetch(`${BASE}/jobs/${job_id}`, { headers: HEADERS });
    if (res.status === 404) return transcribe(path, jobKey, { label, language });
    const job = (await res.json()).data;
    if (job.status === "done") return job.result;
    if (job.status === "failed") throw new Error(job.error);
  }
}
```

## Contoh khas: jawaban wawancara satu pembicara

Satu file per jawaban, tanpa label, `job_key` mengikuti ID jawaban:

```
POST /jobs   file=<rekaman>, language=id, job_key=voice-{voiceAnswerId}
GET  /jobs/{job_id}   →  data.result.transcript (teks polos)
```

Simpan `job_id`, simpan `segments` bila Anda perlu timestamp atau `confidence` untuk koreksi manual, lalu hapus ketergantungan pada service setelah transkrip tersimpan.
