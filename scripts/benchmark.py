#!/usr/bin/env python3
"""
Throughput benchmark for a running deployment (one or many replicas).

Sends recordings to every replica at once, the way a busy client would, and
reports how fast the whole fleet transcribes. Run it once per configuration
(model, beam size, replicas x threads), change the service's env between runs,
and compare the runs with --report.

    scripts/benchmark.py rekaman/ --urls http://localhost:7890 --replicas 8 --repeat 4 \
        --need-minutes 20000 --hours 16 --out hasil.jsonl
    scripts/benchmark.py --report hasil.jsonl
"""
import argparse
import json
import os
import queue
import re
import statistics
import sys
import threading
import time
import uuid
from pathlib import Path

import requests

AUDIO_EXT = {'.mp3', '.wav', '.m4a', '.mp4', '.ogg', '.opus', '.webm', '.flac'}


def expand_urls(spec):
    """'http://h:7890-7893,http://other:8000' -> one URL per replica."""
    urls = []
    for part in spec.split(','):
        part = part.strip().rstrip('/')
        match = re.match(r'^(.*:)(\d+)-(\d+)$', part)
        if match:
            head, first, last = match.group(1), int(match.group(2)), int(match.group(3))
            urls += [f'{head}{port}' for port in range(first, last + 1)]
        elif part:
            urls.append(part)
    return urls


def find_audio(paths):
    files = []
    for path in map(Path, paths):
        if path.is_dir():
            files += sorted(p for p in path.iterdir() if p.suffix.lower() in AUDIO_EXT)
        elif path.is_file():
            files.append(path)
        else:
            sys.exit(f'Tidak ditemukan: {path}')
    if not files:
        sys.exit('Tidak ada file audio.')
    return files


class Replica:
    def __init__(self, url, api_key):
        self.url = url
        self.headers = {'X-Api-Key': api_key} if api_key else {}

    def health(self):
        res = requests.get(f'{self.url}/health', timeout=10)
        res.raise_for_status()
        return res.json()

    def run(self, path, key, timeout):
        """Submit one recording and wait for it. Returns a record dict."""
        began = time.time()
        with open(path, 'rb') as audio:
            while True:
                res = requests.post(
                    f'{self.url}/jobs', headers=self.headers, files={'file': audio},
                    data={'job_key': key}, timeout=120,
                )
                if res.status_code != 503:  # 503: queue full, try again shortly
                    break
                audio.seek(0)
                time.sleep(2)
                if time.time() - began > timeout:
                    return self._failed(path, 'antrian penuh terlalu lama', began)
        if res.status_code != 202:
            return self._failed(path, f'{res.status_code} {res.text[:120]}', began)
        job_id = res.json()['data']['job_id']

        while time.time() - began < timeout:
            time.sleep(1)
            res = requests.get(f'{self.url}/jobs/{job_id}', headers=self.headers, timeout=30)
            if res.status_code != 200:
                return self._failed(path, f'poll {res.status_code}', began)
            job = res.json()['data']
            if job['status'] == 'failed':
                return self._failed(path, job.get('error', 'failed'), began)
            if job['status'] == 'done':
                return {
                    'replica': self.url,
                    'file': path.name,
                    'ok': True,
                    'audio_seconds': job['result']['audio_info']['duration_seconds'],
                    'processing_seconds': job['finished_at'] - job['started_at'],
                    'latency_seconds': time.time() - began,
                    'transcript': job['result']['transcript'],
                }
        return self._failed(path, 'timeout', began)

    def _failed(self, path, error, began):
        return {'replica': self.url, 'file': path.name, 'ok': False, 'error': error,
                'latency_seconds': time.time() - began}


def percentile(values, p):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * p))] if ordered else 0.0


def run_benchmark(replicas, files, repeat, inflight, timeout, warmup):
    run_id = uuid.uuid4().hex[:8]
    if warmup:
        # The first decode on a replica is slower; keep it out of the numbers.
        print('Pemanasan: satu rekaman per replika...', flush=True)
        threads = [
            threading.Thread(target=r.run, args=(files[0], f'bench-{run_id}-warm-{i}', timeout))
            for i, r in enumerate(replicas)
        ]
        [t.start() for t in threads]
        [t.join() for t in threads]

    tasks = queue.Queue()
    for n, path in enumerate(files * repeat):
        tasks.put((n, path))
    total = tasks.qsize()
    records, lock = [], threading.Lock()

    def worker(replica):
        # A replica always has `inflight` jobs queued, so it never idles and
        # a fast replica takes more of the work.
        while True:
            try:
                n, path = tasks.get_nowait()
            except queue.Empty:
                return
            record = replica.run(path, f'bench-{run_id}-{n}', timeout)
            record['n'] = n
            with lock:
                records.append(record)

    started = time.time()
    threads = [
        threading.Thread(target=worker, args=(r,), daemon=True)
        for r in replicas for _ in range(inflight)
    ]
    [t.start() for t in threads]
    while any(t.is_alive() for t in threads):
        time.sleep(3)
        with lock:
            done = len(records)
        print(f'\r  {done}/{total} selesai, {time.time() - started:.0f}s', end='', flush=True)
    print()
    return records, time.time() - started


def summarize(records, wall, replicas_info, args):
    count = args.replicas or len(replicas_info)
    good = [r for r in records if r['ok']]
    audio = sum(r['audio_seconds'] for r in good)
    speeds = [r['audio_seconds'] / r['processing_seconds'] for r in good if r['processing_seconds'] > 0]
    latencies = [r['latency_seconds'] for r in good]
    health = replicas_info[0]
    summary = {
        'tag': args.tag or f"{health.get('model')}-b{health.get('beam_size', '?')}-"
                           f"{count}x{health.get('cpu_threads')}",
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': health.get('model'),
        'beam_size': health.get('beam_size'),
        'compute_type': health.get('compute_type'),
        'replicas': count,
        'cpu_threads': health.get('cpu_threads'),
        'jobs': len(records),
        'failed': len(records) - len(good),
        'audio_minutes': round(audio / 60, 2),
        'wall_seconds': round(wall, 1),
        'fleet_realtime': round(audio / wall, 2) if wall else 0,
        'replica_realtime_median': round(statistics.median(speeds), 2) if speeds else 0,
        'replica_realtime_p10': round(percentile(speeds, 0.10), 2),
        'latency_p50': round(percentile(latencies, 0.50), 1),
        'latency_p95': round(percentile(latencies, 0.95), 1),
    }
    if args.need_minutes and summary['fleet_realtime']:
        summary['projected_hours'] = round(
            args.need_minutes * 60 / summary['fleet_realtime'] / 3600, 1
        )
    return summary


def print_summary(s, records, args):
    print(f"\n=== {s['tag']} ===")
    print(f"model {s['model']} ({s['compute_type']}), beam {s['beam_size']}, "
          f"{s['replicas']} replika x {s['cpu_threads']} thread")
    print(f"{s['jobs']} job ({s['failed']} gagal), {s['audio_minutes']} menit audio, "
          f"selesai dalam {s['wall_seconds']} dtk")
    print(f"Kecepatan gabungan : {s['fleet_realtime']}x realtime")
    print(f"Per replika        : median {s['replica_realtime_median']}x, "
          f"p10 {s['replica_realtime_p10']}x (job paling lambat)")
    print(f"Latensi job        : p50 {s['latency_p50']} dtk, p95 {s['latency_p95']} dtk")
    per = {}
    for r in records:
        if r['ok']:
            entry = per.setdefault(r['replica'], [0, 0.0])
            entry[0] += 1
            entry[1] += r['audio_seconds']
    for url, (jobs, seconds) in sorted(per.items()):
        print(f"  {url:<32} {jobs:>4} job  {seconds / 60:>7.1f} menit")
    for r in records:
        if not r['ok']:
            print(f"  GAGAL {r['file']} @ {r['replica']}: {r['error']}")
    if 'projected_hours' in s:
        verdict = ''
        if args.hours:
            verdict = ' -> MEMENUHI target' if s['projected_hours'] <= args.hours \
                else f" -> TIDAK memenuhi target {args.hours} jam"
        print(f"Proyeksi {args.need_minutes:.0f} menit audio: {s['projected_hours']} jam{verdict}")


def report(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    cols = [('tag', 'konfigurasi', 30), ('beam_size', 'beam', 4), ('replicas', 'repl', 4),
            ('cpu_threads', 'thr', 3), ('jobs', 'job', 4), ('failed', 'gagal', 5),
            ('fleet_realtime', 'x realtime', 10), ('replica_realtime_median', '/replika', 8),
            ('latency_p95', 'p95 dtk', 7), ('projected_hours', 'proyeksi jam', 12)]
    print('  '.join(f'{title:<{w}}' for _, title, w in cols))
    for row in sorted(rows, key=lambda r: -r['fleet_realtime']):
        print('  '.join(f'{str(row.get(key, "-")):<{w}}' for key, _, w in cols))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('audio', nargs='*', help='file audio atau folder berisi rekaman asli')
    parser.add_argument('--urls', default='http://localhost:8000',
                        help='replika, dipisah koma; rentang port: http://host:7890-7897')
    parser.add_argument('--api-key', default=os.getenv('API_KEY', ''), help='default: env API_KEY')
    parser.add_argument('--repeat', type=int, default=3, help='berapa kali tiap file dikirim (beban)')
    parser.add_argument('--replicas', type=int,
                        help='jumlah replika di balik satu URL gateway (default: jumlah URL)')
    parser.add_argument('--inflight', type=int, default=2,
                        help='job yang antre per replika (per URL; di balik gateway otomatis dikali --replicas)')
    parser.add_argument('--timeout', type=int, default=3600, help='batas tunggu per job (detik)')
    parser.add_argument('--no-warmup', action='store_true')
    parser.add_argument('--need-minutes', type=float, help='total menit audio yang harus diproses')
    parser.add_argument('--hours', type=float, help='batas waktu (jam) untuk --need-minutes')
    parser.add_argument('--tag', help='nama konfigurasi di laporan')
    parser.add_argument('--out', help='tambahkan ringkasan ke file .jsonl ini')
    parser.add_argument('--save-transcripts', help='simpan transkrip ke folder ini, untuk dibaca manual')
    parser.add_argument('--report', help='tampilkan tabel perbandingan dari file .jsonl, lalu keluar')
    args = parser.parse_args()

    if args.report:
        return report(args.report)
    if not args.audio:
        parser.error('beri file audio atau folder')

    replicas = [Replica(url, args.api_key) for url in expand_urls(args.urls)]
    info = []
    for replica in replicas:
        try:
            info.append(replica.health())
        except Exception as e:
            sys.exit(f'{replica.url} tidak siap: {e}')
    differing = {(h.get('model'), h.get('cpu_threads'), h.get('beam_size')) for h in info}
    if len(differing) > 1:
        print(f'Peringatan: replika tidak seragam: {differing}', file=sys.stderr)

    files = find_audio(args.audio)
    print(f'{args.replicas or len(replicas)} replika, {len(files)} file x {args.repeat} = {len(files) * args.repeat} job')
    # Behind a gateway one URL fronts many replicas and needs that many jobs queued.
    per_url = args.inflight * (args.replicas or len(replicas)) // len(replicas)
    records, wall = run_benchmark(replicas, files, args.repeat, per_url,
                                  args.timeout, not args.no_warmup)
    summary = summarize(records, wall, info, args)
    print_summary(summary, records, args)

    if args.save_transcripts:
        folder = Path(args.save_transcripts) / summary['tag']
        folder.mkdir(parents=True, exist_ok=True)
        for r in sorted(records, key=lambda r: r['n']):
            if r['ok'] and not (folder / f"{Path(r['file']).stem}.txt").exists():
                (folder / f"{Path(r['file']).stem}.txt").write_text(r['transcript'] + '\n')
        print(f'Transkrip: {folder}')
    if args.out:
        with open(args.out, 'a') as out:
            out.write(json.dumps(summary) + '\n')
        print(f'Ringkasan ditambahkan ke {args.out}')
    sys.exit(1 if summary['failed'] else 0)


if __name__ == '__main__':
    main()
