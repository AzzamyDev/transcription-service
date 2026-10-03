"""Job queue and API, with a stand-in pipeline (no GPU or models needed).

Run: pip install pytest flask flask-cors flask-swagger-ui python-dotenv numpy
     python -m pytest tests
"""
import io
import threading
import time

import pytest

import src.api.api as api


class FakePipeline:
    def __init__(self):
        self.release = threading.Event()
        self.release.set()
        self.calls = []

    def _result(self, segments):
        return {
            'segments': segments,
            'summary': 'ringkasan',
            'sentiment': 'NEUTRAL',
            'flagged_phrases': [],
            'audio_info': {'duration_seconds': 10, 'duration_minutes': 0.17, 'sample_rate': 16000},
            'statistics': {'total_duration': 10, 'total_words': 2, 'num_speakers': 1, 'speakers': {}},
            'timings': {'total': 1.0},
        }

    def process_tracks(self, tracks, include_summary=True):
        self.release.wait(5)
        self.calls.append(('tracks', [(t['label'], t['offset'], t['name']) for t in tracks]))
        return self._result([
            {'speaker': t['label'], 'start': t['offset'], 'end': t['offset'] + 1, 'duration': 1, 'text': 'halo'}
            for t in tracks
        ])

    def process(self, audio, num_speakers=None, min_speakers=None, max_speakers=None, include_summary=True):
        self.release.wait(5)
        self.calls.append(('mixed', num_speakers))
        return self._result([{'speaker': 'SPEAKER_00', 'start': 0, 'end': 1, 'duration': 1, 'text': 'halo'}])


@pytest.fixture
def client():
    pipeline = FakePipeline()
    api.init(pipeline)
    api.app.config['TESTING'] = True
    with api.app.test_client() as c:
        c.pipeline = pipeline
        yield c


def wait_done(client, job_id, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = client.get(f'/jobs/{job_id}').get_json()['data']
        if data['status'] in ('done', 'failed'):
            return data
        time.sleep(0.05)
    raise AssertionError('job did not finish')


def post_tracks(client, key='sess-1', offsets=('0', '8.28')):
    return client.post('/jobs', data={
        'job_key': key,
        'tracks': [(io.BytesIO(b'a'), 'inmate.mp3'), (io.BytesIO(b'b'), 'family.mp3')],
        'labels': ['INMATE', 'FAMILY:c1'],
        'names': ['Warga Binaan', 'Ibu'],
        'offsets': list(offsets),
    }, content_type='multipart/form-data')


def test_tracks_job_answers_at_once_and_finishes(client):
    client.pipeline.release.clear()
    res = post_tracks(client)
    assert res.status_code == 202
    job = res.get_json()['data']
    assert job['status'] in ('queued', 'processing')
    # The API stays responsive while the job runs.
    assert client.get('/ping').status_code == 200
    client.pipeline.release.set()
    done = wait_done(client, job['job_id'])
    assert done['status'] == 'done'
    assert [s['speaker'] for s in done['result']['segments']] == ['INMATE', 'FAMILY:c1']
    assert done['result']['segments'][1]['start'] == 8.28
    assert client.pipeline.calls == [('tracks', [('INMATE', 0.0, 'Warga Binaan'), ('FAMILY:c1', 8.28, 'Ibu')])]


def test_resending_a_key_returns_the_same_job(client):
    first = post_tracks(client, key='sess-2').get_json()['data']
    second = post_tracks(client, key='sess-2').get_json()['data']
    assert first['job_id'] == second['job_id']
    wait_done(client, first['job_id'])
    assert len(client.pipeline.calls) == 1


def test_rejects_tracks_without_labels_or_offsets(client):
    res = client.post('/jobs', data={
        'tracks': [(io.BytesIO(b'a'), 'a.mp3')],
        'labels': [],
        'offsets': ['0'],
    }, content_type='multipart/form-data')
    assert res.status_code == 400
    assert client.post('/jobs', data={}, content_type='multipart/form-data').status_code == 400


def test_mixed_file_job_passes_speaker_count(client):
    res = client.post('/jobs', data={
        'file': (io.BytesIO(b'x'), 'call.mp3'),
        'num_speakers': '3',
    }, content_type='multipart/form-data')
    done = wait_done(client, res.get_json()['data']['job_id'])
    assert done['status'] == 'done'
    assert client.pipeline.calls == [('mixed', 3)]


def test_unknown_job_is_404(client):
    assert client.get('/jobs/nope').status_code == 404


def test_legacy_transcribe_still_answers_synchronously(client):
    res = client.post('/transcribe', data={
        'file': (io.BytesIO(b'x'), 'call.mp3'),
        'num_speakers': '2',
    }, content_type='multipart/form-data')
    body = res.get_json()
    assert res.status_code == 200
    assert body['success'] is True
    assert 'segments' not in body['data']
    assert body['data']['transcript'] == 'SPEAKER_00: halo'


def test_watchdog_reports_a_job_stuck_past_its_limit():
    from src.services.jobs import JobQueue

    stuck = []
    hang = threading.Event()
    jobs = JobQueue(lambda payload: hang.wait(5), job_timeout_seconds=1, on_stuck=stuck.append)
    job = jobs.submit({'audio': b'x'})
    deadline = time.time() + 5
    while not stuck and time.time() < deadline:
        time.sleep(0.1)
    hang.set()
    assert stuck and stuck[0].id == job.id


def test_full_queue_is_refused_not_buffered():
    from src.services.jobs import JobQueue, QueueFullError

    block = threading.Event()
    jobs = JobQueue(lambda payload: block.wait(5), max_pending=1)
    jobs.submit({'n': 1})  # taken by the worker
    time.sleep(0.2)
    jobs.submit({'n': 2})  # waits
    with pytest.raises(QueueFullError):
        jobs.submit({'n': 3})
    block.set()


def test_legacy_transcribe_refuses_when_too_many_callers_wait(client):
    taken = [api.sync_waiters.acquire(blocking=False) for _ in range(4)]
    try:
        res = client.post('/transcribe', data={'file': (io.BytesIO(b'x'), 'c.mp3')},
                          content_type='multipart/form-data')
        assert res.status_code == 503
        assert client.get('/ping').status_code == 200
    finally:
        for ok in taken:
            if ok:
                api.sync_waiters.release()
