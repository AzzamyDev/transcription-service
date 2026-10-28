"""Job queue and API, with a stand-in pipeline (no GPU or models needed).

Run: python -m pytest tests
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
        self.transcriber = type('T', (), {'model_size': 'fake'})()

    def process(self, tracks, language=None):
        self.release.wait(5)
        self.calls.append(([(t['label'], t['offset']) for t in tracks], language))
        segments = [
            {'speaker': t['label'], 'start': t['offset'], 'end': t['offset'] + 1,
             'duration': 1, 'text': 'halo', 'confidence': 0.9}
            for t in tracks
        ]
        return {
            'transcript': 'halo',
            'segments': segments,
            'language': language,
            'audio_info': {'duration_seconds': 10, 'sample_rate': 16000},
            'statistics': {'total_words': 1, 'speech_seconds': 1, 'speakers': {}},
            'timings': {'total': 1.0},
        }


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
        'tracks': [(io.BytesIO(b'a'), 'left.mp3'), (io.BytesIO(b'b'), 'right.mp3')],
        'labels': ['LEFT', 'RIGHT:c1'],
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
    assert [s['speaker'] for s in done['result']['segments']] == ['LEFT', 'RIGHT:c1']
    assert done['result']['segments'][1]['start'] == 8.28
    assert client.pipeline.calls == [([('LEFT', 0.0), ('RIGHT:c1', 8.28)], None)]


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


def test_unknown_job_is_404(client):
    assert client.get('/jobs/nope').status_code == 404


def test_one_file_without_a_label_is_transcribed_with_its_language(client):
    res = client.post('/jobs', data={
        'file': (io.BytesIO(b'x'), 'answer.webm'),
        'language': 'en',
    }, content_type='multipart/form-data')
    done = wait_done(client, res.get_json()['data']['job_id'])
    assert done['status'] == 'done'
    assert client.pipeline.calls == [([(None, 0.0)], 'en')]


def test_fields_of_older_clients_are_ignored(client):
    res = client.post('/jobs', data={
        'file': (io.BytesIO(b'x'), 'call.mp3'),
        'num_speakers': '2',
        'include_summary': 'false',
        'names': 'Ibu',
    }, content_type='multipart/form-data')
    assert res.status_code == 202


@pytest.mark.parametrize('language', ['indonesian', 'I', 'id-ID', '12'])
def test_a_bad_language_code_is_refused(client, language):
    res = client.post('/jobs', data={
        'file': (io.BytesIO(b'x'), 'a.mp3'), 'language': language,
    }, content_type='multipart/form-data')
    assert res.status_code == 400


def test_transcribe_answers_synchronously_with_segments(client):
    res = client.post('/transcribe', data={
        'file': (io.BytesIO(b'x'), 'answer.mp3'),
    }, content_type='multipart/form-data')
    body = res.get_json()
    assert res.status_code == 200
    assert body['success'] is True
    assert body['data']['transcript'] == 'halo'
    assert body['data']['segments'][0]['text'] == 'halo'
    assert 'request_total' in body['data']['timings']


def test_the_api_key_guards_work_but_not_probes(client, monkeypatch):
    monkeypatch.setattr(api.Config, 'API_KEY', 'secret')
    form = {'file': (io.BytesIO(b'x'), 'a.mp3')}
    assert client.post('/jobs', data=form, content_type='multipart/form-data').status_code == 401
    assert client.get('/jobs/nope').status_code == 401
    assert client.get('/ping').status_code == 200
    assert client.get('/health').status_code == 200
    assert client.get('/api/swagger.json').status_code == 200
    res = client.post('/jobs', data={'file': (io.BytesIO(b'x'), 'a.mp3')},
                      content_type='multipart/form-data', headers={'X-Api-Key': 'secret'})
    assert res.status_code == 202


def test_health_reports_loading_until_the_model_is_ready(client, monkeypatch):
    assert client.get('/health').get_json()['model'] == 'fake'
    monkeypatch.setattr(api, 'pipeline', None)
    assert client.get('/health').status_code == 503
    res = client.post('/jobs', data={'file': (io.BytesIO(b'x'), 'a.mp3')},
                      content_type='multipart/form-data')
    assert res.status_code == 503


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
