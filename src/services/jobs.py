"""
Background transcription jobs: requests are accepted at once and processed
by worker threads, so no HTTP request ever waits on the GPU.
"""
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


class QueueFullError(Exception):
    """Too many jobs are waiting"""


@dataclass
class Job:
    id: str
    key: Optional[str]
    payload: Any
    status: str = 'queued'  # queued | processing | done | failed
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None


class JobQueue:
    def __init__(
        self,
        handler: Callable[[Any], dict],
        concurrency: int = 1,
        max_pending: int = 200,
        retention_seconds: int = 24 * 3600,
    ):
        self._handler = handler
        self._max_pending = max_pending
        self._retention_seconds = retention_seconds
        self._jobs: Dict[str, Job] = {}
        self._by_key: Dict[str, str] = {}
        self._pending: 'queue.Queue[str]' = queue.Queue()
        self._lock = threading.Lock()
        for i in range(max(1, concurrency)):
            threading.Thread(
                target=self._work, name=f'asr-worker-{i}', daemon=True
            ).start()

    def submit(self, payload: Any, key: Optional[str] = None) -> Job:
        """Queue a job; a key already queued, running or done returns that job."""
        with self._lock:
            self._prune()
            if key and key in self._by_key:
                existing = self._jobs.get(self._by_key[key])
                if existing and existing.status != 'failed':
                    return existing
            if self._pending.qsize() >= self._max_pending:
                raise QueueFullError(
                    f'{self._max_pending} jobs are already waiting'
                )
            job = Job(id=uuid.uuid4().hex, key=key, payload=payload)
            self._jobs[job.id] = job
            if key:
                self._by_key[key] = job.id
        self._pending.put(job.id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def position(self, job: Job) -> int:
        """Jobs ahead of a queued one, counting the one being processed."""
        if job.status != 'queued':
            return 0
        with self._lock:
            return sum(
                1
                for other in self._jobs.values()
                if other.status in ('queued', 'processing')
                and other.created_at < job.created_at
            )

    def stats(self) -> Dict[str, int]:
        with self._lock:
            counts = {'queued': 0, 'processing': 0}
            for job in self._jobs.values():
                if job.status in counts:
                    counts[job.status] += 1
            return counts

    def _work(self):
        while True:
            job = self.get(self._pending.get())
            if job is None:
                continue
            job.status = 'processing'
            job.started_at = time.time()
            try:
                job.result = self._handler(job.payload)
                job.status = 'done'
            except Exception as e:
                traceback.print_exc()
                job.error = str(e) or e.__class__.__name__
                job.status = 'failed'
            finally:
                job.finished_at = time.time()
                # The audio is no longer needed once processed.
                job.payload = None

    def _prune(self):
        cutoff = time.time() - self._retention_seconds
        for job_id, job in list(self._jobs.items()):
            if job.finished_at and job.finished_at < cutoff:
                del self._jobs[job_id]
                if job.key and self._by_key.get(job.key) == job_id:
                    del self._by_key[job.key]
