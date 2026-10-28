"""Audio decoding with PyAV (FFmpeg is bundled in its wheel), in memory:
nothing is written to disk. faster-whisper's own decode_audio is not used
because it breaks on newer PyAV releases."""
import itertools
from io import BytesIO

import av
import numpy as np

from src.core.config import Config
from src.core.exceptions import AudioFileError

# Samples resampled at a time, to keep memory flat on long recordings.
GROUP_SAMPLES = 500_000


def decode(audio: bytes) -> np.ndarray:
    """Float32 mono samples at 16 kHz, from any container FFmpeg reads (mp3,
    wav, m4a/mp4, ogg, opus, webm, flac ...)."""
    try:
        resampler = av.audio.resampler.AudioResampler(
            format='s16', layout='mono', rate=Config.TARGET_SAMPLE_RATE
        )
        chunks = []
        with av.open(BytesIO(audio), mode='r') as container:
            frames = _group(_skip_invalid(container.decode(audio=0)))
            for frame in itertools.chain(frames, [None]):  # None flushes
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray().reshape(-1))
    except Exception as e:
        raise AudioFileError(f"Could not read the audio: {e}")
    if not chunks:
        raise AudioFileError("The audio holds no samples")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def _skip_invalid(frames):
    iterator = iter(frames)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            return
        except av.error.InvalidDataError:
            continue


def _group(frames):
    fifo = av.audio.fifo.AudioFifo()
    for frame in frames:
        frame.pts = None  # Ignore timestamp check.
        fifo.write(frame)
        if fifo.samples >= GROUP_SAMPLES:
            yield fifo.read()
    if fifo.samples > 0:
        yield fifo.read()
