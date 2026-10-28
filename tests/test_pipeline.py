"""The pipeline puts tracks on one timeline (decoding and Whisper stubbed)."""
import numpy as np
import pytest

import src.utils.pipeline as pl
from src.core.exceptions import AudioFileError
from src.services.transcriber import WhisperTranscriber
from src.utils.audio import decode


class FakeTranscriber:
    """Says one line at 1-2 s for every recording, named after its length."""
    merge_consecutive_segments = staticmethod(WhisperTranscriber.merge_consecutive_segments)

    def transcribe_speech(self, samples, sample_rate, language=None):
        return [{'start': 1.0, 'end': 2.0, 'text': f'{len(samples)} {language}', 'confidence': 0.9}]


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(pl, 'decode', lambda audio: np.zeros(len(audio) * 16000, dtype=np.float32))
    pipe = pl.TranscriptionPipeline.__new__(pl.TranscriptionPipeline)
    pipe.transcriber = FakeTranscriber()
    return pipe


def test_tracks_are_shifted_by_their_offset_and_labelled(pipeline):
    result = pipeline.process([
        {'label': 'A', 'offset': 0.0, 'audio': b'xx'},
        {'label': 'B', 'offset': 8.5, 'audio': b'x'},
    ], language='id')
    assert [(s['speaker'], s['start'], s['end']) for s in result['segments']] == [
        ('A', 1.0, 2.0), ('B', 9.5, 10.5),
    ]
    assert result['transcript'] == 'A: 32000 id\nB: 16000 id'
    assert result['audio_info']['duration_seconds'] == 9.5
    assert result['statistics']['speakers']['B'] == {'words': 2, 'speech_seconds': 1.0, 'turns': 1}


def test_an_unlabelled_file_gives_plain_text(pipeline):
    result = pipeline.process([{'label': None, 'offset': 0.0, 'audio': b'x'}], language=None)
    assert result['transcript'] == '16000 id'  # Config.LANGUAGE is the default
    assert result['statistics']['speakers'] == {}
    assert result['statistics']['total_words'] == 2


def test_a_turn_of_one_speaker_stays_on_one_line():
    segments = [
        {'speaker': 'A', 'text': 'halo'}, {'speaker': 'A', 'text': 'bu'}, {'speaker': 'B', 'text': 'ya'},
    ]
    assert pl.transcript_of(segments) == 'A: halo bu\nB: ya'


def test_audio_that_cannot_be_read_is_reported():
    with pytest.raises(AudioFileError):
        decode(b'this is not audio')
