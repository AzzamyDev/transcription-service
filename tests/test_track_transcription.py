"""Silence filtering on per-speaker tracks (Whisper is stubbed)."""
import numpy as np

from src.services.transcriber import WhisperTranscriber


def test_drops_text_whisper_invents_over_silence():
    rate = 16000
    samples = np.zeros(rate * 30, dtype=np.float32)
    t = np.arange(rate * 5) / rate
    samples[rate * 10:rate * 15] = 0.2 * np.sin(2 * np.pi * 220 * t)  # speech at 10-15 s

    transcriber = WhisperTranscriber.__new__(WhisperTranscriber)
    transcriber.transcribe_tensor = lambda *args, **kwargs: {'segments': [
        {'start': 1.0, 'end': 4.0, 'text': ' Terima kasih telah menonton', 'no_speech_prob': 0.1, 'avg_logprob': -0.2},
        {'start': 10.0, 'end': 14.5, 'text': ' Halo Bu, apa kabar', 'no_speech_prob': 0.05, 'avg_logprob': -0.3},
        {'start': 11.0, 'end': 12.0, 'text': ' hmm', 'no_speech_prob': 0.9, 'avg_logprob': -1.5},
        {'start': 20.0, 'end': 25.0, 'text': '   ', 'no_speech_prob': 0.1, 'avg_logprob': -0.2},
    ]}

    segments = transcriber.transcribe_speech(samples, rate)

    assert segments == [{'start': 10.0, 'end': 14.5, 'text': 'Halo Bu, apa kabar'}]
