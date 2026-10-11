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

    assert segments == [{'start': 10.0, 'end': 14.5, 'text': 'Halo Bu, apa kabar', 'confidence': 0.741}]


def test_drops_a_stock_line_whisper_is_unsure_of_and_shortens_loops():
    rate = 16000
    t = np.arange(rate * 20) / rate
    samples = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    transcriber = WhisperTranscriber.__new__(WhisperTranscriber)
    transcriber.transcribe_tensor = lambda *args, **kwargs: {'segments': [
        {'start': 0.0, 'end': 2.0, 'text': ' Terima kasih.', 'no_speech_prob': 0.2, 'avg_logprob': -0.9},
        {'start': 3.0, 'end': 5.0, 'text': ' Terima kasih ya Bu.', 'no_speech_prob': 0.05, 'avg_logprob': -0.1},
        {'start': 5.5, 'end': 5.8, 'text': ' terima kasih', 'no_speech_prob': 0.05, 'avg_logprob': -0.1},
        {'start': 6.0, 'end': 9.0, 'text': ' udah udah udah udah udah udah', 'no_speech_prob': 0.1, 'avg_logprob': -0.4},
    ]}

    segments = transcriber.transcribe_speech(samples, rate)

    assert [s['text'] for s in segments] == ['Terima kasih ya Bu.', 'udah udah …']


def test_merging_weights_confidence_by_duration():
    merged = WhisperTranscriber.merge_consecutive_segments([
        {'speaker': 'A', 'start': 0.0, 'end': 3.0, 'duration': 3.0, 'text': 'halo', 'confidence': 0.9},
        {'speaker': 'A', 'start': 3.2, 'end': 4.2, 'duration': 1.0, 'text': 'bu', 'confidence': 0.5},
    ])
    assert merged == [{'speaker': 'A', 'start': 0.0, 'end': 4.2, 'duration': 4.2, 'text': 'halo bu', 'confidence': 0.8}]


def test_a_window_spanning_two_replies_is_split_at_the_pause():
    from types import SimpleNamespace as W
    from src.services.transcriber import split_on_pauses

    seg = W(start=7.9, end=30.0, text=' kamu sehat? Insyaallah bisa.', avg_logprob=-0.07, no_speech_prob=0.01, words=[
        W(start=7.9, end=8.3, word=' kamu', probability=0.9),
        W(start=8.4, end=8.9, word=' sehat?', probability=0.8),
        W(start=26.0, end=26.6, word=' Insyaallah', probability=0.5),
        W(start=26.7, end=27.0, word=' bisa.', probability=0.3),
    ])
    parts = split_on_pauses(seg)
    assert [(p['start'], p['text'].strip(), p['confidence']) for p in parts] == [
        (7.9, 'kamu sehat?', 0.85),
        (26.0, 'Insyaallah bisa.', 0.4),
    ]


def test_speech_parted_by_a_long_silence_is_read_separately(monkeypatch):
    import src.services.transcriber as tr
    rate = 16000
    monkeypatch.setattr(tr, 'get_speech_timestamps', lambda audio, opts: [
        {'start': 1 * rate, 'end': 4 * rate},
        {'start': 5 * rate, 'end': 7 * rate},     # 1 s pause: same region
        {'start': 25 * rate, 'end': 27 * rate},   # 18 s silence: new region
        {'start': 28 * rate, 'end': 58 * rate},   # would pass 28 s: new region
    ])
    assert tr.speech_regions(np.zeros(60 * rate, dtype=np.float32), rate) == [
        (1 * rate, 7 * rate), (25 * rate, 27 * rate), (28 * rate, 58 * rate),
    ]
