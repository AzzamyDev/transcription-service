"""
The transcription pipeline: decode each recording, transcribe its speech, put
every line on one timeline and describe the result.
"""
import time
from typing import Dict, List, Optional

from src.core.config import Config
from src.services.transcriber import WhisperTranscriber
from src.utils.audio import decode


class TranscriptionPipeline:
    def __init__(self, whisper_model: Optional[str] = None):
        print("Initializing transcription pipeline", flush=True)
        self.transcriber = WhisperTranscriber(whisper_model)
        print("Pipeline ready", flush=True)

    def process(self, tracks: List[Dict], language: Optional[str] = None) -> Dict:
        """
        Transcribe recordings. Each track is {'label', 'offset', 'audio'}:
        'audio' is the file's bytes, 'offset' is when it starts on the common
        timeline in seconds, and 'label' (may be None) names its speaker.
        A call recorded one track per speaker gives a labelled transcript; a
        single recording with no label gives plain text.
        """
        language = language or Config.LANGUAGE
        started = time.time()
        timings = {}

        step = time.time()
        decoded = []
        duration = 0.0
        for track in tracks:
            samples = decode(track['audio'])
            duration = max(duration, track['offset'] + len(samples) / Config.TARGET_SAMPLE_RATE)
            decoded.append((track, samples))
        timings['decoding'] = time.time() - step

        step = time.time()
        segments = []
        for track, samples in decoded:
            speech = self.transcriber.transcribe_speech(
                samples, Config.TARGET_SAMPLE_RATE, language=language
            )
            for seg in speech:
                start = round(seg['start'] + track['offset'], 2)
                end = round(seg['end'] + track['offset'], 2)
                segments.append({
                    'speaker': track['label'],
                    'start': start,
                    'end': end,
                    'duration': round(end - start, 2),
                    'text': seg['text'],
                    'confidence': seg.get('confidence'),
                })
        segments.sort(key=lambda seg: (seg['start'], seg['end']))
        segments = self.transcriber.merge_consecutive_segments(segments)
        timings['transcription'] = time.time() - step
        timings['total'] = time.time() - started

        print(
            f"Transcribed {duration:.0f}s of audio in {timings['total']:.1f}s "
            f"({duration / max(timings['total'], 0.01):.1f}x realtime)",
            flush=True,
        )
        return {
            'transcript': transcript_of(segments),
            'segments': segments,
            'language': language,
            'audio_info': {
                'duration_seconds': round(duration, 2),
                'sample_rate': Config.TARGET_SAMPLE_RATE,
            },
            'statistics': statistics_of(segments),
            'timings': {name: round(value, 2) for name, value in timings.items()},
        }


def transcript_of(segments: List[Dict]) -> str:
    """One line per turn: "LABEL: text", or just the text when unlabelled."""
    lines = []
    current = object()
    for seg in segments:
        if seg['speaker'] == current and lines:
            lines[-1] += f" {seg['text']}"
        else:
            lines.append(f"{seg['speaker']}: {seg['text']}" if seg['speaker'] else seg['text'])
            current = seg['speaker']
    return "\n".join(lines)


def statistics_of(segments: List[Dict]) -> Dict:
    speakers = {}
    for seg in segments:
        if not seg['speaker']:
            continue
        entry = speakers.setdefault(seg['speaker'], {'words': 0, 'speech_seconds': 0.0, 'turns': 0})
        entry['words'] += len(seg['text'].split())
        entry['speech_seconds'] = round(entry['speech_seconds'] + seg['duration'], 2)
        entry['turns'] += 1
    return {
        'total_words': sum(len(seg['text'].split()) for seg in segments),
        'speech_seconds': round(sum(seg['duration'] for seg in segments), 2),
        'speakers': speakers,
    }
