"""
Speech transcription with faster-whisper (CTranslate2): the same Whisper
weights, several times faster and lighter on the CPU, with a voice-activity
filter so silence and noise are not transcribed at all.
"""
from typing import List, Dict, Optional
import math
import numpy as np

from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps

from src.core.config import Config
from src.core.exceptions import TranscriptionError, ModelLoadError
from src.services.hallucinations import collapse_loops, is_invented


def confidence_of(avg_logprob: float) -> float:
    """Whisper's mean token log-probability as a 0-1 score."""
    return round(min(1.0, max(0.0, math.exp(avg_logprob))), 3)


def speech_regions(audio: np.ndarray, sample_rate: int) -> List[tuple]:
    """
    Sample ranges where someone speaks, found by Silero VAD. Speech parted
    by a long silence is read as its own region: decoding replies seconds
    apart in one window misplaces the words at its edges.
    """
    speech = get_speech_timestamps(
        audio, VadOptions(min_silence_duration_ms=500, speech_pad_ms=300)
    )
    gap = int(Config.REGION_GAP_S * sample_rate)
    longest = int(Config.REGION_MAX_S * sample_rate)
    regions = []
    for chunk in speech:
        if regions and chunk['start'] - regions[-1][1] <= gap \
                and chunk['end'] - regions[-1][0] <= longest:
            regions[-1][1] = chunk['end']
        else:
            regions.append([chunk['start'], chunk['end']])
    return [tuple(r) for r in regions]


def split_on_pauses(seg) -> List[Dict]:
    """
    One line per stretch of speech. A decoded window can span two replies
    seconds apart; its words' times split it, and the words' own
    probabilities score each part (the window's avg_logprob is shared by
    all its segments, so it cannot tell them apart).
    """
    words = list(seg.words or [])
    base = {'avg_logprob': seg.avg_logprob, 'no_speech_prob': seg.no_speech_prob}
    if not words:
        return [{**base, 'start': seg.start, 'end': seg.end, 'text': seg.text,
                 'confidence': confidence_of(seg.avg_logprob)}]
    groups, current = [], [words[0]]
    for word in words[1:]:
        if word.start - current[-1].end > Config.SPLIT_PAUSE_S:
            groups.append(current)
            current = []
        current.append(word)
    groups.append(current)
    return [
        {
            **base,
            'start': group[0].start,
            'end': group[-1].end,
            'text': ''.join(w.word for w in group),
            'confidence': round(sum(w.probability for w in group) / len(group), 3),
        }
        for group in groups
    ]


class WhisperTranscriber:
    """Speech-to-text with faster-whisper on the CPU"""

    def __init__(self, model_size: Optional[str] = None):
        self.model_size = model_size or Config.WHISPER_MODEL
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(
                f"Loading Whisper model {self.model_size} "
                f"({Config.WHISPER_COMPUTE_TYPE}, {Config.WHISPER_CPU_THREADS} threads)...",
                flush=True,
            )
            self.model = WhisperModel(
                self.model_size,
                device='cpu',
                compute_type=Config.WHISPER_COMPUTE_TYPE,
                cpu_threads=Config.WHISPER_CPU_THREADS,
                download_root=Config.MODEL_DIR,
            )
            print("Whisper model loaded", flush=True)
        except Exception as e:
            raise ModelLoadError(f"Failed to load Whisper model: {str(e)}")

    def _decode(self, samples: np.ndarray, sample_rate: int, language: Optional[str]) -> Dict[str, any]:
        """Decode 16 kHz mono audio; segments carry Whisper's own scores."""
        try:
            audio = samples.astype(np.float32, copy=False)
            peak = max(abs(float(audio.max(initial=0))), abs(float(audio.min(initial=0))))
            if peak > 1.0:
                audio = audio / peak

            out = []
            for first, last in speech_regions(audio, sample_rate):
                segments, _ = self.model.transcribe(
                    audio[first:last],
                    language=language,
                    task='transcribe',
                    beam_size=Config.BEAM_SIZE,
                    best_of=Config.BEST_OF,
                    # Retried hotter only when a decode loops or is unsure.
                    temperature=[0.0, 0.2, 0.4, 0.6],
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=False,
                    repetition_penalty=1.1,
                    hotwords=Config.HOTWORDS or None,
                    word_timestamps=True,
                )
                offset = first / sample_rate
                for seg in segments:
                    for part in split_on_pauses(seg):
                        part['start'] += offset
                        part['end'] += offset
                        out.append(part)
            return {'segments': out}
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {str(e)}")

    def transcribe_speech(
        self,
        samples: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        Transcribe one recording. Whisper tends to invent text over silence,
        so segments that are quiet or that Whisper itself rates as non-speech
        are dropped.
        """
        try:
            result = self._decode(samples, sample_rate, language)
            segments = []
            for seg in result['segments']:
                text = seg['text'].strip()
                if not text:
                    continue
                window = samples[int(seg['start'] * sample_rate):int(seg['end'] * sample_rate)]
                if window.size == 0:
                    continue
                level_db = 20 * np.log10(np.sqrt(np.mean(window ** 2)) + 1e-10)
                if level_db < Config.SILENCE_DB:
                    continue
                if (
                    seg.get('no_speech_prob', 0) > Config.NO_SPEECH_THRESHOLD
                    and seg.get('avg_logprob', 0) < Config.LOGPROB_THRESHOLD
                ):
                    continue
                confidence = seg.get('confidence')
                if confidence is None:
                    confidence = confidence_of(seg.get('avg_logprob', 0))
                # A stock line ("Terima kasih.") said with little certainty, or
                # faster than anyone can say it, is Whisper reading a click or
                # a breath, not the speaker.
                if is_invented(text):
                    rate = len(text) / max(seg['end'] - seg['start'], 0.01)
                    if confidence < Config.INVENTED_MIN_CONFIDENCE or rate > Config.MAX_CHARS_PER_S:
                        continue
                segments.append({
                    'start': float(seg['start']),
                    'end': float(seg['end']),
                    'text': collapse_loops(text),
                    'confidence': confidence,
                })
            return segments

        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {str(e)}")

    @staticmethod
    def merge_consecutive_segments(
        segments: List[Dict],
        gap_threshold: Optional[float] = None
    ) -> List[Dict]:
        """Join consecutive lines of one speaker that are closer than the gap."""
        if not segments:
            return []

        gap_threshold = gap_threshold or Config.MERGE_GAP_THRESHOLD

        merged = []
        current = segments[0].copy()

        for seg in segments[1:]:
            gap = seg['start'] - current['end']

            if seg['speaker'] == current['speaker'] and gap < gap_threshold:
                if current.get('confidence') is not None and seg.get('confidence') is not None:
                    weight = (current['end'] - current['start'], seg['end'] - seg['start'])
                    total = sum(weight) or 1
                    current['confidence'] = round(
                        (current['confidence'] * weight[0] + seg['confidence'] * weight[1]) / total, 3
                    )
                current['end'] = seg['end']
                current['duration'] = current['end'] - current['start']
                current['text'] += ' ' + seg['text']
            else:
                merged.append(current)
                current = seg.copy()

        merged.append(current)
        return merged
