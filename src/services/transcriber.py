"""
Speech transcription with faster-whisper (CTranslate2): the same Whisper
weights, several times faster and lighter on the GPU, with a voice-activity
filter so silence and noise are not transcribed at all.
"""
from typing import List, Dict, Optional
import math
import torch
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
    """Speech-to-text transcription using Whisper"""

    def __init__(self, model_size: Optional[str] = None):
        self.model_size = model_size or Config.WHISPER_MODEL
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading Whisper model (faster-whisper): {self.model_size}...", flush=True)
            cuda = torch.cuda.is_available()
            self.model = WhisperModel(
                self.model_size,
                device='cuda' if cuda else 'cpu',
                compute_type='float16' if cuda else 'int8',
                download_root=Config.MODEL_DIR,
            )
            print("Whisper model loaded successfully", flush=True)
        except Exception as e:
            raise ModelLoadError(f"Failed to load Whisper model: {str(e)}")

    def transcribe_tensor(
        self,
        audio_tensor: torch.Tensor,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        task: str = 'transcribe'
    ) -> Dict[str, any]:
        """Transcribe 16 kHz audio; segments carry Whisper's own scores."""
        try:
            language = language or Config.LANGUAGE
            if isinstance(audio_tensor, torch.Tensor):
                if audio_tensor.dim() == 2:
                    audio_tensor = audio_tensor.squeeze(0)
                audio_np = audio_tensor.cpu().numpy()
            else:
                audio_np = audio_tensor
            if audio_np.dtype != np.float32:
                audio_np = audio_np.astype(np.float32)
            peak = max(abs(float(audio_np.max(initial=0))), abs(float(audio_np.min(initial=0))))
            if peak > 1.0:
                audio_np = audio_np / peak

            out = []
            for first, last in speech_regions(audio_np, sample_rate):
                segments, _ = self.model.transcribe(
                    audio_np[first:last],
                    language=language,
                    task=task,
                    beam_size=5,
                    best_of=5,
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

    def transcribe_with_speakers(
        self,
        audio_tensor: torch.Tensor,
        sample_rate: int,
        diarization_segments: List[Dict],
        language: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """Transcribe with speakers - SILENT"""
        try:
            # Get transcription (silent)
            result = self.transcribe_tensor(audio_tensor, sample_rate, language)
            whisper_segments = result['segments']

            # Align segments (silent)
            aligned = self._align_segments(whisper_segments, diarization_segments)

            # NO PRINTS - pipeline will show final count
            return aligned

        except Exception as e:
            raise TranscriptionError(f"Transcription with speakers failed: {str(e)}")

    def transcribe_speech(
        self,
        samples: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        Transcribe one speaker's own track. Whisper tends to invent text over
        silence, which a per-speaker track has plenty of, so segments that are
        quiet or that Whisper itself rates as non-speech are dropped.
        """
        try:
            result = self.transcribe_tensor(
                torch.from_numpy(samples), sample_rate, language
            )
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
                # a breath, not the caller.
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
            raise TranscriptionError(f"Track transcription failed: {str(e)}")

    @staticmethod
    def _align_segments(
        whisper_segments: List[Dict],
        diarization_segments: List[Dict]
    ) -> List[Dict[str, any]]:
        """Align segments - SILENT"""
        aligned = []

        for dia_seg in diarization_segments:
            text_parts = []
            scores = []
            dia_start = dia_seg['start']
            dia_end = dia_seg['end']

            for whisper_seg in whisper_segments:
                whisper_start = whisper_seg['start']
                whisper_end = whisper_seg['end']

                if whisper_end < dia_start or whisper_start > dia_end:
                    continue

                overlap_start = max(whisper_start, dia_start)
                overlap_end = min(whisper_end, dia_end)
                overlap = overlap_end - overlap_start

                whisper_duration = whisper_end - whisper_start
                if overlap > whisper_duration * Config.OVERLAP_THRESHOLD:
                    text_parts.append(whisper_seg['text'].strip())
                    scores.append(
                        whisper_seg['confidence'] if whisper_seg.get('confidence') is not None
                        else confidence_of(whisper_seg.get('avg_logprob', 0))
                    )

            combined_text = collapse_loops(' '.join(text_parts).strip())

            if combined_text:
                aligned.append({
                    'speaker': dia_seg['speaker'],
                    'start': dia_seg['start'],
                    'end': dia_seg['end'],
                    'duration': dia_seg['duration'],
                    'text': combined_text,
                    'confidence': round(sum(scores) / len(scores), 3),
                })

        # NO PRINT - silent
        return aligned

    @staticmethod
    def merge_consecutive_segments(
        segments: List[Dict],
        gap_threshold: Optional[float] = None
    ) -> List[Dict]:
        """Merge segments - SILENT"""
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

        # NO PRINT - pipeline shows final count
        return merged