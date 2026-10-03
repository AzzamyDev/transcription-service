"""
Speech transcription using OpenAI Whisper - CLEAN OUTPUT
No prints during processing to avoid conflicts with spinner
"""
from typing import List, Dict, Optional
import whisper
import torch
import numpy as np
import os

from src.core.config import Config
from src.core.exceptions import TranscriptionError, ModelLoadError


class WhisperTranscriber:
    """Speech-to-text transcription using Whisper"""

    def __init__(self, model_size: Optional[str] = None):
        """Initialize transcriber"""
        self.model_size = model_size or Config.WHISPER_MODEL
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load Whisper model with download progress"""
        try:
            # Enable download progress bar
            os.environ['TQDM_DISABLE'] = '0'

            print(f"Loading Whisper model: {self.model_size}...", flush=True)

            self.model = whisper.load_model(
                self.model_size,
                download_root=None,
                in_memory=False
            )

            if torch.cuda.is_available():
                print("Using GPU for transcription", flush=True)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

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
        """Transcribe audio - SILENT during processing"""
        try:
            language = language or Config.LANGUAGE

            # NO PRINT HERE - spinner is active!

            if isinstance(audio_tensor, torch.Tensor):
                if audio_tensor.dim() == 2:
                    audio_tensor = audio_tensor.squeeze(0)
                audio_np = audio_tensor.cpu().numpy()
            else:
                audio_np = audio_tensor

            if audio_np.dtype != np.float32:
                audio_np = audio_np.astype(np.float32)

            if audio_np.max() > 1.0 or audio_np.min() < -1.0:
                audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()))

            result = self.model.transcribe(
                audio_np,
                language=language,
                task=task,
                verbose=False,  # CRITICAL: No verbose output!
                fp16=torch.cuda.is_available(),
                condition_on_previous_text=False,
                temperature=0.0,
                beam_size=1,
                best_of=1,
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
            )

            # NO PRINT HERE - let pipeline handle it
            return result

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
                segments.append({
                    'start': float(seg['start']),
                    'end': float(seg['end']),
                    'text': text,
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

            combined_text = ' '.join(text_parts).strip()

            if combined_text:
                aligned.append({
                    'speaker': dia_seg['speaker'],
                    'start': dia_seg['start'],
                    'end': dia_seg['end'],
                    'duration': dia_seg['duration'],
                    'text': combined_text
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
                current['end'] = seg['end']
                current['duration'] = current['end'] - current['start']
                current['text'] += ' ' + seg['text']
            else:
                merged.append(current)
                current = seg.copy()

        merged.append(current)

        # NO PRINT - pipeline shows final count
        return merged