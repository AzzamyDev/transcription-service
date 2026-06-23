"""
FINAL CLEAN VERSION - Perfect Output
No spinner conflicts, just clean step-by-step progress
"""
import time
from typing import Dict, Optional, Union, BinaryIO

from src.core.config import Config
from src.core.exceptions import *
from src.utils.audio_processor import AudioProcessor
from src.services.diarizer import SpeakerDiarizer
from src.services.transcriber import WhisperTranscriber
from src.services.summarizer import AISummarizer
from src.utils.formatter import TranscriptFormatter
from src.utils.analyzer import analyze_sentiment, extract_flagged_phrases


class AudioTranscriptionPipeline:
    """Audio transcription with clean output"""

    def __init__(
        self,
        whisper_model: Optional[str] = None,
        language: Optional[str] = None,
        enable_summary: bool = True
    ):
        is_valid, errors = Config.validate()
        if not is_valid:
            raise ConfigurationError(
                f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        print("\n" + "=" * 80, flush=True)
        print("Initializing Audio Transcription Pipeline", flush=True)
        print("=" * 80, flush=True)

        self.audio_processor = AudioProcessor()
        self.diarizer = SpeakerDiarizer()
        self.transcriber = WhisperTranscriber(whisper_model)
        self.summarizer = AISummarizer() if enable_summary else None
        self.language = language or Config.LANGUAGE

        print("=" * 80, flush=True)
        print("✓ Pipeline ready\n", flush=True)

    def process(
        self,
        audio_input: Union[bytes, BinaryIO],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        include_summary: bool = True
    ) -> Dict[str, any]:

        timings = {}
        total_start = time.time()

        try:
            print("\n" + "=" * 80, flush=True)
            print("STARTING AUDIO TRANSCRIPTION", flush=True)
            print("=" * 80, flush=True)

            # Step 1
            print("\n[1/5] Validating audio", flush=True)
            step_start = time.time()
            audio_info = self.audio_processor.validate_audio(audio_input)
            timings['validation'] = time.time() - step_start
            print(f"✓ [1/5] Validated - {timings['validation']:.2f}s", flush=True)
            print(f"    Duration: {audio_info['duration_minutes']:.2f} minutes", flush=True)

            # Step 2
            print("\n[2/5] Preparing audio", flush=True)
            step_start = time.time()
            audio_tensor, sample_rate = self.audio_processor.prepare_for_processing(audio_input)
            wav_bytes = self.audio_processor.prepare_wav_bytes(audio_input)
            timings['preparation'] = time.time() - step_start
            print(f"✓ [2/5] Prepared - {timings['preparation']:.2f}s", flush=True)
            print(f"    Tensor: {audio_tensor.shape}", flush=True)

            # Step 3
            print("\n[3/5] Speaker diarization", flush=True)
            step_start = time.time()
            diarization_segments = self.diarizer.diarize(
                wav_bytes,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            timings['diarization'] = time.time() - step_start
            num_spk = len(set(s['speaker'] for s in diarization_segments))
            print(f"✓ [3/5] Diarization complete - {timings['diarization']:.2f}s", flush=True)
            print(f"    Found {num_spk} speakers in {len(diarization_segments)} segments", flush=True)

            # Step 4
            print("\n[4/5] Transcribing audio", flush=True)
            step_start = time.time()
            transcribed_segments = self.transcriber.transcribe_with_speakers(
                audio_tensor,
                sample_rate,
                diarization_segments,
                language=self.language
            )
            merged_segments = self.transcriber.merge_consecutive_segments(transcribed_segments)
            timings['transcription'] = time.time() - step_start
            print(f"✓ [4/5] Transcription complete - {timings['transcription']:.2f}s", flush=True)
            print(f"    Generated {len(merged_segments)} final segments", flush=True)

            # Step 5
            summary = None
            transcript_text = TranscriptFormatter.format_segments_to_text(
                merged_segments,
                include_timestamps=False
            )
            if include_summary and self.summarizer and self.summarizer.is_available():
                print("\n[5/5] Generating AI summary", flush=True)
                step_start = time.time()
                statistics = TranscriptFormatter.calculate_statistics(merged_segments)
                summary = self.summarizer.create_summary(
                    transcript_text,
                    statistics,
                    language=self.language
                )
                timings['summarization'] = time.time() - step_start
                print(f"✓ [5/5] Summary generated - {timings['summarization']:.2f}s", flush=True)
            else:
                print("\n[5/5] Skipping AI summary", flush=True)
                timings['summarization'] = 0

            # Sentiment & flagged phrases analysis
            sentiment = analyze_sentiment(summary or '')
            flagged_phrases = extract_flagged_phrases(merged_segments)

            statistics = TranscriptFormatter.calculate_statistics(merged_segments)
            timings['total'] = time.time() - total_start

            self._display_timing_summary(timings)

            print("\n" + "=" * 80, flush=True)
            print("✓ PROCESSING COMPLETE", flush=True)
            print("=" * 80, flush=True)

            return {
                'success': True,
                'audio_info': audio_info,
                'segments': merged_segments,
                'statistics': statistics,
                'summary': summary,
                'sentiment': sentiment,
                'flagged_phrases': flagged_phrases,
                'timings': timings
            }

        except Exception as e:
            print(f"\n✗ ERROR: {str(e)}", flush=True)
            raise

    def _display_timing_summary(self, timings: Dict[str, float]):
        print("\n" + "=" * 80, flush=True)
        print("TIMING BREAKDOWN", flush=True)
        print("=" * 80, flush=True)

        steps = [
            ('validation', '1. Audio Validation'),
            ('preparation', '2. Audio Preparation'),
            ('diarization', '3. Speaker Diarization'),
            ('transcription', '4. Transcription'),
            ('summarization', '5. AI Summary')
        ]

        total = timings.get('total', 0)

        for key, label in steps:
            duration = timings.get(key, 0)
            percentage = (duration / total * 100) if total > 0 else 0
            bar_length = int(percentage / 2)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"{label:.<30} {duration:>6.2f}s [{bar}] {percentage:>5.1f}%", flush=True)

        print("-" * 80, flush=True)
        print(f"{'TOTAL TIME':.<30} {total:>6.2f}s", flush=True)
        print("=" * 80, flush=True)