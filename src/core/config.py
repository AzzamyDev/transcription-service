"""
Configuration module for Audio Transcription System
Tesla V100 32GB Optimized
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration - Optimized for Tesla V100"""

    # Base directory
    BASE_DIR = Path(__file__).parent.parent.parent

    # Audio settings
    TARGET_SAMPLE_RATE = 16000
    NORMALIZE_DB = -20.0
    MAX_AUDIO_SIZE_MB = int(os.getenv('MAX_AUDIO_SIZE_MB', '1000'))
    SUPPORTED_FORMATS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.webm'}

    # Model settings
    WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'large-v3')
    LANGUAGE = os.getenv('LANGUAGE', 'id')

    # API Keys
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN', '')

    # Ollama settings (self-hosted, gratis)
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5:14b')

    # Processing settings
    MERGE_GAP_THRESHOLD = 1.0  # seconds
    OVERLAP_THRESHOLD = 0.3

    # Per-speaker tracks: segments quieter than this, or that Whisper rates
    # as non-speech with low confidence, are silence it transcribed anyway.
    SILENCE_DB = float(os.getenv('SILENCE_DB', '-45'))
    NO_SPEECH_THRESHOLD = 0.6
    LOGPROB_THRESHOLD = -1.0

    # Background jobs
    ASR_CONCURRENCY = int(os.getenv('ASR_CONCURRENCY', '1'))
    ASR_MAX_PENDING = int(os.getenv('ASR_MAX_PENDING', '200'))

    @classmethod
    def validate(cls) -> tuple[bool, list[str]]:
        """
        Validate configuration

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check HuggingFace token
        if not cls.HUGGINGFACE_TOKEN:
            errors.append("HUGGINGFACE_TOKEN is required for speaker diarization")

        # Check Whisper model
        valid_models = {'tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'}
        if cls.WHISPER_MODEL not in valid_models:
            errors.append(f"Invalid WHISPER_MODEL. Must be one of: {valid_models}")

        return len(errors) == 0, errors