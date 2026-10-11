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
    # Kept on a volume so a new container does not download it again.
    MODEL_DIR = os.getenv('MODEL_DIR', '/app/models')
    # Words of prison calls Whisper should lean towards when unsure.
    HOTWORDS = os.getenv(
        'HOTWORDS',
        'Warga Binaan, WBP, lapas, rutan, kunjungan, remisi, pembebasan bersyarat, '
        'asimilasi, sidang, pengacara, koperasi, saldo, voucher, wartel, '
        'Assalamualaikum, Waalaikumsalam, alhamdulillah, insyaallah',
    )

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
    # Below this, a stock line such as "Terima kasih." is taken as invented.
    INVENTED_MIN_CONFIDENCE = 0.6
    # Faster than speech: "terima kasih" in 0.3 s is a noise, not words.
    MAX_CHARS_PER_S = 22
    # A pause longer than this between words starts a new transcript line.
    SPLIT_PAUSE_S = 1.0
    # Speech parted by more than this is decoded separately; a region is
    # kept within Whisper's 30-second window.
    REGION_GAP_S = 2.0
    REGION_MAX_S = 28.0

    # Background jobs. Queued jobs hold their audio in memory.
    ASR_MAX_PENDING = int(os.getenv('ASR_MAX_PENDING', '30'))
    # A job running longer than this is taken as hung: the service restarts.
    ASR_JOB_TIMEOUT_MINUTES = int(os.getenv('ASR_JOB_TIMEOUT_MINUTES', '60'))
    # Legacy /transcribe callers each hold a request thread while they wait.
    SYNC_MAX_WAITERS = 4
    SYNC_MAX_WAIT_SECONDS = int(os.getenv('SYNC_MAX_WAIT_SECONDS', '1800'))

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