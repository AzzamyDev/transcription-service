"""
Configuration, read from the environment (a `.env` file is loaded too).
"""
import os
import re
import socket
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    BASE_DIR = Path(__file__).parent.parent.parent

    # Audio
    TARGET_SAMPLE_RATE = 16000
    MAX_AUDIO_SIZE_MB = int(os.getenv('MAX_AUDIO_SIZE_MB', '1000'))

    # Model: any name faster-whisper accepts (tiny ... large-v3, large-v3-turbo).
    WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'large-v3-turbo')
    WHISPER_COMPUTE_TYPE = os.getenv('WHISPER_COMPUTE_TYPE', 'int8')
    # Threads per process; scale out with replicas, not with threads.
    WHISPER_CPU_THREADS = int(os.getenv('WHISPER_CPU_THREADS', '4'))
    # Read from WHISPER_LANGUAGE: plain LANGUAGE is the locale variable GNU
    # systems already set (en_US:en). Empty means Whisper detects it.
    LANGUAGE = os.getenv('WHISPER_LANGUAGE', 'id') or None
    BEAM_SIZE = int(os.getenv('WHISPER_BEAM_SIZE', '5'))
    # Only matters when a decode is retried at a higher temperature.
    BEST_OF = int(os.getenv('WHISPER_BEST_OF', '5'))
    # Kept on a volume so a new container does not download it again.
    MODEL_DIR = os.getenv('MODEL_DIR', str(BASE_DIR / 'models'))
    # Words Whisper should lean towards when unsure, comma separated.
    HOTWORDS = os.getenv('HOTWORDS', '')

    # Prefixes every job id, so a gateway can send a job's polls back to the
    # replica that holds it. In Docker the hostname is the container's short id,
    # which the compose network resolves.
    REPLICA_ID = re.sub(r'[^A-Za-z0-9]', '', os.getenv('REPLICA_ID') or socket.gethostname())[:12] or 'local'

    # Empty disables authentication; otherwise requests send X-Api-Key.
    API_KEY = os.getenv('API_KEY', '')

    # A segment quieter than this, or that Whisper rates as non-speech with
    # low confidence, is silence it transcribed anyway.
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
    # Consecutive lines of one speaker closer than this are joined.
    MERGE_GAP_THRESHOLD = 1.0

    # Background jobs. Queued jobs hold their audio in memory.
    ASR_MAX_PENDING = int(os.getenv('ASR_MAX_PENDING', '30'))
    # A job running longer than this is taken as hung: the process exits.
    ASR_JOB_TIMEOUT_MINUTES = int(os.getenv('ASR_JOB_TIMEOUT_MINUTES', '60'))
    # Each /transcribe caller holds a request thread while it waits.
    SYNC_MAX_WAITERS = 4
    SYNC_MAX_WAIT_SECONDS = int(os.getenv('SYNC_MAX_WAIT_SECONDS', '1800'))
