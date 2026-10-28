"""Exceptions raised by the transcription service"""


class TranscriptionServiceError(Exception):
    """Base for errors raised by this service"""


class AudioFileError(TranscriptionServiceError):
    """The audio could not be read"""


class ModelLoadError(TranscriptionServiceError):
    """The speech model could not be loaded"""


class TranscriptionError(TranscriptionServiceError):
    """Transcription failed"""
