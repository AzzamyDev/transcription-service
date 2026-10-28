"""Stub faster-whisper when it is not installed, so the API and job queue can
be tested without the model libraries."""
import importlib.util
import sys
from unittest.mock import MagicMock

if importlib.util.find_spec('faster_whisper') is None:
    for name in ('faster_whisper', 'faster_whisper.vad'):
        sys.modules[name] = MagicMock()
