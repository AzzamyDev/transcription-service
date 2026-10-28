"""Stub faster-whisper when it is not installed, so the API and job queue can
be tested without the model libraries."""
import importlib.util
import sys
from unittest.mock import MagicMock

if importlib.util.find_spec('faster_whisper') is None:
    for name in ('faster_whisper', 'faster_whisper.vad'):
        sys.modules[name] = MagicMock()


import pytest


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    """A developer's .env may set API_KEY; tests run without authentication
    unless one asks for it."""
    from src.core.config import Config
    monkeypatch.setattr(Config, 'API_KEY', '')
