"""Stub the GPU/audio libraries the package imports eagerly, so the API and
job queue can be tested on a machine without them."""
import importlib.util
import sys
from unittest.mock import MagicMock

for root, names in {
    'torch': ['torch'],
    'torchaudio': ['torchaudio'],
    'whisper': ['whisper'],
    'pydub': ['pydub'],
    'soundfile': ['soundfile'],
    'scipy': ['scipy', 'scipy.signal'],
    'requests': ['requests'],
    'psutil': ['psutil'],
    'pyannote': ['pyannote', 'pyannote.audio', 'pyannote.core'],
}.items():
    if root not in sys.modules and importlib.util.find_spec(root) is None:
        for name in names:
            sys.modules[name] = MagicMock()
