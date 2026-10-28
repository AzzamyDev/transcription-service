#!/usr/bin/env python3
"""Development server: loads the model, then serves with Flask."""
import sys

from src.core.config import Config

print(f"Loading model {Config.WHISPER_MODEL}...", flush=True)
try:
    from src.utils.pipeline import TranscriptionPipeline
    pipeline = TranscriptionPipeline()
except Exception as e:
    print(f"Failed to load the model: {e}", flush=True)
    sys.exit(1)

import src.api.api as api_module
api_module.init(pipeline)

if __name__ == '__main__':
    api_module.main()
