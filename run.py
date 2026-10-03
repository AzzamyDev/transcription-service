#!/usr/bin/env python3
"""
Run Audio Transcription API Server - DEVELOPMENT MODE
Preloads models and injects into Flask app before starting dev server
"""
import sys

print("=" * 80)
print("SIMTELPAS AI - Starting up (Development Mode)...")
print("=" * 80)

# Load configuration
print("\n[1/3] Loading configuration...")
from src.core.config import Config
print(f"✓ Model: {Config.WHISPER_MODEL}")
print(f"✓ Language: {Config.LANGUAGE}")

# Preload AI models
print("\n[2/3] Initializing AI models (this may take a while)...")
try:
    from src.utils.pipeline import AudioTranscriptionPipeline
    pipeline = AudioTranscriptionPipeline()
    print("✓ All models loaded successfully")
except Exception as e:
    print(f"✗ Failed to load models: {e}")
    print("\nPlease check your configuration and try again.")
    sys.exit(1)

# Inject pipeline into Flask app
print("\n[3/3] Starting API server...")
import src.api.api as api_module
api_module.init(pipeline)
print("✓ Pipeline injected into Flask app")

# Start Flask development server
from src.api.api import main

if __name__ == '__main__':
    main()