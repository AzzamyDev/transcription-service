"""WSGI entry point: the app is imported without a model; gunicorn.py loads
it in the worker after the fork (see post_worker_init)."""
from src.api.api import app

__all__ = ['app']
