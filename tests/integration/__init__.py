"""Tests that need a live Postgres, Redis or ffmpeg.

All of them skip rather than fail when the service is absent - see the note in
``tests/conftest.py`` - so ``pytest`` is useful on a laptop with nothing running.
"""
