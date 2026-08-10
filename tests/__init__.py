"""Marks ``tests`` as a package.

Two reasons, both mechanical rather than stylistic:

* mypy resolves ``tests/test_x.py`` as the bare module ``test_x`` without this, so
  the ``tests.*`` per-module override in ``pyproject.toml`` never matches and the
  suite is type-checked as strictly as the library.
* pytest derives module names from the path, so two test files with the same
  basename in different directories collide at import time.
"""
