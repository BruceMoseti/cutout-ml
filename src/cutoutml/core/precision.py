"""The precision vocabulary, kept apart from the code that resolves it.

:data:`Precision` is three string literals, but it appears in the signature of nearly
everything: registry specs, adapter constructors, benchmark cases, job rows. Its natural
home is :mod:`cutoutml.core.devices` - the module that decides whether a requested
precision is usable on the hardware in front of it - and that module imports torch.

Naming a precision and running one are different jobs. The registry and the API do the
first and never the second, so the vocabulary lives here, where importing it costs
nothing. :mod:`cutoutml.core.devices` re-exports it, so callers that do need the
resolution logic still see one name.
"""

from __future__ import annotations

from typing import Literal, get_args

Precision = Literal["fp32", "fp16", "bf16"]

PRECISIONS: tuple[Precision, ...] = get_args(Precision)
"""Every valid precision, for validation and for CLI ``choices``."""
