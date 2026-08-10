"""Password hashing and JWT issuance/verification.

bcrypt is used directly rather than through passlib: passlib 1.7.x reads
``bcrypt.__about__``, which was removed in bcrypt 4.1, and the workaround is more code
than the ~30 lines here.

Two details that are easy to get wrong and are handled explicitly:

* **bcrypt truncates at 72 bytes.** Anything longer is silently ignored by the
  algorithm, which means a 200-character password is only as strong as its first 72
  bytes *and* that bcrypt 5.x now raises instead of truncating. Passwords are
  pre-hashed with SHA-256 and base64-encoded to a fixed 44-byte string, so length is
  bounded, no entropy is discarded, and there is no NUL-byte truncation issue either.
* **Tokens are verified with an explicit algorithm allow-list.** Passing the algorithm
  on decode is what prevents the classic ``alg: none`` and RS256-to-HS256 confusion
  attacks. ``iss`` and ``exp`` are both required and validated.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import uuid
from typing import Any

import bcrypt
import jwt

from cutoutml.core.config import Settings, get_settings

BCRYPT_ROUNDS = 12
"""~250 ms per hash on a modern CPU. High enough to matter, low enough for a login."""


def _prepare(password: str) -> bytes:
    """Normalise a password to a fixed-length, bcrypt-safe byte string.

    SHA-256 then base64: 44 bytes, well under bcrypt's 72-byte limit, contains no NUL
    bytes, and preserves the full entropy of arbitrarily long inputs.
    """
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.b64encode(digest)


def hash_password(password: str) -> str:
    """Hash a password with a fresh per-hash salt."""
    if not password:
        raise ValueError("password must not be empty")
    return bcrypt.hashpw(_prepare(password), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time verification. Returns ``False`` on a malformed stored hash."""
    try:
        return bcrypt.checkpw(_prepare(password), password_hash.encode())
    except (ValueError, TypeError):
        return False


class TokenError(Exception):
    """Token could not be verified. Callers map this to HTTP 401."""


def create_access_token(
    subject: str | uuid.UUID,
    *,
    settings: Settings | None = None,
    extra_claims: dict[str, Any] | None = None,
    expires_in: int | None = None,
) -> str:
    """Issue a short-lived HS256 access token.

    ``jti`` is included so a token can be individually revoked by a future denylist, and
    ``iat``/``nbf`` so clock-skew handling is the library's problem rather than ours.
    """
    cfg = settings or get_settings()
    now = dt.datetime.now(dt.UTC)
    ttl = expires_in if expires_in is not None else cfg.jwt_access_ttl_seconds
    payload: dict[str, Any] = {
        "sub": str(subject),
        "iss": cfg.jwt_issuer,
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "exp": int((now + dt.timedelta(seconds=ttl)).timestamp()),
        "jti": uuid.uuid4().hex,
        "typ": "access",
    }
    if extra_claims:
        # Reserved claims cannot be overridden: allowing a caller to set `sub` or `exp`
        # would turn a convenience parameter into privilege escalation.
        reserved = {"sub", "iss", "iat", "nbf", "exp", "jti", "typ"}
        payload.update({k: v for k, v in extra_claims.items() if k not in reserved})
    return jwt.encode(payload, cfg.jwt_secret, algorithm=cfg.jwt_algorithm)


def decode_access_token(token: str, *, settings: Settings | None = None) -> dict[str, Any]:
    """Verify and decode a token, raising :class:`TokenError` on any problem."""
    cfg = settings or get_settings()
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            cfg.jwt_secret,
            # Allow-list of exactly one algorithm: this is the defence against
            # algorithm-confusion attacks.
            algorithms=[cfg.jwt_algorithm],
            issuer=cfg.jwt_issuer,
            options={"require": ["exp", "sub", "iss"], "verify_exp": True},
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("token has expired") from exc
    except jwt.InvalidIssuerError as exc:
        raise TokenError("token issuer is not recognised") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError(f"invalid token: {exc}") from exc

    if payload.get("typ") != "access":
        raise TokenError("token is not an access token")
    return payload


def token_subject(token: str, *, settings: Settings | None = None) -> uuid.UUID:
    """The user id a token authenticates, as a UUID."""
    payload = decode_access_token(token, settings=settings)
    try:
        return uuid.UUID(str(payload["sub"]))
    except (KeyError, ValueError) as exc:
        raise TokenError("token subject is not a valid user id") from exc
