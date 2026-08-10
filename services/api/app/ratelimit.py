"""Token-bucket rate limiting, Redis-backed with an in-process fallback.

A token bucket rather than a fixed window: a fixed window lets a client send its whole
quota in the last millisecond of one window and again in the first millisecond of the
next, producing 2x the intended peak. A bucket smooths that out and still allows a
genuine burst up to ``burst`` tokens.

The Redis implementation runs as a **Lua script**, which makes the read-modify-write
atomic. Doing ``GET``, compute, ``SET`` from Python is a race: two concurrent requests
both read the same token count and both succeed, so a limit of 1 admits 2. Under real
concurrency that is not a rare edge case.

When Redis is unavailable the limiter degrades to a per-process in-memory bucket and logs
it. That is a deliberate availability-over-strictness choice: a Redis blip should not
take the API down, and a per-process limit is still a limit. It is *documented* in
``docs/security.md`` rather than hidden, because with N replicas the effective global
limit becomes N x the configured value during a Redis outage.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Any

from cutoutml.core.logging import get_logger

log = get_logger(__name__)

# KEYS[1] = bucket key, ARGV = capacity, refill_per_second, now, requested
# Returns {allowed, tokens_remaining, retry_after_seconds}
_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

local data = redis.call('HMGET', key, 'tokens', 'timestamp')
local tokens = tonumber(data[1])
local timestamp = tonumber(data[2])

if tokens == nil then
  tokens = capacity
  timestamp = now
end

local elapsed = math.max(0, now - timestamp)
tokens = math.min(capacity, tokens + elapsed * refill)

local allowed = 0
local retry_after = 0
if tokens >= requested then
  tokens = tokens - requested
  allowed = 1
else
  retry_after = (requested - tokens) / refill
end

redis.call('HSET', key, 'tokens', tokens, 'timestamp', now)
-- Expire idle buckets so a large key space does not accumulate; two refill periods
-- is long enough that an expiring bucket is always full anyway.
redis.call('EXPIRE', key, math.ceil(capacity / refill) * 2 + 1)

return {allowed, tostring(tokens), tostring(retry_after)}
"""


@dataclasses.dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """Outcome of one limiter check."""

    allowed: bool
    remaining: float
    retry_after: float
    limit: int
    backend: str

    def headers(self) -> dict[str, str]:
        """Standard rate-limit response headers."""
        out = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, int(self.remaining))),
        }
        if not self.allowed:
            out["Retry-After"] = str(max(1, int(self.retry_after + 0.999)))
        return out


class _InMemoryBuckets:
    """Thread-safe per-process token buckets, used when Redis is unavailable."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: dict[str, tuple[float, float]] = {}

    def consume(
        self, key: str, capacity: float, refill: float, now: float, requested: float
    ) -> tuple[bool, float, float]:
        with self._lock:
            tokens, timestamp = self._buckets.get(key, (capacity, now))
            tokens = min(capacity, tokens + max(0.0, now - timestamp) * refill)
            if tokens >= requested:
                tokens -= requested
                self._buckets[key] = (tokens, now)
                return (True, tokens, 0.0)
            retry_after = (requested - tokens) / refill if refill > 0 else 60.0
            self._buckets[key] = (tokens, now)
            return (False, tokens, retry_after)

    def clear(self) -> None:
        with self._lock:
            self._buckets.clear()


class RateLimiter:
    """Token bucket keyed by identity (user id, or client IP when unauthenticated)."""

    def __init__(
        self,
        *,
        per_minute: int = 120,
        burst: int = 30,
        redis_client: Any | None = None,
        namespace: str = "cutoutml:rl",
    ) -> None:
        self.per_minute = max(1, per_minute)
        self.burst = max(1, burst)
        self.namespace = namespace
        self._redis = redis_client
        self._script: Any | None = None
        self._memory = _InMemoryBuckets()
        self._redis_failed = False

    @property
    def capacity(self) -> float:
        """Bucket size: the configured burst, floored at one minute's worth."""
        return float(max(self.burst, 1))

    @property
    def refill_per_second(self) -> float:
        return self.per_minute / 60.0

    def _ensure_script(self) -> Any | None:
        if self._redis is None or self._redis_failed:
            return None
        if self._script is None:
            try:
                self._script = self._redis.register_script(_TOKEN_BUCKET_LUA)
            except Exception as exc:  # pragma: no cover  # noqa: BLE001 - any redis failure must fall back, not 500
                log.warning("ratelimit_script_registration_failed", error=str(exc))
                self._redis_failed = True
                return None
        return self._script

    def check(self, identity: str, *, cost: float = 1.0, limit: int | None = None) -> RateLimitDecision:
        """Consume ``cost`` tokens for ``identity``.

        ``limit`` overrides the default, so a per-user quota stored on the user row can
        raise or lower the shared default without a second limiter instance.
        """
        effective_limit = max(1, limit or self.per_minute)
        capacity = float(max(self.burst, 1))
        refill = effective_limit / 60.0
        key = f"{self.namespace}:{identity}"
        now = time.time()

        script = self._ensure_script()
        if script is not None:
            try:
                result = script(keys=[key], args=[capacity, refill, now, cost])
                allowed = bool(int(result[0]))
                remaining = float(result[1])
                retry_after = float(result[2])
                return RateLimitDecision(
                    allowed, remaining, retry_after, effective_limit, "redis"
                )
            except Exception as exc:  # noqa: BLE001 - any redis failure must fall back, not 500
                # One failure flips to memory for the process lifetime rather than
                # retrying Redis on every request while it is down.
                log.warning("ratelimit_redis_unavailable", error=str(exc), fallback="in-memory")
                self._redis_failed = True

        allowed, remaining, retry_after = self._memory.consume(key, capacity, refill, now, cost)
        return RateLimitDecision(allowed, remaining, retry_after, effective_limit, "memory")

    def reset(self) -> None:
        """Clear in-memory state (tests)."""
        self._memory.clear()
        self._redis_failed = False
        self._script = None


def build_redis_client(url: str) -> Any | None:
    """Create a Redis client, or ``None`` if the library or server is unavailable."""
    try:
        import redis
    except ImportError:  # pragma: no cover
        return None
    try:
        client = redis.Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
        client.ping()
    except Exception as exc:  # noqa: BLE001 - probing a server that may be absent
        log.warning("redis_unavailable", url=url, error=str(exc))
        return None
    return client
