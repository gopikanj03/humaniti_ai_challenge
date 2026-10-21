"""HTTP helpers for the hosted provider: global throttle + retry on 429/5xx."""
from __future__ import annotations

import logging
import random
import threading
import time
from collections import deque
from typing import Callable

import requests

from .config import get_config

log = logging.getLogger(__name__)
RETRY_STATUS = {429, 500, 502, 503, 504}


class Throttle:
    """Sliding-window limiter shared across threads (audio and frame paths run concurrently)."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] >= 60:
                    self._times.popleft()
                if len(self._times) < self.per_minute:
                    self._times.append(now)
                    return
                sleep_for = 60 - (now - self._times[0]) + 0.05
            time.sleep(sleep_for)


_throttle: Throttle | None = None


def throttle() -> Throttle:
    global _throttle
    if _throttle is None:
        _throttle = Throttle(get_config()["groq"]["requests_per_minute"])
    return _throttle


class ProviderError(RuntimeError):
    pass


def _retry_after(resp: requests.Response) -> float | None:
    value = resp.headers.get("retry-after")
    try:
        return float(value) if value else None
    except ValueError:
        return None


def request_with_retry(send: Callable[[], requests.Response]) -> requests.Response:
    """Call `send` (which performs one HTTP request) with throttling and backoff.

    `send` is a callable rather than raw args so multipart file bodies can be
    re-opened on every attempt.
    """
    max_retries = get_config()["groq"]["max_retries"]
    for attempt in range(max_retries + 1):
        throttle().wait()
        try:
            resp = send()
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == max_retries:
                raise ProviderError(f"network error after {attempt + 1} attempts: {type(e).__name__}") from e
            delay = min(60, 2 ** attempt) + random.random()
        else:
            if resp.status_code < 400:
                return resp
            if resp.status_code in (401, 403):
                raise ProviderError(f"provider rejected the API key (HTTP {resp.status_code}); check GROQ_API_KEY")
            if resp.status_code not in RETRY_STATUS or attempt == max_retries:
                # Error bodies can echo request content; log only the status.
                raise ProviderError(f"provider returned HTTP {resp.status_code}")
            delay = _retry_after(resp) or min(60, 2 ** attempt) + random.random()
        log.warning("provider call failed (attempt %d/%d); retrying in %.1fs", attempt + 1, max_retries + 1, delay)
        time.sleep(delay)
    raise ProviderError("unreachable")
