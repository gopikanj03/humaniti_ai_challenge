"""Per-stage JSON cache keyed by file hash. Lives under data/ (gitignored).

Policy (agreed with the project owner): the app never WRITES to the cache, so a newly uploaded
recording or link leaves no transcript or results on disk. It may READ entries that already exist
(e.g. from development runs). Offline dataset scripts call `enable_writes()` so long batch runs on
the free tier can resume after rate limits.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import get_config


def file_sha256(path: str | Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def _path(file_hash: str, stage: str) -> Path:
    # Stage names may include model ids like "whisper-large-v3-turbo"; keep them filename-safe.
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in stage)
    return Path(get_config()["cache"]["dir"]) / file_hash[:16] / f"{safe}.json"


def enable_writes() -> None:
    """For offline dataset scripts only: allow results to be saved so batch runs can resume."""
    get_config()["cache"]["write"] = True


def load(file_hash: str, stage: str) -> Any | None:
    if not get_config()["cache"]["read"]:
        return None
    p = _path(file_hash, stage)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None  # corrupt cache entry: recompute


def save(file_hash: str, stage: str, data: Any) -> None:
    if not get_config()["cache"]["write"]:
        return
    p = _path(file_hash, stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)  # atomic, so a crash never leaves a half-written entry
