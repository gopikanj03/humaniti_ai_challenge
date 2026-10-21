"""Per-stage JSON cache keyed by file hash. Lives under data/ (gitignored)."""
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


def load(file_hash: str, stage: str) -> Any | None:
    if not get_config()["cache"]["enabled"]:
        return None
    p = _path(file_hash, stage)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None  # corrupt cache entry: recompute


def save(file_hash: str, stage: str, data: Any) -> None:
    if not get_config()["cache"]["enabled"]:
        return
    p = _path(file_hash, stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)  # atomic, so a crash never leaves a half-written entry
