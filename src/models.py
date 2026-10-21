"""Fetch local model files on first use, verifying their sha256."""
from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path

import requests

from .config import ROOT, get_config

log = logging.getLogger(__name__)
_lock = threading.Lock()


def ensure_model(name: str) -> Path:
    spec = get_config()["models"][name]
    path = ROOT / spec["path"]
    with _lock:  # the audio and frame paths may ask concurrently
        if path.exists() and _sha256(path) == spec["sha256"]:
            return path
        log.info("downloading model %s", name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        with requests.get(spec["url"], stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for block in r.iter_content(1 << 20):
                    f.write(block)
        if _sha256(tmp) != spec["sha256"]:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"checksum mismatch for model {name}")
        tmp.replace(path)
        return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()
