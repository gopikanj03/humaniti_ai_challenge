"""Load config/config.yaml and apply environment overrides.

Secrets (API keys) come only from environment variables / platform secrets.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"
VALID_PROVIDERS = {"local", "groq"}


class ConfigError(RuntimeError):
    pass


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    # The project .env wins over stale machine-level variables; on hosted platforms
    # there is no .env, so platform secrets are used as-is.
    load_dotenv(ROOT / ".env", override=True)
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["provider"] = os.getenv("PROVIDER", cfg.get("provider", "local")).strip().lower()
    if cfg["provider"] not in VALID_PROVIDERS:
        raise ConfigError(f"PROVIDER must be one of {sorted(VALID_PROVIDERS)}, got {cfg['provider']!r}")

    cfg["logging"]["level"] = os.getenv("LOG_LEVEL", cfg["logging"]["level"])
    # Model names can be set in .env (they override config.yaml), so they are visible next to the API key.
    for env, section, key in (("TRANSCRIBE_MODEL", cfg["transcription"]["groq"], "model"),
                              ("CONDENSE_MODEL", cfg["llm"], "condense_model"),
                              ("JUDGE_MODEL", cfg["llm"], "judge_model")):
        if os.getenv(env, "").strip():
            section[key] = os.getenv(env).strip()
    for env, key in (("CONDENSE_PARALLEL_MODELS", "condense_parallel_models"),
                     ("CONDENSE_FALLBACK_MODELS", "condense_fallback_models"),
                     ("JUDGE_FALLBACK_MODELS", "judge_fallback_models")):
        if os.getenv(env) is not None:
            cfg["llm"][key] = [m.strip() for m in os.getenv(env).split(",") if m.strip()]
    cfg["ffmpeg_bin"] = os.getenv("FFMPEG_BIN", "ffmpeg")
    cfg["ffprobe_bin"] = os.getenv("FFPROBE_BIN", "ffprobe")
    cfg["cache"]["dir"] = str(ROOT / cfg["cache"]["dir"])
    return cfg


def groq_api_key() -> str:
    """Return the Groq key, refusing unless hosted processing is explicitly approved."""
    get_config()  # ensures .env is loaded
    if not _truthy(os.getenv("HOSTED_PROVIDER_APPROVED")):
        raise ConfigError(
            "PROVIDER=groq but HOSTED_PROVIDER_APPROVED is not true. Recordings are confidential; "
            "set it only after the organiser approves hosted processing, or use PROVIDER=local."
        )
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise ConfigError("PROVIDER=groq but GROQ_API_KEY is not set.")
    return key


def setup_logging() -> None:
    level = get_config()["logging"]["level"]
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
