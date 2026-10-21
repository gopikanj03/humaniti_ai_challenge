"""Chat-completion client (OpenAI-compatible Groq API) returning Pydantic-validated JSON."""
from __future__ import annotations

import hashlib
import json
import logging
from typing import TypeVar

import requests
from pydantic import BaseModel, ValidationError

from . import cache
from .config import get_config, groq_api_key
from .net import ProviderError, request_with_retry

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMOutputError(RuntimeError):
    """The model returned unusable JSON twice."""


def estimate_tokens(text: str) -> int:
    """Conservative token estimate without a tokenizer: ~4 chars/token for Latin text, but scripts like
    Malayalam or Devanagari tokenize far less efficiently, so count those characters heavily."""
    ascii_chars = sum(c.isascii() for c in text)
    return int(ascii_chars / 3.5 + (len(text) - ascii_chars) / 1.2) + 8


def repair_truncated_json(raw: str) -> dict | None:
    """Recover a reply cut off by the token limit: keep everything up to the last complete value
    inside the innermost open array/object, then close the open brackets."""
    stack, in_str, esc, last_cut = [], False, False, None
    for i, c in enumerate(raw):
        if in_str:
            esc = (c == "\\") and not esc
            if c == '"' and not esc:
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            stack.append("}" if c == "{" else "]")
        elif c in "}]":
            if not stack:
                return None
            stack.pop()
            last_cut = (i + 1, list(stack))  # a value just closed: a safe place to cut
    if last_cut is None:
        return None
    end, open_stack = last_cut
    try:
        return json.loads(raw[:end] + "".join(reversed(open_stack)))
    except json.JSONDecodeError:
        return None


def prompt_budget(max_tokens: int, system: str) -> int:
    """Tokens left for the user message once the system prompt and reply allowance are reserved."""
    return get_config()["llm"]["max_request_tokens"] - max_tokens - estimate_tokens(system) - 50


def _chat(model: str, messages: list[dict], max_tokens: int, reasoning_effort: str | None) -> str:
    cfg = get_config()
    payload = {
        "model": model,
        "messages": messages,
        "temperature": cfg["llm"]["temperature"],
        "max_completion_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    def send() -> requests.Response:
        return requests.post(f"{cfg['groq']['base_url']}/chat/completions", json=payload,
                             headers={"Authorization": f"Bearer {groq_api_key()}"}, timeout=cfg["groq"]["timeout_s"])

    body = request_with_retry(send).json()
    return body["choices"][0]["message"].get("content") or ""


def chat_json(model: str, system: str, user: str, schema: type[T], *, max_tokens: int,
              reasoning_effort: str | None = None, file_hash: str | None = None, cache_tag: str = "llm") -> T:
    """Ask for JSON matching `schema`; on invalid output, retry once with the validation error."""
    key = None
    if file_hash:
        digest = hashlib.sha256(f"{model}\n{reasoning_effort}\n{system}\n{user}".encode()).hexdigest()[:16]
        key = f"{cache_tag}_{digest}"
        cached = cache.load(file_hash, key)
        if cached is not None:
            return schema.model_validate(cached)

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for attempt in range(2):
        try:
            raw = _chat(model, messages, max_tokens, reasoning_effort)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = repair_truncated_json(raw)  # reply cut off at max_tokens: keep the complete part
                if data is None:
                    raise
                log.warning("%s: reply from %s was truncated; kept the complete part", cache_tag, model)
            result = schema.model_validate(data)
            break
        except ProviderError as e:
            # Groq's JSON mode rejects malformed output server-side with this code; treat like invalid JSON.
            if e.code != "json_validate_failed" or attempt == 1:
                raise
            log.warning("%s: provider rejected malformed JSON from %s (attempt %d)", cache_tag, model, attempt + 1)
            messages = messages[:2] + [{"role": "user", "content": "Your previous reply was not valid JSON. "
                                        "Return only valid JSON in the required format."}]
        except (json.JSONDecodeError, ValidationError) as e:
            # Log the error type and size only: the raw output can contain transcript content.
            log.warning("%s: invalid JSON from %s (attempt %d, %d chars)", cache_tag, model, attempt + 1, len(raw))
            if attempt == 1:
                raise LLMOutputError(f"{cache_tag}: model returned invalid JSON twice") from None
            # Don't echo the bad reply back: it would blow the per-request token budget.
            messages = messages[:2] + [{"role": "user", "content": "Your previous reply failed validation: " +
                                        str(e)[:400] + "\nReturn only valid JSON in the required format."}]

    if key:
        cache.save(file_hash, key, result.model_dump())
    return result


__all__ = ["chat_json", "estimate_tokens", "prompt_budget", "LLMOutputError", "ProviderError"]
