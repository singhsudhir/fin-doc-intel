"""Shared Gemini call helper used by retriever, answer generator, and comparator."""
from __future__ import annotations

import re
import time

import structlog
from google import genai

log = structlog.get_logger()

DEFAULT_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash"]


def generate_with_fallback(
    client: genai.Client,
    contents: str,
    config: genai.types.GenerateContentConfig,
    models: list[str] | None = None,
) -> tuple[genai.types.GenerateContentResponse, str]:
    """Try models in order; on 429 RESOURCE_EXHAUSTED advance to the next.

    Returns (response, model_name_used).
    """
    if models is None:
        models = DEFAULT_MODELS

    last_exc: Exception | None = None
    for model in models:
        try:
            resp = client.models.generate_content(
                model=model, contents=contents, config=config
            )
            if model != models[0]:
                log.info("gemini_fallback_succeeded", model=model)
            return resp, model
        except Exception as exc:
            err = str(exc)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                log.warning("gemini_rate_limited", model=model)
                last_exc = exc
                # Honour the suggested per-minute retry delay if short enough
                m = re.search(r"retryDelay.*?(\d+)s", err)
                delay = int(m.group(1)) + 1 if m else 0
                if delay and delay <= 60 and model == models[-1]:
                    log.info("rate_limit_waiting", seconds=delay)
                    time.sleep(delay)
                continue
            raise

    raise last_exc  # type: ignore[misc]
