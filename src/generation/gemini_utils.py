"""Shared Gemini call helper used by retriever, answer generator, and comparator."""
from __future__ import annotations

import re
import time

import structlog
from google import genai

log = structlog.get_logger()

DEFAULT_MODELS = ["gemini-2.5-flash"]

# Retry delays (seconds) for 503 UNAVAILABLE — temporary server overload
_503_RETRY_DELAYS = [5, 15, 30]


def generate_with_fallback(
    client: genai.Client,
    contents: str,
    config: genai.types.GenerateContentConfig,
    models: list[str] | None = None,
) -> tuple[genai.types.GenerateContentResponse, str]:
    """Try models in order, with retries for transient errors.

    - 429 RESOURCE_EXHAUSTED: advance to the next model immediately.
    - 503 UNAVAILABLE: retry the same model up to 3 times with backoff.
    - Any other error: raise immediately.

    Returns (response, model_name_used).
    """
    if models is None:
        models = DEFAULT_MODELS

    last_exc: Exception | None = None

    for model in models:
        for attempt, retry_delay in enumerate([0] + _503_RETRY_DELAYS):
            if retry_delay:
                log.info("gemini_503_retry", model=model, attempt=attempt, wait=retry_delay)
                time.sleep(retry_delay)
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
                    m = re.search(r"retryDelay.*?(\d+)s", err)
                    delay = int(m.group(1)) + 1 if m else 0
                    if delay and delay <= 60 and model == models[-1]:
                        log.info("rate_limit_waiting", seconds=delay)
                        time.sleep(delay)
                    break  # move to next model
                elif "503" in err or "UNAVAILABLE" in err:
                    log.warning("gemini_unavailable", model=model, attempt=attempt)
                    last_exc = exc
                    if attempt < len(_503_RETRY_DELAYS):
                        continue  # retry with next delay
                    break  # exhausted retries, try next model
                else:
                    raise

    raise last_exc  # type: ignore[misc]
