"""
One small chat wrapper shared by both agents, so the Researcher and the
Reviewer talk to the LLM the same way and the provider is a config choice
rather than something baked into the agent code.

Both supported providers have a free tier that does not require a credit
card: Groq (console.groq.com) and Google Gemini (aistudio.google.com).

Free tiers are rate limited per minute, and a 100-question test run will hit
that ceiling. Rather than letting a case fail, `complete()` waits out a rate
limit and retries — the provider tells us how long to wait, so we honour it.
"""
import re
import time
from functools import lru_cache

import config

# "Please try again in 9.7425s" — providers put the wait in the error message.
_RETRY_AFTER = re.compile(r"try again in ([0-9.]+)\s*s", re.IGNORECASE)
MAX_RATE_LIMIT_RETRIES = 5


@lru_cache(maxsize=1)
def _groq_client():
    from groq import Groq

    return Groq(api_key=config.GROQ_API_KEY)


@lru_cache(maxsize=1)
def _gemini_client():
    from google import genai

    return genai.Client(api_key=config.GEMINI_API_KEY)


def _wait_seconds(exc: Exception, attempt: int) -> float:
    match = _RETRY_AFTER.search(str(exc))
    if match:
        return float(match.group(1)) + 1.0  # small cushion past the stated window
    return min(2 ** attempt, 60)  # exponential fallback when nothing is stated


def complete(system: str, user: str, max_tokens: int = 1000, temperature: float = 0.0) -> str:
    """
    Sends one system+user turn to the configured LLM and returns the text.

    Temperature defaults to 0 — for a grounded Q&A system, reproducibility of
    both the draft and the review verdict matters more than variety.

    Retries on rate limits, honouring the wait the provider asks for. Any
    other error propagates: a bad key or a bad model should fail loudly, not
    be retried five times first.
    """
    last_exc: Exception | None = None
    for attempt in range(MAX_RATE_LIMIT_RETRIES):
        try:
            return _complete_once(system, user, max_tokens, temperature)
        except Exception as exc:  # noqa: BLE001 — re-raised below unless rate limited
            if not _is_rate_limit(exc):
                raise
            last_exc = exc
            delay = _wait_seconds(exc, attempt)
            print(f"    [rate limited — waiting {delay:.1f}s, attempt {attempt + 1}]")
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _is_rate_limit(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) == 429:
        return True
    text = str(exc).lower()
    return "rate limit" in text or "429" in text or "resource_exhausted" in text


def _complete_once(system: str, user: str, max_tokens: int, temperature: float) -> str:
    if config.LLM_PROVIDER == "groq":
        extra = {}
        # gpt-oss models reason before answering, and the reasoning tokens are
        # billed against max_tokens. Keep the effort low so the budget goes to
        # the answer — without this a small max_tokens returns empty content.
        if "gpt-oss" in config.LLM_MODEL:
            extra["reasoning_effort"] = config.GROQ_REASONING_EFFORT

        response = _groq_client().chat.completions.create(
            model=config.LLM_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **extra,
        )
        return (response.choices[0].message.content or "").strip()

    from google.genai import types

    response = _gemini_client().models.generate_content(
        model=config.LLM_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    return (response.text or "").strip()
