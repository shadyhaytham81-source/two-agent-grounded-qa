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

# Providers put the wait in the error message, in either of two shapes:
# "try again in 9.7425s" or "try again in 14m43.872s".
_RETRY_AFTER = re.compile(
    r"try again in (?:([0-9]+)m)?([0-9.]+)\s*s", re.IGNORECASE
)

# Some open models write a visible reasoning block before the answer. Strip it
# centrally so the agents never have to know which model they are talking to.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
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
        minutes = int(match.group(1) or 0)
        return minutes * 60 + float(match.group(2)) + 1.0  # cushion past the window
    return min(2 ** attempt, 60)  # exponential fallback when nothing is stated


def complete(
    system: str,
    user: str,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    model: str | None = None,
) -> str:
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
            return _complete_once(system, user, max_tokens, temperature, model or config.LLM_MODEL)
        except Exception as exc:  # noqa: BLE001 — re-raised below unless rate limited
            if not (_is_rate_limit(exc) or _is_transient(exc)):
                raise
            last_exc = exc
            delay = _wait_seconds(exc, attempt) if _is_rate_limit(exc) else min(2**attempt, 30)
            reason = "rate limited" if _is_rate_limit(exc) else "connection dropped"
            print(f"    [{reason} — waiting {delay:.1f}s, attempt {attempt + 1}]")
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _strip_reasoning(text: str | None) -> str:
    """
    Backstop for models that still inline a reasoning block. Handles the
    unclosed case too — a response cut off mid-block has no closing tag, and
    leaving the opener in would corrupt anything downstream that parses it.
    """
    cleaned = _THINK_BLOCK.sub("", text or "")
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    return cleaned.strip()


def _is_transient(exc: Exception) -> bool:
    """Connection drops and 5xx responses are worth another attempt."""
    if getattr(exc, "status_code", 0) >= 500:
        return True
    text = str(exc).lower()
    return any(
        sign in text
        for sign in ("server disconnected", "connection", "timeout", "temporarily")
    )


def _is_rate_limit(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) == 429:
        return True
    text = str(exc).lower()
    return "rate limit" in text or "429" in text or "resource_exhausted" in text


def _complete_once(system: str, user: str, max_tokens: int, temperature: float, model: str) -> str:
    if config.LLM_PROVIDER == "groq":
        extra = {}
        # gpt-oss models reason before answering, and the reasoning tokens are
        # billed against max_tokens. Keep the effort low so the budget goes to
        # the answer — without this a small max_tokens returns empty content.
        if "gpt-oss" in model:
            extra["reasoning_effort"] = config.GROQ_REASONING_EFFORT
        # Other reasoning models write the reasoning into `content` inside a
        # <think> block. Asking for it to be hidden is far more robust than
        # stripping it after the fact: if the model is cut off mid-block it
        # never emits a closing tag, and the whole response is unparseable.
        elif "qwen" in model:
            extra["reasoning_format"] = "hidden"

        response = _groq_client().chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **extra,
        )
        return _strip_reasoning(response.choices[0].message.content)

    from google.genai import types

    response = _gemini_client().models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    return _strip_reasoning(response.text)
