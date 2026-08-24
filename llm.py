"""
One small chat wrapper shared by both agents, so the Researcher and the
Reviewer talk to the LLM the same way and the provider is a config choice
rather than something baked into the agent code.

Both supported providers have a free tier that does not require a credit
card: Groq (console.groq.com) and Google Gemini (aistudio.google.com).
"""
from functools import lru_cache

import config


@lru_cache(maxsize=1)
def _groq_client():
    from groq import Groq

    return Groq(api_key=config.GROQ_API_KEY)


@lru_cache(maxsize=1)
def _gemini_client():
    from google import genai

    return genai.Client(api_key=config.GEMINI_API_KEY)


def complete(system: str, user: str, max_tokens: int = 1000, temperature: float = 0.0) -> str:
    """
    Sends one system+user turn to the configured LLM and returns the text.

    Temperature defaults to 0 — for a grounded Q&A system, reproducibility of
    both the draft and the review verdict matters more than variety.
    """
    if config.LLM_PROVIDER == "groq":
        response = _groq_client().chat.completions.create(
            model=config.LLM_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
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
