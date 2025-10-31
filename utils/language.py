import os
from typing import Optional

try:
    from openai import AsyncOpenAI

    _HAS_OPENAI = True
except Exception:
    # Allow running without the OpenAI client installed; callers should handle fallback.
    _HAS_OPENAI = False

from .llm_retry import call_llm_with_retries

_PORTUGUESE_REQUEST_PHRASES = (
    "brazilian portuguese",
    "brazilian-portuguese",
    "in brazilian portuguese",
    "in portuguese",
    "português brasileiro",
    "português do brasil",
    "responda em portugues",
    "responda em português",
    "responder em portugues",
    "responder em português",
    "resposta em portugues",
    "resposta em português",
    "texto em portugues",
    "texto em português",
)


def detect_portuguese(text: Optional[str]) -> bool:
    """Lightweight heuristic to detect Portuguese in user text.

    Avoids external dependencies. Not perfect, but good enough to route translation.
    """
    if not text:
        return False
    t = text.lower()
    keywords = [
        " que ", " de ", " para ", " com ", " por ", " uma ", " um ", " dos ", " das ",
        " não ", " mais ", " menos ", " sobre ", " entre ", " como ", " onde ", " quando ",
        " políticas ", " política ", " clima ", " emissões ", " energia ", " brasil ", " português ",
        "ações", "ção", "ções", "ís", "ções ", " país", " municípios", " governador",
    ]
    if any(k in t or t.startswith(k.strip()) or t.endswith(k.strip()) for k in keywords):
        return True
    return any(phrase in t for phrase in _PORTUGUESE_REQUEST_PHRASES)


async def should_respond_in_portuguese(text: Optional[str]) -> bool:
    """Return True if the user text warrants a Brazilian Portuguese response.

    Uses a lightweight OpenAI call when credentials are available, with heuristic
    fallbacks to keep behaviour stable when the API cannot be reached.
    """
    if not text:
        return False

    candidate = text.strip()
    if not candidate:
        return False

    api_key = os.getenv("OPENAI_API_KEY")
    if _HAS_OPENAI and api_key:
        try:
            client = AsyncOpenAI(api_key=api_key)
            model_name = os.getenv("OPENAI_LANGUAGE_DETECT_MODEL", "gpt-4o-mini")
            system = (
                "You decide whether an assistant should answer in Brazilian Portuguese. "
                "If the user message is in Portuguese (any dialect) or explicitly requests "
                "a Portuguese answer, respond with 'pt-br'. Otherwise respond with 'en'."
            )
            user = (
                "User message:\n"
                f"{candidate}\n\n"
                "Respond with either 'pt-br' or 'en' only."
            )
            response = await call_llm_with_retries(
                lambda: client.chat.completions.create(
                    model=model_name,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                ),
                is_async=True,
                provider="openai.language-detect",
            )
            content = ""
            if response.choices:
                content = (response.choices[0].message.content or "").strip().lower()
            if content.startswith("```"):
                content = content.strip("`")
                if "pt" in content:
                    content = "pt-br"
                elif "en" in content:
                    content = "en"
            content = content.replace("\n", " ").replace("\r", " ").strip(' "')
            if content.startswith("pt"):
                return True
            if content.startswith("en"):
                return False
        except Exception:
            # Fall back to heuristics silently on any API issues.
            pass

    # Heuristic fallback: look for Portuguese-specific patterns or requests.
    lowered = candidate.lower()
    if any(phrase in lowered for phrase in _PORTUGUESE_REQUEST_PHRASES):
        return True

    return detect_portuguese(candidate)
