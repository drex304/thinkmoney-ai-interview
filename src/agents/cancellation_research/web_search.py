"""Provider-aware live web search, used only as a cancellation-guidance fallback.

Search follows `--provider`: the reviewer running Anthropic gets Claude's
server-side web search, the reviewer running OpenAI gets the Responses API's
hosted search, and the reviewer running Ollama gets nothing, because a local
model has no hosted search behind it. Nothing here is a scraper — both backends
are the provider's own search tool, so there is no crawler to maintain and no
third API key to hand out.

The unavailable case is a first-class outcome, not an error. `search_available()`
is what lets the caller say "live search is not available here" instead of
claiming it looked and found nothing, and every failure below — missing key,
missing SDK, unparseable reply, merchant not covered — funnels into the same
`None`, which is the degraded answer rather than a broken turn.
"""

import json
import os

# Long enough for a couple of search rounds plus the model reading the pages;
# short enough that a wedged search does not hang the customer's turn.
_TIMEOUT_SECONDS = 45.0

_ANTHROPIC_MODEL = "claude-opus-5"
# The Responses API's hosted search is available on the 4o family; this is also
# the project's OpenAI default, so the search path costs what the agent costs.
_OPENAI_MODEL = "gpt-4o-mini"

_SYSTEM = (
    "You find out how to cancel a subscription with a named merchant, for an "
    "agent handling a UK bank customer. Search the web and prefer the "
    "merchant's own help or account pages over third-party write-ups.\n\n"
    "Answer only from what the search actually returned. If it does not cover "
    "the merchant's cancellation process, say so rather than filling the gap "
    "from memory — a wrong answer here costs the customer money.\n\n"
    "Reply with one JSON object and nothing else:\n"
    '{"found": true|false, "url": string|null, "steps": [string], '
    '"notice_period": string|null, "gotchas": [string]}\n\n'
    "url: the merchant's own cancellation or account page. steps: what the "
    "customer does, in order, in plain English. notice_period: how much notice "
    "the merchant requires, or null. gotchas: minimum terms, exit fees, "
    "refunds, anything that bites after cancelling. If you could not find the "
    "process, return found false and leave the rest empty."
)

_QUESTION = "How does a customer cancel their {merchant} subscription?"

# Which provider the CLI selected. None until `configure()` runs, and an
# unconfigured process deliberately has no search: better that a library caller
# gets the documented degraded answer than that it quietly bills whichever key
# happens to be exported.
_provider: str | None = None


def configure(provider: str | None) -> None:
    """Point search at the provider the CLI was started with."""
    global _provider
    _provider = provider


def search_available() -> bool:
    """Whether a live search can actually run for the configured provider."""
    backend = _BACKENDS.get(_provider or "")
    return backend is not None and backend["available"]()


def search_cancellation(merchant: str) -> dict | None:
    """Cancellation guidance for a merchant, or None if none could be found.

    Returns a dict with `url`, `steps`, `notice_period`, `gotchas` and `results`
    — the shape the cancellation tool renders as unverified guidance.
    """
    backend = _BACKENDS.get(_provider or "")
    if backend is None or not backend["available"]():
        return None
    return backend["search"](merchant)


def _has(module: str, env_var: str) -> bool:
    from importlib.util import find_spec

    return find_spec(module) is not None and bool(os.environ.get(env_var))


def _first_json_object(text: str) -> dict | None:
    """Pull the first JSON object out of a model reply, tolerating prose or fences."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _guidance(text: str, sources: list[dict]) -> dict | None:
    """Turn a backend's reply into guidance, or None if it is not usable."""
    payload = _first_json_object(text)
    if payload is None or not payload.get("found"):
        return None

    steps = [str(step) for step in payload.get("steps") or [] if str(step).strip()]
    url = payload.get("url")
    # Guidance with neither steps nor a page to go to is not guidance. The
    # degraded answer at least tells the customer how to find it themselves.
    if not steps and not url:
        return None

    return {
        "url": url,
        "steps": steps,
        "notice_period": payload.get("notice_period"),
        "gotchas": payload.get("gotchas") or [],
        "results": sources[:5],
    }


def _dedupe(sources: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for source in sources:
        url = source.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append(source)
    return unique


# --- Anthropic -------------------------------------------------------------


def _anthropic_available() -> bool:
    return _has("anthropic", "ANTHROPIC_API_KEY")


def _anthropic_search(merchant: str) -> dict | None:
    import anthropic

    client = anthropic.Anthropic(timeout=_TIMEOUT_SECONDS, max_retries=1)
    response = client.messages.create(
        model=_ANTHROPIC_MODEL,
        max_tokens=8192,
        system=_SYSTEM,
        # The basic search tool, which every current model and the pinned SDK
        # both support. Newer models also offer web_search_20260209, which
        # filters results before they reach the context — a drop-in swap once
        # the anthropic package is upgraded past this project's pin.
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 4}],
        messages=[{"role": "user", "content": _QUESTION.format(merchant=merchant)}],
    )

    text = "".join(
        getattr(block, "text", "")
        for block in response.content
        if getattr(block, "type", None) == "text"
    )
    return _guidance(text, _anthropic_sources(response))


def _anthropic_sources(response) -> list[dict]:
    sources = []
    for block in response.content:
        if getattr(block, "type", None) != "web_search_tool_result":
            continue
        # On a search error `content` is a single error object rather than a
        # list of results. Nothing to cite, and not worth failing the turn over.
        results = block.content
        if not isinstance(results, list):
            continue
        for result in results:
            sources.append(
                {
                    "title": getattr(result, "title", None),
                    "url": getattr(result, "url", None),
                }
            )
    return _dedupe(sources)


# --- OpenAI ----------------------------------------------------------------


def _openai_available() -> bool:
    return _has("openai", "OPENAI_API_KEY")


def _openai_search(merchant: str) -> dict | None:
    from openai import OpenAI

    client = OpenAI(timeout=_TIMEOUT_SECONDS, max_retries=1)
    response = client.responses.create(
        model=_OPENAI_MODEL,
        instructions=_SYSTEM,
        tools=[{"type": "web_search"}],
        input=_QUESTION.format(merchant=merchant),
    )
    return _guidance(response.output_text or "", _openai_sources(response))


def _openai_sources(response) -> list[dict]:
    # Cited pages ride along as annotations on the output text rather than as a
    # separate results block, so they have to be walked out of the message.
    sources = []
    for item in getattr(response, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            for annotation in getattr(part, "annotations", None) or []:
                if getattr(annotation, "type", None) != "url_citation":
                    continue
                sources.append(
                    {
                        "title": getattr(annotation, "title", None),
                        "url": getattr(annotation, "url", None),
                    }
                )
    return _dedupe(sources)


# Ollama is absent on purpose: a local model has no hosted search behind it, and
# a missing entry here is exactly the "search is not available" answer.
_BACKENDS = {
    "anthropic": {"available": _anthropic_available, "search": _anthropic_search},
    "openai": {"available": _openai_available, "search": _openai_search},
}
