"""Tests for the provider-aware cancellation search fallback.

No network: each provider's client is replaced with a stand-in that returns a
canned response in the SDK's shape. What is being tested is the routing (which
backend a `--provider` choice reaches, and when none does) and the parsing —
both of which decide whether a customer gets guidance or an honest miss.
"""

import json
import sys

import pytest
from rich.console import Console

from src.agents.cancellation_research import web_search
from src.config import REQUIRED_ENV_VARS


@pytest.fixture(autouse=True)
def unconfigured(monkeypatch):
    # Every test starts from a process that has not chosen a provider, and with
    # no ambient credentials, so nothing leaks in from the developer's shell.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    web_search.configure(None)
    yield
    web_search.configure(None)


class _Block:
    def __init__(self, **fields):
        self.__dict__.update(fields)


_GUIDANCE = json.dumps(
    {
        "found": True,
        "url": "https://obscure-saas.example/account/billing",
        "steps": ["Sign in.", "Open Billing and cancel."],
        "notice_period": "30 days",
        "gotchas": ["Billed annually in advance."],
    }
)


class _FakeAnthropic:
    """Stands in for anthropic.Anthropic, recording the request it was given."""

    last_request: dict = {}

    def __init__(self, text, sources=()):
        self._response = _Block(
            content=[
                _Block(
                    type="web_search_tool_result",
                    content=[_Block(title=t, url=u) for t, u in sources],
                ),
                _Block(type="text", text=text),
            ]
        )

    def __call__(self, **_kwargs):
        return self

    @property
    def messages(self):
        return self

    def create(self, **request):
        type(self).last_request = request
        return self._response


class _FakeOpenAI:
    """Stands in for openai.OpenAI. Cited pages ride on the output text as
    annotations rather than in a results block, so the fake mirrors that."""

    last_request: dict = {}

    def __init__(self, text, sources=()):
        annotations = [_Block(type="url_citation", title=t, url=u) for t, u in sources]
        self._response = _Block(
            output_text=text,
            output=[_Block(content=[_Block(annotations=annotations)])],
        )

    def __call__(self, **_kwargs):
        return self

    @property
    def responses(self):
        return self

    def create(self, **request):
        type(self).last_request = request
        return self._response


def _use_anthropic(monkeypatch, text, sources=()):
    import anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic(text, sources))
    web_search.configure("anthropic")


def _use_openai(monkeypatch, text, sources=()):
    import openai

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI(text, sources))
    web_search.configure("openai")


class TestProviderRouting:
    def test_an_unconfigured_process_has_no_search(self):
        # A library caller that never went through the CLI must not quietly
        # spend whichever API key happens to be exported.
        assert web_search.search_available() is False
        assert web_search.search_cancellation("Obscure SaaS Ltd") is None

    def test_ollama_has_no_hosted_search(self, monkeypatch):
        # A local model has nothing behind it to search with, even on a machine
        # where other providers' keys are present.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        web_search.configure("ollama")

        assert web_search.search_available() is False
        assert web_search.search_cancellation("Obscure SaaS Ltd") is None

    @pytest.mark.parametrize("provider", ["anthropic", "openai"])
    def test_a_provider_without_its_key_is_unavailable(self, provider):
        web_search.configure(provider)
        assert web_search.search_available() is False
        assert web_search.search_cancellation("Obscure SaaS Ltd") is None

    def test_the_anthropic_provider_reaches_the_anthropic_backend(self, monkeypatch):
        _use_anthropic(monkeypatch, _GUIDANCE)

        assert web_search.search_available() is True
        assert web_search.search_cancellation("Obscure SaaS Ltd")

        request = _FakeAnthropic.last_request
        assert request["tools"][0]["name"] == "web_search"
        assert "Obscure SaaS Ltd" in request["messages"][0]["content"]

    def test_the_openai_provider_reaches_the_openai_backend(self, monkeypatch):
        _use_openai(monkeypatch, _GUIDANCE)

        assert web_search.search_available() is True
        assert web_search.search_cancellation("Obscure SaaS Ltd")

        request = _FakeOpenAI.last_request
        assert request["tools"][0]["type"] == "web_search"
        assert "Obscure SaaS Ltd" in request["input"]

    def test_configuring_a_provider_is_reversible(self, monkeypatch):
        _use_anthropic(monkeypatch, _GUIDANCE)
        web_search.configure("ollama")
        assert web_search.search_available() is False


class TestCliWiring:
    """`--provider` is the only place the backend gets chosen, so the CLI has to
    pass it on — otherwise search is silently dead in the real app."""

    def _run_cli(self, monkeypatch, provider):
        import src.main as main_module

        # The CLI refuses to start a hosted provider without its credential, so
        # supply one. This test is about which search backend `--provider`
        # reaches, not about the key check — that is covered by
        # tests/test_config.py::TestCredentialGuard.
        required_key = REQUIRED_ENV_VARS.get(provider)
        if required_key:
            monkeypatch.setenv(required_key, "test-key")

        monkeypatch.setattr(sys, "argv", ["thinkmoney", "--provider", provider])
        monkeypatch.setattr(main_module, "get_llm", lambda *a, **k: object())
        monkeypatch.setattr(main_module, "build_graph", lambda *a, **k: object())
        # End the input loop immediately: the wiring happens before the first turn.
        monkeypatch.setattr(
            Console, "input", lambda *a, **k: (_ for _ in ()).throw(EOFError)
        )
        main_module.main()

    def test_the_cli_points_search_at_the_chosen_provider(self, monkeypatch):
        self._run_cli(monkeypatch, "anthropic")

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        assert web_search.search_available() is True

    def test_the_cli_leaves_ollama_without_search(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        self._run_cli(monkeypatch, "ollama")

        assert web_search.search_available() is False


class TestAnthropicBackend:
    def test_a_found_result_becomes_guidance(self, monkeypatch):
        _use_anthropic(
            monkeypatch,
            _GUIDANCE,
            sources=[("Cancelling", "https://obscure-saas.example/help/cancel")],
        )
        result = web_search.search_cancellation("Obscure SaaS Ltd")

        assert result["url"] == "https://obscure-saas.example/account/billing"
        assert result["steps"] == ["Sign in.", "Open Billing and cancel."]
        assert result["notice_period"] == "30 days"
        assert result["gotchas"] == ["Billed annually in advance."]
        assert result["results"] == [
            {"title": "Cancelling", "url": "https://obscure-saas.example/help/cancel"}
        ]

    def test_repeated_sources_are_reported_once(self, monkeypatch):
        page = ("Cancelling", "https://obscure-saas.example/help/cancel")
        _use_anthropic(monkeypatch, _GUIDANCE, sources=[page, page])

        assert len(web_search.search_cancellation("Obscure SaaS Ltd")["results"]) == 1

    def test_a_search_error_block_does_not_break_the_sources(self, monkeypatch):
        # On a search failure the result block carries an error object, not a
        # list — citing nothing is fine, failing the turn is not.
        import anthropic

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        fake = _FakeAnthropic(_GUIDANCE)
        fake._response.content[0].content = _Block(type="web_search_tool_result_error")
        monkeypatch.setattr(anthropic, "Anthropic", fake)
        web_search.configure("anthropic")

        assert web_search.search_cancellation("Obscure SaaS Ltd")["results"] == []


class TestOpenAIBackend:
    def test_a_found_result_becomes_guidance(self, monkeypatch):
        _use_openai(
            monkeypatch,
            _GUIDANCE,
            sources=[("Cancelling", "https://obscure-saas.example/help/cancel")],
        )
        result = web_search.search_cancellation("Obscure SaaS Ltd")

        assert result["steps"] == ["Sign in.", "Open Billing and cancel."]
        assert result["results"] == [
            {"title": "Cancelling", "url": "https://obscure-saas.example/help/cancel"}
        ]

    def test_a_reply_with_no_citations_still_yields_guidance(self, monkeypatch):
        _use_openai(monkeypatch, _GUIDANCE)
        assert web_search.search_cancellation("Obscure SaaS Ltd")["results"] == []


class TestReplyHandling:
    """The same parsing guards both backends, so these run on one of them."""

    def test_a_merchant_the_search_could_not_cover_returns_nothing(self, monkeypatch):
        # Better a degraded answer than plausible steps the model made up.
        _use_anthropic(monkeypatch, '{"found": false, "steps": [], "url": null}')
        assert web_search.search_cancellation("Obscure SaaS Ltd") is None

    def test_a_result_with_neither_steps_nor_a_url_is_not_guidance(self, monkeypatch):
        _use_anthropic(monkeypatch, '{"found": true, "steps": [], "url": null}')
        assert web_search.search_cancellation("Obscure SaaS Ltd") is None

    def test_a_url_with_no_steps_is_still_worth_returning(self, monkeypatch):
        _use_anthropic(
            monkeypatch, '{"found": true, "url": "https://x.example", "steps": []}'
        )
        assert web_search.search_cancellation("Obscure SaaS Ltd")["steps"] == []

    def test_blank_steps_are_dropped(self, monkeypatch):
        _use_anthropic(
            monkeypatch,
            '{"found": true, "url": "https://x.example", "steps": ["", "Go."]}',
        )
        assert web_search.search_cancellation("Obscure SaaS Ltd")["steps"] == ["Go."]

    def test_prose_and_fences_around_the_json_are_tolerated(self, monkeypatch):
        _use_anthropic(
            monkeypatch,
            'Here is what I found:\n```json\n{"found": true, "url": '
            '"https://x.example", "steps": ["Cancel."]}\n```',
        )
        assert web_search.search_cancellation("Obscure SaaS Ltd")["steps"] == [
            "Cancel."
        ]

    def test_an_unparseable_reply_returns_nothing_rather_than_raising(
        self, monkeypatch
    ):
        _use_anthropic(monkeypatch, "I could not find anything.")
        assert web_search.search_cancellation("Obscure SaaS Ltd") is None

    def test_a_json_array_is_not_mistaken_for_guidance(self, monkeypatch):
        _use_anthropic(monkeypatch, '["not", "an", "object"]')
        assert web_search.search_cancellation("Obscure SaaS Ltd") is None
