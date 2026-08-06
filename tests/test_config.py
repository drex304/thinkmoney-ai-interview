"""Tests for LLM provider configuration."""

import os
import sys

import pytest

from src import main
from src.config import DEFAULT_MODELS, REQUIRED_ENV_VARS, get_llm

# OpenAI requires OPENAI_API_KEY at instantiation time.
# We use a dummy key for tests that need to create an OpenAI instance.
_DUMMY_OPENAI_KEY = "sk-test-dummy-key-for-testing-only"


@pytest.fixture(autouse=True)
def _ensure_openai_key(monkeypatch):
    """Set a dummy OpenAI key so ChatOpenAI can be instantiated in tests."""
    if not os.environ.get("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", _DUMMY_OPENAI_KEY)


class TestGetLlm:
    def test_openai_returns_correct_type(self):
        llm = get_llm("openai")
        from langchain_openai import ChatOpenAI

        assert isinstance(llm, ChatOpenAI)

    def test_anthropic_returns_correct_type(self):
        llm = get_llm("anthropic")
        from langchain_anthropic import ChatAnthropic

        assert isinstance(llm, ChatAnthropic)

    def test_ollama_returns_correct_type(self):
        llm = get_llm("ollama")
        from langchain_ollama import ChatOllama

        assert isinstance(llm, ChatOllama)

    def test_openai_default_model(self):
        llm = get_llm("openai")
        assert llm.model_name == "gpt-4o-mini"

    def test_anthropic_default_model(self):
        llm = get_llm("anthropic")
        assert llm.model == "claude-opus-5"

    def test_ollama_default_model(self):
        llm = get_llm("ollama")
        assert llm.model == "gpt-oss:20b"

    def test_openai_custom_model(self):
        llm = get_llm("openai", model="gpt-4o")
        assert llm.model_name == "gpt-4o"

    def test_anthropic_custom_model(self):
        llm = get_llm("anthropic", model="claude-sonnet-4-5-20241022")
        assert llm.model == "claude-sonnet-4-5-20241022"

    def test_ollama_custom_model(self):
        llm = get_llm("ollama", model="llama3.1")
        assert llm.model == "llama3.1"

    def test_invalid_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_llm("azure")

    def test_invalid_provider_message_lists_supported(self):
        with pytest.raises(ValueError, match="ollama, openai, anthropic"):
            get_llm("cohere")

    def test_temperature_is_zero(self):
        """Providers that still accept temperature use 0 for deterministic
        output. Anthropic is excluded: Opus 5 removed the sampling parameters
        and uses effort instead — see tests/test_anthropic_provider.py."""
        for provider in ("openai", "ollama"):
            llm = get_llm(provider)
            assert llm.temperature == 0

    def test_openai_fails_without_api_key(self, monkeypatch):
        """ChatOpenAI requires OPENAI_API_KEY at instantiation time.

        Matched on the message rather than caught as a bare Exception: any bug
        that made get_llm raise for some unrelated reason would satisfy a blind
        `raises(Exception)` and the test would keep passing while testing
        nothing.
        """
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(Exception, match="api_key"):
            get_llm("openai")

    def test_anthropic_creates_without_api_key(self, monkeypatch):
        """ChatAnthropic can instantiate without ANTHROPIC_API_KEY set."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        llm = get_llm("anthropic")
        assert llm is not None

    def test_ollama_creates_without_any_env(self):
        """ChatOllama needs no API key — it connects to local server."""
        llm = get_llm("ollama")
        assert llm is not None


class TestDefaultModels:
    """The CLI builds its --help text and its banner from this table.

    Pinned against what get_llm actually constructs because the two drifted
    once: --help advertised a Haiku build long after the Anthropic branch had
    moved to Opus, so the CLI documented a model it never ran.
    """

    def test_table_covers_every_supported_provider(self):
        assert set(DEFAULT_MODELS) == {"ollama", "openai", "anthropic"}

    @pytest.mark.parametrize(
        "provider, attribute",
        [("openai", "model_name"), ("anthropic", "model"), ("ollama", "model")],
    )
    def test_table_matches_what_get_llm_builds(self, provider, attribute):
        assert getattr(get_llm(provider), attribute) == DEFAULT_MODELS[provider]


class TestCredentialGuard:
    """A hosted provider with no key reaches the customer before it fails.

    The constructors take no credential (see test_anthropic_creates_without_
    api_key above), so nothing objects until the first request — which happens
    inside the graph, after the banner has printed and the customer has typed a
    prompt. The CLI checks up front instead, and says which variable to set.
    """

    @staticmethod
    def _unreachable(*args, **kwargs):
        raise AssertionError("the CLI got past the credential check")

    @pytest.mark.parametrize("provider, key", sorted(REQUIRED_ENV_VARS.items()))
    def test_missing_key_exits_before_the_graph_is_built(
        self, monkeypatch, capsys, provider, key
    ):
        monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(sys, "argv", ["thinkmoney", "--provider", provider])
        monkeypatch.setattr(main, "build_graph", self._unreachable)

        with pytest.raises(SystemExit) as exit_info:
            main.main()

        assert exit_info.value.code == 1
        # Rich wraps at the terminal width, so compare on collapsed whitespace.
        printed = " ".join(capsys.readouterr().out.split())
        assert key in printed

    def test_ollama_is_not_gated(self):
        """The demo runs keyless. Gating it would break the documented path."""
        assert "ollama" not in REQUIRED_ENV_VARS
