"""Tests for the Anthropic branch of the provider factory.

Opus 5 removed the sampling parameters: temperature, top_p and top_k are no
longer accepted and sending any value returns a 400. These tests pin the
absence of those keys in the outgoing payload rather than just the absence of
the constructor argument, because LangChain will happily serialise a field that
was set to its default.
"""

import os

import pytest
from langchain_core.messages import HumanMessage

from src.config import get_llm

# ChatOpenAI demands a key at instantiation time; the value is never used.
_DUMMY_OPENAI_KEY = "sk-test-dummy-key-for-testing-only"


@pytest.fixture(autouse=True)
def _ensure_openai_key(monkeypatch):
    if not os.environ.get("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", _DUMMY_OPENAI_KEY)


def _payload(llm):
    """Build the request body LangChain would send for a trivial exchange."""
    return llm._get_request_payload([HumanMessage(content="hello")])


class TestAnthropicModel:
    def test_default_model_is_opus_5(self):
        llm = get_llm("anthropic")
        assert llm.model == "claude-opus-5"

    def test_custom_model_still_overrides(self):
        llm = get_llm("anthropic", model="claude-sonnet-5")
        assert llm.model == "claude-sonnet-5"

    def test_default_model_reaches_the_payload(self):
        assert _payload(get_llm("anthropic"))["model"] == "claude-opus-5"


class TestAnthropicSamplingParams:
    def test_temperature_is_unset(self):
        assert get_llm("anthropic").temperature is None

    def test_temperature_absent_from_payload(self):
        """Any temperature value — including 0.0 — is a 400 on Opus 5."""
        assert "temperature" not in _payload(get_llm("anthropic"))

    def test_top_p_and_top_k_absent_from_payload(self):
        payload = _payload(get_llm("anthropic"))
        assert "top_p" not in payload
        assert "top_k" not in payload

    def test_a_temperature_bearing_client_would_emit_it(self):
        """Guards the test above: prove the payload check can actually fail."""
        from langchain_anthropic import ChatAnthropic

        payload = _payload(ChatAnthropic(model="claude-opus-5", temperature=0))
        assert payload["temperature"] == 0.0


class TestAnthropicBudget:
    def test_max_tokens_is_8192(self):
        """Thinking is on by default on Opus 5 and shares the response budget;
        LangChain's own default of 4096 is not enough."""
        assert get_llm("anthropic").max_tokens == 8192

    def test_max_tokens_reaches_the_payload(self):
        assert _payload(get_llm("anthropic"))["max_tokens"] == 8192

    def test_effort_is_medium(self):
        """effort replaces temperature as the determinism/cost lever."""
        assert get_llm("anthropic").effort == "medium"

    def test_effort_reaches_the_payload(self):
        payload = _payload(get_llm("anthropic"))
        assert payload["output_config"]["effort"] == "medium"


class TestOtherProvidersUnchanged:
    """Only the Anthropic branch moves — the other two accept temperature."""

    def test_openai_still_deterministic(self):
        llm = get_llm("openai")
        assert llm.model_name == "gpt-4o-mini"
        assert llm.temperature == 0

    def test_ollama_still_deterministic(self):
        llm = get_llm("ollama")
        assert llm.model == "gpt-oss:20b"
        assert llm.temperature == 0
