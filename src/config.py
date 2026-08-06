"""LLM provider configuration factory."""

# The model used when --model is not given. The CLI builds its --help text and
# its banner from this table rather than restating the names, so the advertised
# default cannot drift from the one actually constructed. It did once: --help
# named a Haiku build long after the Anthropic branch had moved to Opus.
DEFAULT_MODELS = {
    "ollama": "gpt-oss:20b",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-opus-5",
}

# The credential each hosted provider needs before it can answer at all. Ollama
# runs locally and needs none, which is why the demo works without a key.
#
# Deliberately not enforced in get_llm: the constructors here take no
# credential, so the whole test suite builds every provider without one. The
# check belongs at the CLI edge, where a missing key is a user error worth
# reporting cleanly rather than 12 frames deep on the first prompt.
REQUIRED_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def get_llm(provider: str, model: str | None = None):
    """Create a chat model instance for the given provider.

    Args:
        provider: One of "ollama", "openai", "anthropic".
        model: Optional model name override. Falls back to sensible defaults.

    Returns:
        A LangChain BaseChatModel instance.
    """
    match provider:
        case "openai":
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:
                raise ImportError(
                    "Install langchain-openai to use the OpenAI provider: "
                    "uv add langchain-openai"
                ) from exc
            return ChatOpenAI(model=model or DEFAULT_MODELS["openai"], temperature=0)

        case "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError as exc:
                raise ImportError(
                    "Install langchain-anthropic to use the Anthropic provider: "
                    "uv add langchain-anthropic"
                ) from exc

            return ChatAnthropic(
                model=model or DEFAULT_MODELS["anthropic"],
                max_tokens=8192,
                effort="medium",
            )

        case "ollama":
            try:
                from langchain_ollama import ChatOllama
            except ImportError as exc:
                raise ImportError(
                    "Install langchain-ollama to use the Ollama provider: "
                    "uv add langchain-ollama"
                ) from exc
            return ChatOllama(model=model or DEFAULT_MODELS["ollama"], temperature=0)

        case _:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                "Supported providers: ollama, openai, anthropic"
            )
