"""Tests for the cancellation research agent — external cancellation knowledge only."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agents.cancellation_research.agent import (
    CANCELLATION_RESEARCH_TOOLS,
    _answer_pending_tool_calls,
    _build_system_prompt,
    create_cancellation_research_node,
)
from src.models import MOCK_USER


def _mock_llm(response=None):
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.invoke = MagicMock(return_value=response or AIMessage(content="Hi"))
    return llm


def _handoff_message():
    """The dangling tool call that hands control to this agent."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "find_recurring_payments",
                "args": {"user_id": "USR-2847"},
                "id": "call_find_1",
            }
        ],
    )


def _state(messages=None, user_info=None):
    return {
        "messages": messages if messages is not None else [_handoff_message()],
        "current_agent": "subscriptions",
        "user_info": user_info if user_info is not None else dict(MOCK_USER),
    }


class TestBoundTools:
    def test_binds_exactly_one_tool(self):
        assert len(CANCELLATION_RESEARCH_TOOLS) == 1

    def test_binds_find_cancellation_guide_only(self):
        assert [t.name for t in CANCELLATION_RESEARCH_TOOLS] == [
            "find_cancellation_guide"
        ]

    def test_binds_no_money_affecting_tools(self):
        # Read-only by construction: nothing here can move money or change a
        # mandate. The agent's whole remit is external knowledge.
        names = {t.name for t in CANCELLATION_RESEARCH_TOOLS}
        writes = {
            "cancel_direct_debit",
            "cancel_standing_order",
            "block_merchant_on_card",
            "make_payment",
            "transfer_funds",
            "freeze_card",
        }
        assert not (names & writes)

    def test_binds_no_separate_web_search_tool(self):
        # Directory-first ordering is enforced inside find_cancellation_guide.
        # A second, independent search tool would let the model bypass it.
        names = {t.name for t in CANCELLATION_RESEARCH_TOOLS}
        assert not any("search" in n and "knowledge" not in n for n in names)

    def test_node_binds_that_exact_list(self):
        llm = _mock_llm()
        create_cancellation_research_node(llm)
        llm.bind_tools.assert_called_once_with(CANCELLATION_RESEARCH_TOOLS)


class TestSystemPrompt:
    def _prompt(self):
        return _build_system_prompt(dict(MOCK_USER))

    def test_names_the_directory_as_the_primary_source(self):
        prompt = self._prompt().lower()
        assert "directory" in prompt
        assert "primary" in prompt or "first" in prompt

    def test_describes_web_search_as_a_fallback_only(self):
        prompt = self._prompt().lower()
        assert "fallback" in prompt
        assert "no entry" in prompt or "not in" in prompt

    def test_requires_stating_the_source_of_guidance(self):
        prompt = self._prompt()
        assert "source" in prompt.lower()
        assert "verified_on" in prompt

    def test_requires_labelling_unverified_web_results(self):
        prompt = self._prompt().lower()
        assert "unverified" in prompt

    def test_names_the_only_bound_tool(self):
        assert "find_cancellation_guide" in self._prompt()

    def test_names_the_customer(self):
        assert "Sarah" in _build_system_prompt(dict(MOCK_USER))

    def test_does_not_hardcode_the_default_customer(self):
        prompt = _build_system_prompt({"user_id": "USR-9999", "name": "Alex Rivera"})
        assert "USR-2847" not in prompt
        assert "Sarah" not in prompt
        assert "Alex" in prompt

    def test_carries_the_user_id_from_user_info(self):
        prompt = _build_system_prompt({"user_id": "USR-9999", "name": "Alex Rivera"})
        assert "USR-9999" in prompt

    def test_survives_missing_user_info(self):
        prompt = _build_system_prompt({})
        assert "find_cancellation_guide" in prompt


class TestPendingToolCall:
    def test_returns_tool_message_for_the_pending_call(self):
        llm = _mock_llm()
        node = create_cancellation_research_node(llm)
        result = node(_state())

        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1
        assert tool_messages[0].tool_call_id == "call_find_1"

    def test_tool_message_precedes_the_llm_response(self):
        llm = _mock_llm(AIMessage(content="Here is how to cancel"))
        node = create_cancellation_research_node(llm)
        result = node(_state())

        assert isinstance(result["messages"][0], ToolMessage)
        assert result["messages"][-1].content == "Here is how to cancel"

    def test_tool_message_is_sent_to_the_llm(self):
        llm = _mock_llm()
        node = create_cancellation_research_node(llm)
        node(_state())

        sent = llm.invoke.call_args[0][0]
        assert isinstance(sent[-1], ToolMessage)
        assert sent[-1].tool_call_id == "call_find_1"

    def test_tool_message_names_the_agent(self):
        llm = _mock_llm()
        node = create_cancellation_research_node(llm)
        result = node(_state())

        assert "cancellation research" in result["messages"][0].content.lower()

    def test_answers_every_pending_call_on_the_message(self):
        handoff = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "route_to_agent",
                    "args": {
                        "agent_name": "cancellation_research",
                        "reason": "how to cancel",
                    },
                    "id": "call_route_1",
                },
                {
                    "name": "find_recurring_payments",
                    "args": {"user_id": "USR-2847"},
                    "id": "call_find_1",
                },
            ],
        )
        llm = _mock_llm()
        node = create_cancellation_research_node(llm)
        result = node(_state(messages=[handoff]))

        answered = {
            m.tool_call_id for m in result["messages"] if isinstance(m, ToolMessage)
        }
        assert answered == {"call_route_1", "call_find_1"}

    def test_no_tool_message_when_nothing_is_pending(self):
        # Re-entry from the agent's own tool node: the last message is already a
        # ToolMessage, so synthesising another would corrupt the history.
        messages = [
            HumanMessage(content="how do I cancel Netflix?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "find_cancellation_guide",
                        "args": {"merchant": "Netflix"},
                        "id": "call_guide_1",
                    }
                ],
            ),
            ToolMessage(content="{}", tool_call_id="call_guide_1"),
        ]
        llm = _mock_llm()
        node = create_cancellation_research_node(llm)
        result = node(_state(messages=messages))

        assert not [m for m in result["messages"] if isinstance(m, ToolMessage)]

    def test_no_tool_message_after_a_plain_human_turn(self):
        llm = _mock_llm()
        node = create_cancellation_research_node(llm)
        result = node(_state(messages=[HumanMessage(content="hello")]))

        assert not [m for m in result["messages"] if isinstance(m, ToolMessage)]

    def test_empty_history_is_tolerated(self):
        assert _answer_pending_tool_calls([]) == []


class TestNodeContract:
    def test_sets_current_agent_to_the_bare_name(self):
        llm = _mock_llm()
        node = create_cancellation_research_node(llm)
        assert node(_state())["current_agent"] == "cancellation_research"

    def test_current_agent_is_not_the_node_name(self):
        llm = _mock_llm()
        node = create_cancellation_research_node(llm)
        result = node(_state())
        assert result["current_agent"] != "cancellation_research_agent"
        assert result["current_agent"] != "cancellation_research_tools"

    def test_returns_the_llm_response_last(self):
        llm = _mock_llm(AIMessage(content="Cancel via netflix.com/cancelplan"))
        node = create_cancellation_research_node(llm)
        result = node(_state())
        assert result["messages"][-1].content == "Cancel via netflix.com/cancelplan"

    def test_system_prompt_is_first_message_to_the_llm(self):
        llm = _mock_llm()
        node = create_cancellation_research_node(llm)
        node(_state())

        sent = llm.invoke.call_args[0][0]
        assert sent[0].type == "system"
        assert "find_cancellation_guide" in sent[0].content

    def test_prompt_follows_a_non_default_customer(self):
        llm = _mock_llm()
        node = create_cancellation_research_node(llm)
        node(_state(user_info={"user_id": "USR-9999", "name": "Alex Rivera"}))

        system = llm.invoke.call_args[0][0][0]
        assert "USR-9999" in system.content
        assert "USR-2847" not in system.content

    def test_missing_user_info_does_not_crash(self):
        llm = _mock_llm()
        node = create_cancellation_research_node(llm)
        state = _state()
        del state["user_info"]
        assert node(state)["current_agent"] == "cancellation_research"
