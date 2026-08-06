"""Tests for the subscriptions specialist agent."""

import json
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import pytest

from src.agents.subscriptions.agent import (
    merchants_named_in,
    SUBSCRIPTIONS_TOOLS,
    _build_system_prompt,
    _headline,
    create_subscriptions_node,
)
from src.models import MOCK_USER
from src.agents.subscriptions import find_recurring_payments


def _mock_llm(response=None):
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.invoke = MagicMock(return_value=response or AIMessage(content="Hi"))
    return llm


def _routing_message():
    """The dangling triage tool call the node has to satisfy."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "route_to_agent",
                "args": {"agent_name": "subscriptions", "reason": "wants to cut costs"},
                "id": "call_route_1",
            }
        ],
    )


def _state(messages=None, user_info=None):
    return {
        "messages": messages if messages is not None else [_routing_message()],
        "current_agent": "triage",
        "user_info": user_info if user_info is not None else dict(MOCK_USER),
    }


class TestBoundTools:
    def test_binds_exactly_eight_tools(self):
        assert len(SUBSCRIPTIONS_TOOLS) == 8

    def test_binds_the_named_tools(self):
        assert [t.name for t in SUBSCRIPTIONS_TOOLS] == [
            "find_recurring_payments",
            "get_transaction_history",
            "cancel_standing_order",
            "cancel_direct_debit",
            "block_merchant_on_card",
            "list_cards",
            "get_payees",
            "search_knowledge_base",
        ]

    def test_five_bound_tools_ship_with_the_exercise(self):
        provided = {
            "get_transaction_history",
            "cancel_standing_order",
            "list_cards",
            "get_payees",
            "search_knowledge_base",
        }
        assert provided <= {t.name for t in SUBSCRIPTIONS_TOOLS}

    def test_get_standing_orders_is_not_bound(self):
        # find_recurring_payments already reads that fixture; a second lookup
        # would waste a slot at the eight-tool cap.
        assert "get_standing_orders" not in {t.name for t in SUBSCRIPTIONS_TOOLS}

    def test_no_duplicate_tools(self):
        names = [t.name for t in SUBSCRIPTIONS_TOOLS]
        assert len(names) == len(set(names))

    def test_node_binds_that_exact_list(self):
        llm = _mock_llm()
        create_subscriptions_node(llm)
        llm.bind_tools.assert_called_once_with(SUBSCRIPTIONS_TOOLS)


class TestHeadline:
    def test_matches_the_detection_tool(self):
        report = json.loads(find_recurring_payments.invoke({"user_id": "USR-2847"}))
        headline = _headline("USR-2847")

        assert headline["annualised_total"] == report["totals"]["annualised_total"]
        assert headline["monthly_total"] == report["totals"]["monthly_total"]
        assert headline["identified_saving"] == report["savings"]["identified_saving"]
        assert headline["subscription_count"] == report["totals"]["subscription_count"]

    def test_is_deterministic(self):
        assert _headline("USR-2847") == _headline("USR-2847")


class TestSystemPrompt:
    def _prompt(self):
        return _build_system_prompt(dict(MOCK_USER), _headline(MOCK_USER["user_id"]))

    def test_leads_with_the_annualised_headline(self):
        prompt = self._prompt()
        annualised = _headline(MOCK_USER["user_id"])["annualised_total"]
        opening = prompt[: prompt.index("\n\n", prompt.index(annualised))]
        assert annualised in opening

    def test_states_both_saving_figures_separately(self):
        prompt = self._prompt()
        headline = _headline(MOCK_USER["user_id"])
        assert headline["identified_saving"] in prompt
        assert headline["potential_saving"] in prompt

    def test_forbids_adding_the_two_saving_figures_together(self):
        # The potential figure depends on the customer's answer; presenting a
        # combined number would restate a question as a finding.
        prompt = self._prompt().lower()
        assert "never add them together" in prompt
        assert "no way of knowing" in prompt

    def test_requires_confirmation_before_writes(self):
        prompt = self._prompt().lower()
        assert "confirm" in prompt
        assert "before" in prompt

    def test_forbids_writing_when_the_customer_only_asked_a_question(self):
        prompt = self._prompt().lower()
        assert "a question is not an instruction" in prompt
        assert "do not call a write tool" in prompt

    def test_states_the_precondition_before_the_instruction_to_write(self):
        """Ordering is the bug, not absence.

        The prompt already carried "when the customer asks you to cancel", but
        six sentences below a bolded "Call the write tool. Do not ask first."
        Asked a read-only question, one provider followed the headline and
        opened a cancellation the customer never requested. The licence to write
        has to arrive after the condition that grants it, not before.
        """
        prompt = self._prompt().lower()
        assert prompt.index("a question is not an instruction") < prompt.index(
            "call the write tool"
        )

    def test_tells_the_agent_to_pass_the_merchant_rather_than_a_reference(self):
        """The prompt is no longer where wrong-target protection lives.

        It used to say "act only on that strategy's rail_reference", which is
        advice a model can misread — and did, cancelling Namecheap's mandate
        when asked about Google One. The write tools now resolve the reference
        from the merchant themselves, so the prompt's job is only to stop the
        model supplying one.
        """
        prompt = self._prompt().lower()
        assert "name the merchant, not a reference" in prompt
        assert "mandate_id" in prompt

    def test_tells_the_agent_a_refusal_is_information(self):
        prompt = self._prompt().lower()
        assert "success false" in prompt
        assert "do not retry the same call" in prompt

    def test_carries_the_blocking_is_not_cancelling_caveat(self):
        prompt = self._prompt().lower()
        assert "does not cancel" in prompt
        assert "still owe" in prompt

    def test_warns_against_blanket_cancelling_essentials(self):
        prompt = self._prompt().lower()
        assert "rent" in prompt
        assert "essential" in prompt

    def test_requires_knowledge_base_before_asserting_policy(self):
        prompt = self._prompt()
        assert "search_knowledge_base" in prompt
        assert "fee" in prompt.lower()

    def test_names_the_customer(self):
        prompt = _build_system_prompt(dict(MOCK_USER), _headline("USR-2847"))
        assert "Sarah" in prompt

    def test_carries_the_customer_identifiers_from_user_info(self):
        prompt = _build_system_prompt(
            {
                "user_id": "USR-9999",
                "name": "Alex Rivera",
                "primary_card_id": "CARD-8834",
                "account_id": "ACC-0001",
            },
            _headline("USR-9999"),
        )
        assert "USR-9999" in prompt
        assert "CARD-8834" in prompt
        assert "Alex" in prompt

    def test_does_not_hardcode_the_default_customer(self):
        prompt = _build_system_prompt(
            {
                "user_id": "USR-9999",
                "name": "Alex Rivera",
                "primary_card_id": "CARD-8834",
            },
            _headline("USR-9999"),
        )
        assert "USR-2847" not in prompt
        assert "CARD-5521" not in prompt
        assert "Sarah" not in prompt


class TestPendingRouteToolCall:
    def test_returns_tool_message_for_the_pending_call(self):
        llm = _mock_llm()
        node = create_subscriptions_node(llm)
        result = node(_state())

        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1
        assert tool_messages[0].tool_call_id == "call_route_1"

    def test_tool_message_precedes_the_llm_response(self):
        llm = _mock_llm(AIMessage(content="Here are your subscriptions"))
        node = create_subscriptions_node(llm)
        result = node(_state())

        assert isinstance(result["messages"][0], ToolMessage)
        assert result["messages"][-1].content == "Here are your subscriptions"

    def test_tool_message_is_sent_to_the_llm(self):
        llm = _mock_llm()
        node = create_subscriptions_node(llm)
        node(_state())

        sent = llm.invoke.call_args[0][0]
        assert isinstance(sent[-1], ToolMessage)
        assert sent[-1].tool_call_id == "call_route_1"

    def test_tool_message_names_the_agent(self):
        llm = _mock_llm()
        node = create_subscriptions_node(llm)
        result = node(_state())

        assert "subscriptions" in result["messages"][0].content.lower()

    def test_answers_every_pending_call_on_the_message(self):
        routing = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "route_to_agent",
                    "args": {"agent_name": "subscriptions", "reason": "costs"},
                    "id": "call_route_1",
                },
                {
                    "name": "search_knowledge_base",
                    "args": {"query": "cancellation fees"},
                    "id": "call_kb_1",
                },
            ],
        )
        llm = _mock_llm()
        node = create_subscriptions_node(llm)
        result = node(_state(messages=[routing]))

        answered = {
            m.tool_call_id for m in result["messages"] if isinstance(m, ToolMessage)
        }
        assert answered == {"call_route_1", "call_kb_1"}

    def test_no_tool_message_when_nothing_is_pending(self):
        # Re-entry after the agent's own tool node: the last message already is a
        # ToolMessage, so synthesising another would corrupt the history.
        messages = [
            HumanMessage(content="cancel Netflix"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "find_recurring_payments",
                        "args": {"user_id": "USR-2847"},
                        "id": "call_find_1",
                    }
                ],
            ),
            ToolMessage(content="{}", tool_call_id="call_find_1"),
        ]
        llm = _mock_llm()
        node = create_subscriptions_node(llm)
        result = node(_state(messages=messages))

        assert not [m for m in result["messages"] if isinstance(m, ToolMessage)]

    def test_no_tool_message_for_a_plain_user_turn(self):
        llm = _mock_llm()
        node = create_subscriptions_node(llm)
        result = node(_state(messages=[HumanMessage(content="what do I pay for?")]))

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)


class TestNodeStateUpdate:
    def test_sets_current_agent_to_the_bare_name(self):
        llm = _mock_llm()
        node = create_subscriptions_node(llm)
        result = node(_state())

        assert result["current_agent"] == "subscriptions"

    def test_returns_only_messages_and_current_agent(self):
        llm = _mock_llm()
        node = create_subscriptions_node(llm)
        result = node(_state())

        assert set(result) == {"messages", "current_agent"}

    def test_returns_callable(self):
        assert callable(create_subscriptions_node(_mock_llm()))


class TestCustomerIdentifiersComeFromState:
    def test_user_id_follows_a_non_default_customer(self):
        llm = _mock_llm()
        node = create_subscriptions_node(llm)
        node(
            _state(
                user_info={
                    "user_id": "USR-9999",
                    "name": "Alex Rivera",
                    "primary_card_id": "CARD-8834",
                    "account_id": "ACC-0001",
                }
            )
        )

        system = llm.invoke.call_args[0][0][0]
        assert "USR-9999" in system.content
        assert "USR-2847" not in system.content

    def test_card_id_follows_a_non_default_customer(self):
        llm = _mock_llm()
        node = create_subscriptions_node(llm)
        node(
            _state(
                user_info={
                    "user_id": "USR-9999",
                    "name": "Alex Rivera",
                    "primary_card_id": "CARD-8834",
                }
            )
        )

        system = llm.invoke.call_args[0][0][0]
        assert "CARD-8834" in system.content
        assert "CARD-5521" not in system.content

    def test_falls_back_gracefully_with_empty_user_info(self):
        llm = _mock_llm()
        node = create_subscriptions_node(llm)
        result = node(_state(user_info={}))

        assert result["current_agent"] == "subscriptions"
        system = llm.invoke.call_args[0][0][0]
        assert "Customer" in system.content

    def test_system_prompt_is_first_message_sent(self):
        llm = _mock_llm()
        node = create_subscriptions_node(llm)
        routing = _routing_message()
        node(_state(messages=[routing]))

        sent = llm.invoke.call_args[0][0]
        assert sent[0].type == "system"
        assert sent[1] is routing
        assert len(sent) == 3


class TestMerchantsNamedIn:
    """Which subscriptions the customer's own words identify.

    Feeds the wrong-merchant guard, so what matters as much as the matches is
    the non-matches: a request naming nothing must stay empty, or the guard
    would refuse ordinary requests like "cancel it".
    """

    def test_a_named_merchant_is_found(self):
        assert merchants_named_in("cancel Google One") == ["Google One"]

    def test_an_alias_from_the_directory_is_found(self):
        assert merchants_named_in("cancel fitlife") == ["FitLife Gym"]

    @pytest.mark.parametrize(
        "request_text",
        [
            "cancel it",
            "cancel my gym membership",
            "cancel the cheaper one",
            "what am I paying for?",
            "",
        ],
    )
    def test_a_request_naming_nothing_is_empty(self, request_text):
        assert merchants_named_in(request_text) == []

    def test_an_ambiguous_name_returns_every_match(self):
        """ "Netflix" is two subscriptions, so neither may be ruled out.

        Returning only one would make acting on the other look like a
        contradiction and refuse a legitimate request.
        """
        assert set(merchants_named_in("cancel Netflix")) == {
            "Netflix",
            "James Wilson - Shared Netflix",
        }
