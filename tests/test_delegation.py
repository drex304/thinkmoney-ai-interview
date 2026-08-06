"""Tests for the deterministic subscriptions -> cancellation research handoff.

The handoff is triggered by the payment rail in code, never by the model
choosing to delegate, so every assertion here runs without an LLM decision.

The failure this file exists to prevent: `unresolved_card_subs` is a WORK
QUEUE, not an inventory. If the cancellation research node does not drain what it handled,
the router re-reads the same non-empty queue every time control leaves the
subscriptions agent and the turn cycles until LangGraph's recursion limit
of 25 aborts it.
"""

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from src.agents.cancellation_research.agent import (
    create_cancellation_research_node,
    _drain_queue,
    merchant_matches,
)
from src.agents.subscriptions import card_subs_from_tool_messages
from src.graph import (
    _choice_from_offer,
    _merchants_already_researched,
    _route_from_cancellation_research,
    _route_from_subscriptions,
    build_graph,
)
from src.models import AgentState, MOCK_USER
from src.tools.rails import options_for_rail, resolve_rail
from src.agents.subscriptions import find_recurring_payments


def _guide_call(merchant: str, call_id: str = "call_guide"):
    """An AIMessage carrying a find_cancellation_guide call."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "find_cancellation_guide",
                "args": {"merchant": merchant},
                "id": call_id,
            }
        ],
    )


def _route_call(agent_name: str = "subscriptions", call_id: str = "call_route"):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "route_to_agent",
                "args": {"agent_name": agent_name, "reason": "subscriptions"},
                "id": call_id,
            }
        ],
    )


def _find_recurring_call(call_id: str = "call_find"):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "find_recurring_payments",
                "args": {"user_id": MOCK_USER["user_id"]},
                "id": call_id,
            }
        ],
    )


def _report_message(subscriptions, call_id: str = "call_find"):
    return ToolMessage(
        content=json.dumps({"subscriptions": subscriptions}),
        name="find_recurring_payments",
        tool_call_id=call_id,
    )


def _state(messages, queue=None):
    state = {
        "messages": messages,
        "current_agent": "subscriptions",
        "user_info": dict(MOCK_USER),
    }
    if queue is not None:
        state["unresolved_card_subs"] = queue
    return state


class TestStateSchema:
    def test_unresolved_card_subs_is_declared(self):
        assert "unresolved_card_subs" in AgentState.__annotations__

    def test_unresolved_card_subs_is_a_list(self):
        assert AgentState.__annotations__["unresolved_card_subs"] is list

    def test_docstring_documents_it_as_a_work_queue(self):
        """The semantics are the whole point — an inventory would never drain."""
        doc = AgentState.__doc__.lower()
        assert "unresolved_card_subs" in doc
        assert "work queue" in doc
        assert "still needing guidance" in doc


class TestRouterFromSubscriptions:
    """All three branches, no LLM involved."""

    def test_pending_tool_calls_go_to_the_tool_node(self):
        state = _state([_find_recurring_call()], queue=["Spotify"])
        assert _route_from_subscriptions(state) == "subscriptions_tools"

    def test_non_empty_queue_goes_to_cancellation_research(self):
        state = _state([AIMessage(content="Here is the breakdown.")], queue=["Spotify"])
        assert _route_from_subscriptions(state) == "cancellation_research_agent"

    def test_drained_queue_returns_to_triage(self):
        """The direct guard against the recursion-limit loop."""
        state = _state([AIMessage(content="All done.")], queue=[])
        assert _route_from_subscriptions(state) == "triage"
        assert _route_from_subscriptions(state) != "cancellation_research_agent"

    def test_absent_queue_returns_to_triage(self):
        state = _state([AIMessage(content="All done.")])
        assert _route_from_subscriptions(state) == "triage"

    def test_tool_calls_win_over_a_non_empty_queue(self):
        """Cancellation research waits until the agent has finished its own tool work."""
        state = _state([_find_recurring_call()], queue=["Spotify", "Adobe"])
        assert _route_from_subscriptions(state) == "subscriptions_tools"


class TestRouterFromCancellationResearch:
    def test_pending_tool_calls_go_to_the_cancellation_research_tool_node(self):
        state = _state([_guide_call("Spotify")], queue=["Spotify"])
        assert _route_from_cancellation_research(state) == "cancellation_research_tools"

    def test_always_returns_control_to_subscriptions(self):
        state = _state([AIMessage(content="Cancel it in account settings.")], queue=[])
        assert _route_from_cancellation_research(state) == "subscriptions_agent"

    def test_does_not_return_to_triage(self):
        """FR-22: cancellation research hands back to the specialist, not to the router."""
        state = _state([AIMessage(content="Done.")], queue=["Spotify"])
        assert _route_from_cancellation_research(state) != "triage"


class TestQueuePopulation:
    """The tool node reads the queue off find_recurring_payments rail labels."""

    def test_card_rail_merchants_are_queued(self):
        messages = [
            _report_message(
                [
                    {"merchant": "Spotify", "rail": "card_on_file"},
                    {"merchant": "Netflix", "rail": "direct_debit"},
                ]
            )
        ]
        assert card_subs_from_tool_messages(messages) == ["Spotify"]

    def test_direct_debit_and_standing_order_are_not_queued(self):
        messages = [
            _report_message(
                [
                    {"merchant": "Netflix", "rail": "direct_debit"},
                    {"merchant": "Landlord", "rail": "standing_order"},
                ]
            )
        ]
        assert card_subs_from_tool_messages(messages) == []

    def test_other_tool_results_are_ignored(self):
        messages = [
            ToolMessage(
                content=json.dumps(
                    {"subscriptions": [{"merchant": "X", "rail": "card_on_file"}]}
                ),
                name="get_transaction_history",
                tool_call_id="call_other",
            )
        ]
        assert card_subs_from_tool_messages(messages) == []

    def test_non_json_content_is_survived(self):
        messages = [
            ToolMessage(
                content="not json",
                name="find_recurring_payments",
                tool_call_id="call_find",
            )
        ]
        assert card_subs_from_tool_messages(messages) == []

    def test_duplicates_are_collapsed(self):
        messages = [
            _report_message([{"merchant": "Spotify", "rail": "card_on_file"}]),
            _report_message(
                [{"merchant": "Spotify", "rail": "card_on_file"}], call_id="c2"
            ),
        ]
        assert card_subs_from_tool_messages(messages) == ["Spotify"]

    def test_real_corpus_queues_every_card_subscription(self):
        report = json.loads(
            find_recurring_payments.invoke({"user_id": MOCK_USER["user_id"]})
        )
        expected = [
            s["merchant"]
            for s in report["subscriptions"]
            if s["rail"] == "card_on_file"
        ]
        queued = card_subs_from_tool_messages(
            [_report_message(report["subscriptions"])]
        )
        assert queued == expected
        assert queued, "the corpus must contain card-on-file subscriptions"


class TestAlreadyResearched:
    """A merchant researched earlier in the turn is never re-queued."""

    def test_reads_merchants_off_guide_calls(self):
        messages = [_guide_call("Spotify"), _guide_call("Adobe", "call_2")]
        assert _merchants_already_researched(messages) == ["Spotify", "Adobe"]

    def test_ignores_other_tool_calls(self):
        assert _merchants_already_researched([_find_recurring_call()]) == []

    def test_ignores_plain_messages(self):
        assert _merchants_already_researched([HumanMessage(content="hi")]) == []


class TestMerchantMatching:
    def test_exact_match(self):
        assert merchant_matches("Spotify", "Spotify")

    def test_case_and_whitespace_insensitive(self):
        assert merchant_matches("Apple  iCloud+", "apple icloud+")

    def test_statement_description_matches_the_brand_asked_about(self):
        assert merchant_matches("Adobe Creative Cloud", "Adobe")

    def test_unrelated_merchants_do_not_match(self):
        assert not merchant_matches("Spotify", "Google One")

    def test_empty_never_matches(self):
        assert not merchant_matches("Spotify", "")


class TestQueueDrain:
    """The cancellation research node's state update is what terminates the handoff."""

    def test_removes_exactly_the_merchant_it_handled(self):
        remaining = _drain_queue(
            ["Spotify", "Adobe Creative Cloud"], _guide_call("Spotify")
        )
        assert remaining == ["Adobe Creative Cloud"]

    def test_finishing_drains_everything_it_was_handed(self):
        """No tool calls means cancellation research has said its piece; re-queueing loops."""
        remaining = _drain_queue(
            ["Spotify", "Adobe Creative Cloud"], AIMessage(content="Here is how.")
        )
        assert remaining == []

    def test_empty_queue_stays_empty(self):
        assert _drain_queue([], _guide_call("Spotify")) == []

    def test_node_state_update_drains_the_handled_merchant(self):
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        llm.invoke = MagicMock(return_value=_guide_call("Spotify"))
        node = create_cancellation_research_node(llm)

        result = node(
            _state([_route_call()], queue=["Spotify", "Adobe Creative Cloud"])
        )

        assert result["unresolved_card_subs"] == ["Adobe Creative Cloud"]

    def test_node_always_reports_the_queue(self):
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        llm.invoke = MagicMock(return_value=AIMessage(content="Cancel in settings."))
        node = create_cancellation_research_node(llm)

        result = node(_state([_route_call()], queue=["Spotify"]))

        assert result["unresolved_card_subs"] == []

    def test_node_still_returns_messages_and_current_agent(self):
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        llm.invoke = MagicMock(return_value=AIMessage(content="Cancel in settings."))
        node = create_cancellation_research_node(llm)

        result = node(_state([_route_call()], queue=["Spotify"]))

        assert result["current_agent"] == "cancellation_research"
        assert result["messages"][-1].content == "Cancel in settings."


class TestGraphWiring:
    @pytest.fixture
    def edges(self):
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        graph = build_graph(llm)
        declared = set(graph.builder.edges)
        for source, branches in graph.builder.branches.items():
            for branch in branches.values():
                declared |= {
                    (source, target) for target in (branch.ends or {}).values()
                }
        return declared

    def test_subscriptions_can_hand_off_to_cancellation_research(self, edges):
        assert ("subscriptions_agent", "cancellation_research_agent") in edges

    def test_cancellation_research_returns_to_subscriptions(self, edges):
        assert ("cancellation_research_agent", "subscriptions_agent") in edges

    def test_cancellation_research_does_not_return_to_triage(self, edges):
        assert ("cancellation_research_agent", "triage") not in edges

    def test_subscriptions_still_returns_to_triage(self, edges):
        assert ("subscriptions_agent", "triage") in edges


class TestFullTurn:
    """The regression test for the recursion-limit failure.

    The customer names Spotify here. That is what puts it in the queue: the
    handoff is scoped to the merchants the turn is about, so a request that
    names none of them never reaches cancellation research at all — which is
    what TestReadOnlyTurn below covers.
    """

    @pytest.fixture
    def visits(self):
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        llm.invoke = MagicMock(
            side_effect=[
                _route_call(),  # triage routes
                _find_recurring_call(),  # subscriptions calls detection
                AIMessage(content="Spotify is on your card."),  # subscriptions answers
                AIMessage(
                    content="Cancel Spotify in account settings."
                ),  # cancellation research answers
                AIMessage(content="That is everything."),  # subscriptions wraps up
                AIMessage(content="Anything else?"),  # triage closes
            ]
        )
        graph = build_graph(llm)

        seen = []
        for update in graph.stream(
            {
                "messages": [HumanMessage(content="How do I cancel Spotify?")],
                "user_info": dict(MOCK_USER),
                "current_agent": "triage",
            },
            stream_mode="updates",
        ):
            seen.extend(update.keys())
        return seen

    def test_cancellation_research_is_entered_exactly_once(self, visits):
        assert visits.count("cancellation_research_agent") == 1

    def test_trace_is_subscriptions_cancellation_research_subscriptions_triage(
        self, visits
    ):
        first_cancellation_research = visits.index("cancellation_research_agent")
        assert "subscriptions_agent" in visits[:first_cancellation_research]
        assert "subscriptions_agent" in visits[first_cancellation_research + 1 :]
        assert "triage" in visits[first_cancellation_research + 1 :]

    def test_the_turn_terminates_well_inside_the_recursion_limit(self, visits):
        assert len(visits) < 25


class TestReadOnlyTurn:
    """A question that names no merchant must not reach cancellation research.

    The trigger was once the rail label alone, so holding any card-billed
    subscription sent every one of them for research. Asked "what subscriptions
    am I paying for?", the agent answered with the list and then, unprompted,
    the cancellation steps for a gym — five directory lookups to produce advice
    nobody had asked for.
    """

    @pytest.fixture
    def visits(self):
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        llm.invoke = MagicMock(
            side_effect=[
                _route_call(),  # triage routes
                _find_recurring_call(),  # subscriptions calls detection
                AIMessage(content="Here is what you pay for."),  # subscriptions answers
                AIMessage(content="Anything else?"),  # triage closes
            ]
        )
        graph = build_graph(llm)

        seen = []
        for update in graph.stream(
            {
                "messages": [
                    HumanMessage(content="What subscriptions am I paying for?")
                ],
                "user_info": dict(MOCK_USER),
                "current_agent": "triage",
            },
            stream_mode="updates",
        ):
            seen.extend(update.keys())
        return seen

    def test_cancellation_research_is_never_entered(self, visits):
        assert "cancellation_research_agent" not in visits

    def test_the_answer_still_comes_back_through_triage(self, visits):
        assert "subscriptions_agent" in visits
        assert visits[-1] == "triage"

    def test_queue_is_drained_at_the_end_of_the_turn(self):
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        llm.invoke = MagicMock(
            side_effect=[
                _route_call(),
                _find_recurring_call(),
                AIMessage(content="Spotify is on your card."),
                AIMessage(content="Cancel Spotify in account settings."),
                AIMessage(content="That is everything."),
                AIMessage(content="Anything else?"),
            ]
        )
        final = build_graph(llm).invoke(
            {
                "messages": [HumanMessage(content="What am I paying for?")],
                "user_info": dict(MOCK_USER),
                "current_agent": "triage",
            }
        )
        assert final["unresolved_card_subs"] == []


class TestWriteNeedsAConfirmedReferent:
    """A write must be bound to a subscription the customer identified.

    `resolve_rail` makes the mechanism and the reference correct for whatever
    merchant is named. Nothing checked the name, so a model naming the wrong
    subscription produced a correct cancellation of the wrong thing. This is the
    check that catches it — and, as importantly, the one that must not fire on
    "cancel it".
    """

    def _run(self, request, merchant):
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        llm.invoke = MagicMock(
            side_effect=[
                _route_call(),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "block_merchant_on_card",
                            "args": {
                                "user_id": MOCK_USER["user_id"],
                                "merchant": merchant,
                            },
                            "id": "call_write",
                        }
                    ],
                ),
                AIMessage(content="done"),
                AIMessage(content="anything else?"),
            ]
        )
        graph = build_graph(llm, checkpointer=MemorySaver())
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=request)],
                "user_info": dict(MOCK_USER),
                "current_agent": "triage",
            },
            {"configurable": {"thread_id": "wrong-merchant"}},
        )
        return result

    def _refusal(self, result):
        for message in result["messages"]:
            if (
                isinstance(message, ToolMessage)
                and message.tool_call_id == "call_write"
            ):
                return message.content
        return None

    def test_a_different_merchant_is_refused_before_the_gate(self):
        result = self._run("cancel Google One", "Spotify")
        assert "__interrupt__" not in result
        refusal = self._refusal(result)
        assert "Refused" in refusal
        assert "Google One" in refusal

    def test_the_merchant_they_named_reaches_the_gate(self):
        result = self._run("cancel Google One", "Google One")
        assert "__interrupt__" in result

    def test_a_request_naming_no_merchant_asks_instead_of_choosing(self):
        """The fail-open case, inverted.

        "cancel it" used to let the model pick freely, so a wrong pick went
        through silently. Which subscription someone means by "it" is not in the
        data, so the write is refused and the question goes back to the person
        who knows the answer.
        """
        result = self._run("cancel it", "Spotify")
        assert "__interrupt__" not in result
        assert "has not said which subscription" in self._refusal(result)

    def test_the_options_are_scoped_to_what_was_proposed(self):
        """The proposed name is a hint about the conversation, not authority.

        Unscoped, a customer asking about Netflix was offered every direct debit
        on the account — which put their domain renewal in a list about
        streaming. The customer still chooses; the scope only decides what is
        worth putting in front of them.
        """
        refusal = self._refusal(self._run("cancel it", "Spotify"))
        assert "Spotify" in refusal
        for unrelated in ("FitLife Gym", "Google One", "Apple iCloud+"):
            assert unrelated not in refusal

    def test_an_unmatched_proposal_falls_back_to_the_whole_rail(self):
        """An empty question is worse than a broad one."""
        refusal = self._refusal(self._run("cancel it", "Not A Real Merchant"))
        for merchant in ("FitLife Gym", "Spotify", "Google One"):
            assert merchant in refusal

    def test_the_customer_can_still_answer_with_another_name(self):
        # Scoping must not trap them inside the model's suggestion.
        refusal = self._refusal(self._run("cancel it", "Spotify"))
        assert "number or a name" in refusal

    def test_the_options_are_limited_to_the_rail_that_tool_acts_on(self):
        """A card block cannot act on a direct debit, so it must not offer one."""
        refusal = self._refusal(self._run("cancel it", "Spotify"))
        assert "Netflix" not in refusal
        assert "Namecheap" not in refusal

    def test_the_options_are_numbered_so_the_answer_can_be_bound(self):
        """Customers answer a numbered list with a number.

        The first version asked for a name, and a customer replying "2" named
        nothing — so the write was refused for having no referent and the same
        question came back. The list is numbered and kept in state, so the
        answer resolves against the question that produced it.
        """
        refusal = self._refusal(self._run("cancel it", "Spotify"))
        assert "1." in refusal and "2." in refusal
        assert "reply with a number or a name" in refusal.lower()

    def test_the_agent_is_told_not_to_renumber_the_list(self):
        # The index is resolved against the list the code produced, so an agent
        # that renumbers would make "2" select something else entirely.
        refusal = self._refusal(self._run("cancel it", "Spotify"))
        assert "exactly as written" in refusal
        assert "Do not renumber" in refusal

    def test_a_category_word_does_not_bind_the_write(self):
        """ "my gym membership" identifies a subscription to a human, not to code.

        Matching it would mean guessing from category words, which is the kind
        of inference this check exists to remove. The cost is one extra
        question; the alternative is acting on an assumption.
        """
        result = self._run("cancel my gym membership", "FitLife Gym")
        assert "__interrupt__" not in result
        assert "has not said which subscription" in self._refusal(result)


class TestChoiceFromOffer:
    """Resolving "2" against the list that produced the question."""

    OFFER = ["Netflix", "Namecheap Domain Renewal"]

    def test_a_number_selects_that_entry(self):
        assert _choice_from_offer("1", self.OFFER) == ["Netflix"]
        assert _choice_from_offer("2", self.OFFER) == ["Namecheap Domain Renewal"]

    def test_surrounding_whitespace_is_tolerated(self):
        assert _choice_from_offer("  2  ", self.OFFER) == ["Namecheap Domain Renewal"]

    @pytest.mark.parametrize("answer", ["0", "3", "99"])
    def test_a_number_outside_the_list_selects_nothing(self, answer):
        assert _choice_from_offer(answer, self.OFFER) == []

    @pytest.mark.parametrize(
        "answer",
        ["£2.99", "cancel 2 of these", "the 2nd one", "two", "yes"],
    )
    def test_only_a_bare_number_counts_as_a_selection(self, answer):
        """Anything else would be reading a choice into text that made none."""
        assert _choice_from_offer(answer, self.OFFER) == []

    def test_nothing_offered_means_nothing_selected(self):
        assert _choice_from_offer("1", []) == []
        assert _choice_from_offer("1", None) == []


class TestEssentialsAreNeverOffered:
    """Asked about Netflix, the list offered the customer their rent."""

    def test_rent_is_not_among_the_standing_orders_offered(self):
        merchants = [row["merchant"] for row in options_for_rail("standing_order")]
        assert "James Wilson - Shared Netflix" in merchants
        assert "Landlord - Premier Properties" not in merchants

    def test_essentials_are_still_cancellable_when_named(self):
        """Excluded from the menu, not from the account.

        A customer who genuinely means the landlord standing order can say so,
        and resolve_rail still finds it.
        """
        match, error = resolve_rail("Landlord - Premier Properties", "standing_order")
        assert error == ""
        assert match["reference"] == "SO-101"


class TestChoicePhrasing:
    """How people actually answer a numbered list."""

    OFFER = ["Netflix", "James Wilson - Shared Netflix"]

    @pytest.mark.parametrize(
        "answer", ["1", " 1 ", "cancel 1", "number 1", "option 1", "the 1", "please 1"]
    )
    def test_a_number_with_a_leading_verb_still_selects(self, answer):
        """ "cancel 1" is how the customer answered, and it was refused.

        Requiring a bare digit sent them back round the same question.
        """
        assert _choice_from_offer(answer, self.OFFER) == ["Netflix"]

    @pytest.mark.parametrize(
        "answer",
        ["cancel 2 of these", "£2.99", "the 2nd one", "two", "yes", "1 and 2"],
    )
    def test_anything_that_is_not_purely_a_choice_selects_nothing(self, answer):
        assert _choice_from_offer(answer, self.OFFER) == []
