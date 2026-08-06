"""Tests for the confirmation gate carried by the write tools.

The guarantee under test is structural, not conversational: no write tool
executes until the graph has halted with `interrupt()` and been resumed with an
approval. A prompt instruction telling the model to ask first is a request; this
is a gate the model cannot route around, because the gate is inside the tool and
the tool will not reach its own body until a human answer comes back through the
checkpointer.
"""

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.confirmation import CONFIRMED_TOOLS, _succeeded, is_approval
from src.graph import build_graph
from src.models import MOCK_USER
from src.agents.subscriptions import detection
from src.tools import cards as card_tools
from src.tools import payments as payment_tools


def _route_call(call_id: str = "call_route"):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "route_to_agent",
                "args": {"agent_name": "subscriptions", "reason": "cancellation"},
                "id": call_id,
            }
        ],
    )


def _tool_call(name: str, args: dict, call_id: str):
    return {"name": name, "args": args, "id": call_id}


def _ai_with_calls(*calls):
    return AIMessage(content="", tool_calls=list(calls))


def _cancel_dd_call(call_id: str = "call_dd"):
    return _ai_with_calls(
        _tool_call(
            "cancel_direct_debit",
            {"user_id": MOCK_USER["user_id"], "merchant": "Netflix"},
            call_id,
        )
    )


@pytest.fixture
def spies(monkeypatch):
    """Record every write-tool execution, so 'it did not run' is assertable."""
    calls: dict[str, list] = {
        "cancel_direct_debit": [],
        "cancel_standing_order": [],
        "block_merchant_on_card": [],
        "find_recurring_payments": [],
    }

    def wrap(tool, key):
        # Patch the body the gate guards, not the gated function, so a recorded
        # call means the tool really executed rather than merely being asked for.
        gated = getattr(tool.func, "__wrapped__", None)
        original = gated or tool.func

        def recorder(*args, **kwargs):
            calls[key].append(kwargs or args)
            return original(*args, **kwargs)

        if gated is not None:
            monkeypatch.setattr(tool.func, "__wrapped__", recorder)
        else:
            monkeypatch.setattr(tool, "func", recorder)

    wrap(payment_tools.cancel_direct_debit, "cancel_direct_debit")
    wrap(card_tools.block_merchant_on_card, "block_merchant_on_card")
    wrap(detection.find_recurring_payments, "find_recurring_payments")
    wrap(payment_tools.cancel_standing_order, "cancel_standing_order")
    return calls


def _scripted_llm(*responses):
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.invoke = MagicMock(side_effect=list(responses))
    return llm


def _run(llm, thread_id: str = "test-thread"):
    """Start a turn on a checkpointed graph and return (graph, config, result)."""
    checkpointer = MemorySaver()
    graph = build_graph(llm, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="Cancel my Netflix direct debit.")],
            "user_info": dict(MOCK_USER),
            "current_agent": "triage",
        },
        config,
    )
    return graph, config, result


def _tool_messages(messages, name: str) -> list[ToolMessage]:
    return [
        m
        for m in messages
        if isinstance(m, ToolMessage) and getattr(m, "name", None) == name
    ]


def _answer_to(messages, tool_call_id: str):
    """The ToolMessage answering a specific call, or None."""
    for message in messages:
        if isinstance(message, ToolMessage) and message.tool_call_id == tool_call_id:
            return message
    return None


class TestWriteToolRegistry:
    """Carrying the decorator *is* the registration, so this is the whole check
    that no money-moving tool was added without a gate."""

    def test_the_three_money_moving_tools_are_gated(self):
        assert CONFIRMED_TOOLS >= {
            "cancel_standing_order",
            "cancel_direct_debit",
            "block_merchant_on_card",
        }

    def test_read_tools_are_not_gated(self):
        assert "find_recurring_payments" not in CONFIRMED_TOOLS
        assert "find_cancellation_guide" not in CONFIRMED_TOOLS
        assert "get_transaction_history" not in CONFIRMED_TOOLS

    def test_every_gated_name_is_a_tool_the_subscriptions_agent_can_call(self):
        from src.agents.subscriptions import SUBSCRIPTIONS_TOOLS

        bound = {tool.name for tool in SUBSCRIPTIONS_TOOLS}
        assert CONFIRMED_TOOLS <= bound

    def test_the_injected_call_id_is_hidden_from_the_model(self):
        # It is there for the audit trail; a model that can see it will fill it
        # in, and it counts against the schema a small model has to reason about.
        for tool in (
            payment_tools.cancel_direct_debit,
            card_tools.block_merchant_on_card,
            payment_tools.cancel_standing_order,
        ):
            assert "tool_call_id" not in tool.tool_call_schema.model_fields


class TestApprovalParsing:
    @pytest.mark.parametrize(
        "value", [True, "yes", "Yes", " y ", "approve", "confirm", "ok"]
    )
    def test_approvals(self, value):
        assert is_approval(value) is True

    @pytest.mark.parametrize(
        "value", [False, None, "", "no", "n", "cancel", "stop", "maybe"]
    )
    def test_refusals(self, value):
        assert is_approval(value) is False

    @pytest.mark.parametrize("value", [{"approved": True}, ["yes"], 1, object()])
    def test_an_unrecognised_shape_is_a_refusal(self, value):
        # The gate does not interpret payloads it was not given. A front end
        # that resumes with something structured normalises it to a string or a
        # bool first; anything else leaves the money where it is.
        assert is_approval(value) is False


class TestDescriptions:
    """Each tool describes its own action — the customer is asked about the
    concrete thing being stopped, not about the tool being run."""

    def test_direct_debit_description_names_the_mandate(self):
        text = payment_tools.cancel_direct_debit.func.describe({"merchant": "Netflix"})
        assert "DD-4471" in text and "direct debit" in text.lower()

    def test_standing_order_description_names_the_order(self):
        text = payment_tools.cancel_standing_order.func.describe(
            {"merchant": "James Wilson - Shared Netflix"}
        )
        assert "SO-102" in text and "standing order" in text.lower()

    def test_block_description_carries_the_caveat(self):
        text = card_tools.block_merchant_on_card.func.describe(
            {"card_id": "CARD-5521", "merchant": "FitLife Gym"}
        )
        assert "FitLife Gym" in text and "CARD-5521" in text
        assert "not the contract" in text.lower()

    def test_direct_debit_description_names_the_merchant_behind_the_mandate(self):
        """A reference the customer cannot check makes the gate decorative.

        Asked to cancel Google One — card-on-file, no mandate at all — the agent
        called cancel_direct_debit on DD-4472, which is Namecheap's domain
        renewal. The confirmation said only "Cancel the direct debit DD-4472",
        so the customer approved the wrong subscription being cancelled and was
        told Google One had been dealt with. Naming the merchant is what makes
        that mismatch visible while it can still be refused.
        """
        text = payment_tools.cancel_direct_debit.func.describe(
            {"merchant": "Namecheap Domain Renewal"}
        )
        assert "Namecheap" in text
        assert "£79.99" in text

    def test_direct_debit_description_carries_the_contract_caveat(self):
        """The caveat changes the decision, so it belongs before the answer.

        It used to appear only in the agent's reply once the money had already
        been stopped, which is too late to be a warning.
        """
        text = payment_tools.cancel_direct_debit.func.describe(
            {"merchant": "Netflix"}
        ).lower()
        assert "not the contract" in text
        assert "still owe" in text

    def test_unknown_mandate_is_flagged_rather_than_described(self):
        text = payment_tools.cancel_direct_debit.func.describe(
            {"merchant": "Nonsense Ltd"}
        ).lower()
        assert "no recurring payment" in text

    def test_standing_order_description_names_the_payee(self):
        text = payment_tools.cancel_standing_order.func.describe(
            {"merchant": "James Wilson - Shared Netflix"}
        )
        assert "Shared Netflix" in text
        assert "£25.00" in text

    def test_unknown_standing_order_is_flagged_rather_than_described(self):
        text = payment_tools.cancel_standing_order.func.describe(
            {"merchant": "Nonsense Ltd"}
        ).lower()
        assert "no recurring payment" in text


class TestSuccessFlagParsing:
    """What the audit trail records as the outcome of a write."""

    def test_a_success_payload_is_read_as_success(self):
        assert _succeeded('{"success": true, "mandate_id": "DD-4471"}') is True

    def test_a_failure_payload_is_not(self):
        assert _succeeded('{"success": false, "error": "no such mandate"}') is False

    def test_the_phrase_inside_a_message_does_not_count(self):
        # Only the tool's own top-level verdict is the verdict.
        payload = '{"success": false, "note": "an earlier \\"success\\": true"}'
        assert _succeeded(payload) is False

    @pytest.mark.parametrize("value", ["not json at all", "", "[1, 2]", None])
    def test_anything_unreadable_is_not_a_success(self, value):
        assert _succeeded(value) is False


class TestNoGraphNoWrite:
    def test_a_write_called_outside_a_graph_refuses(self):
        # There is no customer to ask, so there is nobody to authorise it. The
        # answer is a refusal rather than an exception: nothing happened, and
        # the caller is told so in the same shape a declined write returns.
        result = json.loads(
            payment_tools.cancel_standing_order.invoke(
                {
                    "name": "cancel_standing_order",
                    "args": {"merchant": "Landlord - Premier Properties"},
                    "id": "call_outside",
                    "type": "tool_call",
                }
            ).content
        )
        assert result["success"] is False
        assert result["executed"] is False


class TestInterruptBeforeWrite:
    @pytest.fixture
    def started(self, spies):
        llm = _scripted_llm(_route_call(), _cancel_dd_call())
        graph, config, result = _run(llm)
        return graph, config, result, spies

    def test_the_graph_halts(self, started):
        _, _, result, _ = started
        assert "__interrupt__" in result

    def test_the_write_tool_has_not_run(self, started):
        _, _, _, spies = started
        assert spies["cancel_direct_debit"] == []

    def test_the_prompt_says_what_will_happen(self, started):
        _, _, result, _ = started
        payload = result["__interrupt__"][0].value
        assert "DD-4471" in json.dumps(payload)

    def test_the_prompt_lists_the_pending_actions(self, started):
        _, _, result, _ = started
        payload = result["__interrupt__"][0].value
        assert [a["tool"] for a in payload["actions"]] == ["cancel_direct_debit"]


class TestResumeWithApproval:
    @pytest.fixture
    def resumed(self, spies):
        llm = _scripted_llm(
            _route_call(),
            _cancel_dd_call(),
            AIMessage(content="Your Netflix direct debit is cancelled."),
            AIMessage(content="Anything else?"),
        )
        graph, config, _ = _run(llm, "approve-thread")
        final = graph.invoke(Command(resume="yes"), config)
        return final, spies

    def test_the_write_tool_ran(self, resumed):
        _, spies = resumed
        assert len(spies["cancel_direct_debit"]) == 1

    def test_the_turn_completes(self, resumed):
        final, _ = resumed
        assert "__interrupt__" not in final

    def test_the_tool_reported_success(self, resumed):
        final, _ = resumed
        messages = _tool_messages(final["messages"], "cancel_direct_debit")
        assert json.loads(messages[-1].content)["success"] is True


class TestResumeWithRefusal:
    @pytest.fixture
    def refused(self, spies):
        llm = _scripted_llm(
            _route_call(),
            _cancel_dd_call(),
            AIMessage(content="No problem, I have left it in place."),
            AIMessage(content="Anything else?"),
        )
        graph, config, _ = _run(llm, "refuse-thread")
        final = graph.invoke(Command(resume="no"), config)
        return final, spies

    def test_the_write_tool_never_ran(self, refused):
        _, spies = refused
        assert spies["cancel_direct_debit"] == []

    def test_the_pending_call_is_still_answered(self, refused):
        """An unanswered tool call is a provider 400, declined or not."""
        final, _ = refused
        assert _tool_messages(final["messages"], "cancel_direct_debit")

    def test_the_answer_says_the_customer_declined(self, refused):
        final, _ = refused
        payload = json.loads(
            _tool_messages(final["messages"], "cancel_direct_debit")[-1].content
        )
        assert payload["success"] is False
        assert "declin" in json.dumps(payload).lower()

    def test_the_turn_completes(self, refused):
        final, _ = refused
        assert "__interrupt__" not in final


class TestReadToolsAreNotGated:
    def test_detection_runs_without_an_interrupt(self, spies):
        llm = _scripted_llm(
            _route_call(),
            _ai_with_calls(
                _tool_call(
                    "find_recurring_payments",
                    {"user_id": MOCK_USER["user_id"]},
                    "call_find",
                )
            ),
            AIMessage(content="Here is what you pay for."),
            AIMessage(content="Cancel Spotify in your account settings."),
            AIMessage(content="That is everything."),
            AIMessage(content="Anything else?"),
        )
        _, _, result = _run(llm, "read-thread")
        assert "__interrupt__" not in result
        # Answered by the tool node, so the call really executed. (The spy count
        # is not the assertion: the agent's own prompt headline calls detection
        # too, on every node entry.)
        assert _answer_to(result["messages"], "call_find") is not None


class TestMixedCalls:
    """A read and a write on the same AI message: refusing blocks only the write."""

    @pytest.fixture
    def refused_mixed(self, spies):
        llm = _scripted_llm(
            _route_call(),
            _ai_with_calls(
                _tool_call(
                    "find_recurring_payments",
                    {"user_id": MOCK_USER["user_id"]},
                    "call_find",
                ),
                _tool_call(
                    "cancel_direct_debit",
                    {"user_id": MOCK_USER["user_id"], "merchant": "Netflix"},
                    "call_dd",
                ),
            ),
            AIMessage(content="I have left the direct debit alone."),
            AIMessage(content="Cancel Spotify in your account settings."),
            AIMessage(content="That is everything."),
            AIMessage(content="Anything else?"),
        )
        graph, config, first = _run(llm, "mixed-thread")
        final = graph.invoke(Command(resume="no"), config)
        return first, final, spies

    def test_it_still_halted(self, refused_mixed):
        first, _, _ = refused_mixed
        assert "__interrupt__" in first

    def test_only_the_write_was_blocked(self, refused_mixed):
        _, final, spies = refused_mixed
        assert spies["cancel_direct_debit"] == []
        detection = _answer_to(final["messages"], "call_find")
        assert "subscriptions" in json.loads(detection.content)

    def test_every_pending_call_was_answered(self, refused_mixed):
        _, final, _ = refused_mixed
        answered = {
            m.tool_call_id for m in final["messages"] if isinstance(m, ToolMessage)
        }
        assert {"call_find", "call_dd"} <= answered

    def test_the_work_queue_still_filled_from_the_read(self, refused_mixed):
        """The read half of the message must keep driving the cancellation research handoff."""
        _, final, _ = refused_mixed
        assert final["unresolved_card_subs"] == []


class TestBuildGraphSignature:
    def test_checkpointer_is_optional(self):
        llm = _scripted_llm(AIMessage(content="Hi"))
        assert build_graph(llm) is not None

    def test_the_checkpointer_is_used_when_supplied(self):
        llm = _scripted_llm(AIMessage(content="Hi"))
        checkpointer = MemorySaver()
        assert build_graph(llm, checkpointer=checkpointer).checkpointer is checkpointer
