"""Tests for the CLI's checkpointer wiring and confirmation handling.

The confirmation gate (`interrupt()` inside the subscriptions tool node) is only
half a feature until someone can answer it. These tests cover the other half:
the CLI holds a checkpointer, passes a thread id on every call, notices the halt
in the stream, asks the customer, and resumes the paused turn with their answer.

They also pin the consequence of owning conversation state in the checkpointer —
each turn sends ONLY the new HumanMessage. Re-sending the accumulated history
would duplicate every message the checkpointer already holds.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from src import main as cli
from src.graph import build_graph
from src.models import MOCK_USER
from src.tools import payments as payment_tools
from src.tools import cards as card_tools


class FakeConsole:
    """A console that records what was printed and replays scripted input."""

    def __init__(self, answers: list[str] | None = None):
        self.answers = list(answers or [])
        self.printed: list[str] = []

    def print(self, *args, **kwargs):
        self.printed.append(" ".join(str(a) for a in args))

    def input(self, prompt: str = "") -> str:
        self.printed.append(str(prompt))
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)

    @contextmanager
    def _status(self, *args, **kwargs):
        yield self

    def status(self, *args, **kwargs):
        return self._status(*args, **kwargs)

    @property
    def output(self) -> str:
        return "\n".join(self.printed)


def _scripted_llm(*responses):
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.invoke = MagicMock(side_effect=list(responses))
    return llm


def _route_to_subscriptions(call_id: str = "call_route"):
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


def _cancel_netflix(call_id: str = "call_dd"):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "cancel_direct_debit",
                "args": {"user_id": MOCK_USER["user_id"], "merchant": "Netflix"},
                "id": call_id,
            }
        ],
    )


def _cancellation_turn_llm():
    """A scripted turn that reaches the write tool and then wraps up."""
    return _scripted_llm(
        _route_to_subscriptions(),
        _cancel_netflix(),
        AIMessage(content="That is done — nothing further will be collected."),
        AIMessage(content="Anything else I can help with?"),
    )


@pytest.fixture
def write_spy(monkeypatch):
    """Record write-tool executions so 'it did not run' is assertable."""
    calls: dict[str, list] = {
        "cancel_direct_debit": [],
        "cancel_standing_order": [],
        "block_merchant_on_card": [],
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
    wrap(payment_tools.cancel_standing_order, "cancel_standing_order")
    return calls


def _checkpointed_graph(llm):
    return build_graph(llm, checkpointer=MemorySaver())


class TestInterruptDetection:
    """Reading the halt out of a `stream_mode='values'` snapshot."""

    def test_payload_returned_when_present(self):
        interrupt = MagicMock()
        interrupt.value = {"type": "confirmation_required", "message": "Confirm?"}
        snapshot = {"messages": [], "__interrupt__": [interrupt]}

        assert cli._interrupt_payload(snapshot) == interrupt.value

    def test_none_when_no_interrupt(self):
        assert cli._interrupt_payload({"messages": []}) is None

    def test_none_when_interrupt_list_empty(self):
        assert cli._interrupt_payload({"messages": [], "__interrupt__": []}) is None

    def test_none_for_non_dict_snapshot(self):
        assert cli._interrupt_payload(["not", "a", "snapshot"]) is None

    def test_plain_string_payload_survives(self):
        interrupt = MagicMock()
        interrupt.value = "Confirm?"
        assert cli._interrupt_payload({"__interrupt__": [interrupt]}) == "Confirm?"


class TestConfirmationText:
    """What the customer is shown before they answer."""

    def _payload(self):
        return {
            "type": "confirmation_required",
            "message": "Before I change anything, please confirm:",
            "actions": [
                {
                    "tool": "cancel_direct_debit",
                    "args": {"user_id": MOCK_USER["user_id"], "merchant": "Netflix"},
                    "description": "Cancel the Netflix direct debit DD-4471.",
                }
            ],
        }

    def test_includes_the_message(self):
        text = cli._confirmation_text(self._payload())
        assert "Before I change anything" in text

    def test_includes_each_action_description(self):
        text = cli._confirmation_text(self._payload())
        assert "Cancel the Netflix direct debit DD-4471." in text

    def test_action_not_repeated_when_message_already_lists_it(self):
        payload = self._payload()
        payload["message"] += "\n- Cancel the Netflix direct debit DD-4471."
        text = cli._confirmation_text(payload)
        assert text.count("Cancel the Netflix direct debit DD-4471.") == 1

    def test_string_payload_used_verbatim(self):
        assert cli._confirmation_text("Confirm the cancellation?") == (
            "Confirm the cancellation?"
        )


class TestAskConfirmation:
    """Prompting for the answer that is handed back to `Command(resume=...)`."""

    def test_returns_what_the_customer_typed(self):
        console = FakeConsole(["yes"])
        assert cli._ask_confirmation(console, {"message": "Confirm?"}) == "yes"

    def test_shows_the_confirmation_text(self):
        console = FakeConsole(["yes"])
        cli._ask_confirmation(
            console,
            {"message": "Confirm?", "actions": [{"description": "Cancel DD-4471."}]},
        )
        assert "Cancel DD-4471." in console.output

    def test_refusal_is_passed_through_unchanged(self):
        console = FakeConsole(["no thanks"])
        assert cli._ask_confirmation(console, {"message": "Confirm?"}) == "no thanks"

    def test_abandoned_prompt_is_a_refusal(self):
        """Ctrl-D at the confirmation must not be read as approval."""
        console = FakeConsole([])  # raises EOFError
        assert cli._ask_confirmation(console, {"message": "Confirm?"}) == "no"


class TestRunTurnConfirmation:
    """A full turn through the real graph, answered from the CLI."""

    def _turn(self, answers):
        console = FakeConsole(answers)
        graph = _checkpointed_graph(_cancellation_turn_llm())
        config = cli._thread_config("cli-test")
        state = cli.run_turn(graph, console, config, "Cancel my Netflix direct debit.")
        return console, state

    def test_approval_executes_the_write(self, write_spy):
        _, state = self._turn(["yes"])
        assert len(write_spy["cancel_direct_debit"]) == 1
        assert "__interrupt__" not in state

    def test_refusal_executes_nothing(self, write_spy):
        _, state = self._turn(["no"])
        assert write_spy["cancel_direct_debit"] == []

    def test_refusal_still_answers_the_tool_call(self, write_spy):
        _, state = self._turn(["no"])
        answers = [
            m
            for m in state["messages"]
            if isinstance(m, ToolMessage) and m.tool_call_id == "call_dd"
        ]
        assert len(answers) == 1
        assert "declined" in answers[0].content.lower()

    def test_customer_is_shown_what_will_happen(self, write_spy):
        console, _ = self._turn(["yes"])
        assert "DD-4471" in console.output

    def test_turn_completes_after_resuming(self, write_spy):
        _, state = self._turn(["yes"])
        assert isinstance(state["messages"][-1], AIMessage)
        assert state["messages"][-1].content


class TestRunTurnHistory:
    """The checkpointer owns the history, so the CLI must not re-send it."""

    def test_only_the_new_human_message_is_sent(self):
        graph = MagicMock()
        graph.get_state.return_value = MagicMock(
            values={"messages": [HumanMessage(content="earlier turn")]}
        )
        graph.stream.return_value = iter([{"messages": [], "current_agent": "triage"}])

        cli.run_turn(graph, FakeConsole(), cli._thread_config("t"), "and now this")

        sent = graph.stream.call_args.args[0]
        assert [m.content for m in sent["messages"]] == ["and now this"]

    def test_thread_config_passed_on_every_invocation(self):
        graph = MagicMock()
        graph.get_state.return_value = MagicMock(values={"messages": []})
        graph.stream.return_value = iter([{"messages": []}])
        config = cli._thread_config("thread-42")

        cli.run_turn(graph, FakeConsole(), config, "hello")

        assert graph.stream.call_args.args[1] == config
        assert config["configurable"]["thread_id"] == "thread-42"

    def test_history_is_not_duplicated_across_turns(self):
        llm = _scripted_llm(
            AIMessage(content="Hello, how can I help?"),
            AIMessage(content="Still here."),
        )
        graph = _checkpointed_graph(llm)
        console = FakeConsole()
        config = cli._thread_config("cli-history")

        cli.run_turn(graph, console, config, "hello")
        state = cli.run_turn(graph, console, config, "hello again")

        humans = [m.content for m in state["messages"] if isinstance(m, HumanMessage)]
        assert humans == ["hello", "hello again"]

    def test_resume_does_not_relog_earlier_messages(self, write_spy):
        """Messages logged before the halt must not be printed twice on resume."""
        console = FakeConsole(["yes"])
        graph = _checkpointed_graph(_cancellation_turn_llm())
        cli.run_turn(graph, console, cli._thread_config("cli-relog"), "Cancel Netflix.")
        assert console.output.count("Routing:") == 1


class TestThreadConfig:
    def test_generates_a_thread_id_when_none_given(self):
        config = cli._thread_config()
        assert config["configurable"]["thread_id"]

    def test_generated_thread_ids_are_distinct(self):
        first = cli._thread_config()["configurable"]["thread_id"]
        second = cli._thread_config()["configurable"]["thread_id"]
        assert first != second


class TestMainWiring:
    """`main()` builds a checkpointed graph and drives it on one thread."""

    def _run_main(self, inputs):
        console = FakeConsole(inputs)
        graph = MagicMock()
        graph.get_state.return_value = MagicMock(values={"messages": []})
        graph.stream.side_effect = lambda *a, **k: iter([{"messages": []}])

        with (
            patch.object(cli, "Console", return_value=console),
            patch.object(cli, "get_llm", return_value=MagicMock()),
            patch.object(cli, "build_graph", return_value=graph) as build,
            patch("sys.argv", ["src", "--provider", "ollama"]),
        ):
            cli.main()

        return build, graph, console

    def test_checkpointer_passed_to_build_graph(self):
        build, _, _ = self._run_main(["quit"])
        checkpointer = build.call_args.kwargs.get("checkpointer")
        assert isinstance(checkpointer, MemorySaver)

    def test_every_turn_carries_the_same_thread_id(self):
        _, graph, _ = self._run_main(["first", "second", "quit"])
        thread_ids = {
            call.args[1]["configurable"]["thread_id"]
            for call in graph.stream.call_args_list
        }
        assert len(thread_ids) == 1

    def test_turns_send_only_their_own_message(self):
        _, graph, _ = self._run_main(["first", "second", "quit"])
        sent = [
            [m.content for m in call.args[0]["messages"]]
            for call in graph.stream.call_args_list
        ]
        assert sent == [["first"], ["second"]]


class TestPrintReply:
    """What of a turn's messages actually reaches the customer."""

    def _replies(self, *contents):
        console = FakeConsole()
        state = {"messages": [AIMessage(content=c) for c in contents]}
        cli._print_reply(console, state)
        return console.printed

    def test_each_distinct_message_is_shown(self):
        replies = self._replies("Here is your list.", "Anything else?")
        assert len(replies) == 2

    def test_a_repeated_message_is_shown_once(self):
        """Triage closes by restating the specialist, so the customer read the
        same subscription list twice in a row."""
        replies = self._replies("Which one did you mean?", "Which one did you mean?")
        assert len(replies) == 1
        assert "Which one did you mean?" in replies[0]

    def test_a_restatement_differing_only_in_wrapping_is_still_a_repeat(self):
        replies = self._replies("Which one did you mean?", "which one\ndid   you mean?")
        assert len(replies) == 1

    def test_a_genuinely_different_closer_still_appears(self):
        replies = self._replies("Here is your list.", "Anything else I can help with?")
        assert len(replies) == 2

    def test_the_first_wording_is_the_one_kept(self):
        """The specialist speaks before triage, and it holds the real figures."""
        replies = self._replies("You pay £118.95 a month.", "you pay £118.95 A MONTH.")
        assert "£118.95 a month." in replies[0]

    def test_a_turn_that_said_nothing_prints_nothing(self):
        assert self._replies() == []

    def test_messages_before_this_turn_are_not_reprinted(self):
        console = FakeConsole()
        state = {
            "messages": [
                AIMessage(content="Old answer."),
                AIMessage(content="New answer."),
            ]
        }
        cli._print_reply(console, state, since=1)
        assert len(console.printed) == 1
        assert "New answer." in console.printed[0]
