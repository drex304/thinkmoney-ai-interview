"""Tests for the money-movement audit trail.

The bug these exist for: the agent asked "shall I proceed?" in prose instead of
calling the write tool, the customer said yes, and nothing ran — while the agent
reported the action as done. There was no record of any of it.

Two defences are asserted here. The trail records every stage of a write, so a
request with no execution is visible after the fact. And a write request must
produce an actual tool execution, so the prose-confirmation failure fails a test
rather than a customer.
"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src import audit
from src.graph import build_graph
from src.models import MOCK_USER

BLOCK_CALL = {
    "name": "block_merchant_on_card",
    "args": {
        "user_id": MOCK_USER["user_id"],
        "card_id": "CARD-5521",
        "merchant": "Apple iCloud+",
    },
    "id": "call_block_1",
}


@pytest.fixture(autouse=True)
def audit_file(tmp_path, monkeypatch):
    """Point the trail at a temp file so tests never touch the repo's log."""
    path = tmp_path / "audit.log"
    monkeypatch.setenv("THINKMONEY_AUDIT_LOG", str(path))
    return path


def _events(path):
    return [entry["event"] for entry in audit.read_entries(path)]


class TestRecordFormat:
    def test_each_record_is_one_json_line(self, audit_file):
        audit.record("first", detail="a")
        audit.record("second", detail="b")
        lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert [json.loads(line)["event"] for line in lines] == ["first", "second"]

    def test_every_record_is_timestamped(self, audit_file):
        entry = audit.record("something")
        assert entry["timestamp"].endswith("+00:00")

    def test_the_trail_is_append_only(self, audit_file):
        audit.record("first")
        audit.record("second")
        audit.record("third")
        assert _events(audit_file) == ["first", "second", "third"]

    def test_a_truncated_final_line_does_not_hide_the_rest(self, audit_file):
        audit.record("first")
        with audit_file.open("a", encoding="utf-8") as handle:
            handle.write('{"event": "cut off mid-writ')
        assert _events(audit_file) == ["first"]

    def test_an_unwritable_path_never_breaks_the_caller(self, monkeypatch, tmp_path):
        # A full disk must not be the reason a cancellation fails.
        blocked = tmp_path / "not-a-dir"
        blocked.write_text("", encoding="utf-8")
        monkeypatch.setenv("THINKMONEY_AUDIT_LOG", str(blocked / "audit.log"))
        assert audit.record("still returns")["event"] == "still returns"


class TestWriteLifecycle:
    def test_a_request_is_recorded_before_anything_executes(self, audit_file):
        audit.write_requested(BLOCK_CALL)
        entry = audit.read_entries(audit_file)[0]
        assert entry["event"] == "write_requested"
        assert entry["tool"] == "block_merchant_on_card"
        assert entry["arguments"]["merchant"] == "Apple iCloud+"

    def test_the_answer_that_was_read_as_approval_is_recorded(self, audit_file):
        audit.write_approved(BLOCK_CALL, "yes")
        entry = audit.read_entries(audit_file)[0]
        assert entry["event"] == "write_approved"
        assert entry["answer"] == "yes"

    def test_a_refusal_is_recorded_with_what_the_customer_said(self, audit_file):
        audit.write_refused(BLOCK_CALL, "not right now")
        entry = audit.read_entries(audit_file)[0]
        assert entry["event"] == "write_refused"
        assert entry["answer"] == "not right now"

    def test_execution_records_the_tools_own_success_flag(self, audit_file):
        audit.write_executed(BLOCK_CALL, succeeded=False, detail='{"success": false}')
        entry = audit.read_entries(audit_file)[0]
        assert entry["succeeded"] is False


def _graph_with_pending_block():
    """A graph driven through triage into a pending block, with a checkpointer.

    The turn has to travel the real path — triage routes, the subscriptions
    agent emits the write, the tool node gates it — because the bug being
    guarded against lives in that sequence, not in any one node.
    """

    class ScriptedLLM:
        def __init__(self):
            self.turns = 0

        def bind_tools(self, tools, **kwargs):
            return self

        def invoke(self, messages):
            self.turns += 1
            if self.turns == 1:  # triage
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "route_to_agent",
                            "args": {
                                "agent_name": "subscriptions",
                                "reason": "customer wants to block a merchant",
                            },
                            "id": "call_route_1",
                        }
                    ],
                )
            if self.turns == 2:  # subscriptions agent: calls the write tool
                return AIMessage(content="", tool_calls=[BLOCK_CALL])
            return AIMessage(content="That merchant is blocked.")

    return build_graph(ScriptedLLM(), checkpointer=MemorySaver())


class TestWriteRequestProducesExecution:
    """The regression test for the bug.

    A customer asking for a block must result in the write tool actually
    running once approved — not in a reassuring sentence and no execution.
    """

    def _run(self, answer):
        graph = _graph_with_pending_block()
        config = {"configurable": {"thread_id": f"audit-{answer}"}}
        state = {
            "messages": [HumanMessage(content="block Apple iCloud+ please")],
            "user_info": MOCK_USER,
            "current_agent": "triage",
            "unresolved_card_subs": [],
        }
        graph.invoke(state, config)
        graph.invoke(Command(resume=answer), config)

    def test_approval_executes_the_write(self, audit_file):
        self._run("yes")
        events = _events(audit_file)
        assert "write_requested" in events
        assert "write_approved" in events
        assert "write_executed" in events, (
            "approved write never executed — the agent may be asking for "
            "confirmation in prose instead of calling the tool"
        )

    def test_every_approved_request_has_a_matching_execution(self, audit_file):
        # Deduped on tool_call_id: interrupt() replays the node, so the halted
        # pass and the resumed pass each log the request. The invariant that
        # matters is that nothing was requested and left unexecuted.
        self._run("yes")
        entries = audit.read_entries(audit_file)
        requested = {
            e["tool_call_id"] for e in entries if e["event"] == "write_requested"
        }
        executed = {
            e["tool_call_id"] for e in entries if e["event"] == "write_executed"
        }
        assert requested == executed
        assert requested == {BLOCK_CALL["id"]}

    def test_the_execution_records_the_success_the_tool_reported(self, audit_file):
        # Not just that the write ran: a trail that logs every execution as
        # failed is as useless as no trail, and nothing else asserts the flag.
        self._run("yes")
        executed = [
            e for e in audit.read_entries(audit_file) if e["event"] == "write_executed"
        ]
        assert [e["succeeded"] for e in executed] == [True]

    def test_a_replayed_request_does_not_execute_the_write_twice(self, audit_file):
        # The node runs twice; the money must move once.
        self._run("yes")
        entries = audit.read_entries(audit_file)
        executed = [e for e in entries if e["event"] == "write_executed"]
        assert len(executed) == 1

    def test_refusal_records_the_refusal_and_executes_nothing(self, audit_file):
        self._run("no")
        events = _events(audit_file)
        assert "write_refused" in events
        assert "write_executed" not in events

    def test_the_trail_shows_the_full_sequence_in_order(self, audit_file):
        self._run("yes")
        events = _events(audit_file)
        assert events.index("write_requested") < events.index("write_approved")
        assert events.index("write_approved") < events.index("write_executed")


class TestPromptForbidsProseConfirmation:
    """The prompt must not reintroduce the failure.

    Instructing the model to ask before calling guarantees it never calls, and
    the gate it was meant to reach never fires.
    """

    def _prompt(self) -> str:
        from src.agents.subscriptions.agent import _SYSTEM_PROMPT

        return _SYSTEM_PROMPT.lower()

    def test_it_tells_the_agent_to_call_the_tool_rather_than_ask(self):
        """The phrase is now 'do not ask again', not 'do not ask first'.

        The instruction is now conditional on the customer having asked for the
        change, because the unconditional form made a provider open a
        cancellation off a read-only question. What must survive is the half
        this class guards: once asked, call the tool instead of asking in prose.
        """
        prompt = self._prompt()
        assert "do not ask again" in prompt
        assert "once they have asked, call the write tool" in prompt

    def test_it_explains_that_the_gate_is_the_confirmation(self):
        assert "that gate is the confirmation" in self._prompt()

    def test_it_warns_that_asking_in_prose_loses_the_action(self):
        assert "silently never happens" in self._prompt()
