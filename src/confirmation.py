"""The confirmation gate for tools that move money.

A tool wrapped in `@requires_confirmation` halts the graph before it runs and
executes only on an explicit yes. The gate is on the tool rather than in the
system prompt because a prompt instruction is a request the model can skip; this
is a stop it cannot route around.

Why on the tool rather than in the tool node: it is the pattern LangGraph
documents, and it makes a refusal self-answering. The tool returns an ordinary
result either way, so there is never an unanswered `tool_call_id` for the graph
to patch up — which is a provider 400 — and no message surgery anywhere.

The decorator is also the registration. `CONFIRMED_TOOLS` collects every name
that carries it, so a test can assert that the set of gated tools matches the
set of tools that move money. Adding a write tool without the decorator is a
failing test rather than a silent hole.

One constraint worth knowing before changing this. LangGraph re-runs the whole
node on resume, so everything before `interrupt()` happens twice: the request is
audited twice and the description is built twice, both harmless. Resume values
are matched to interrupt calls strictly by index, so a tool must issue exactly
one interrupt per call — never a loop, never a conditional second one.
"""

from __future__ import annotations

import functools
import inspect
import json
from typing import Callable

from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt

from src import audit

# Every tool carrying the decorator, filled at import time.
CONFIRMED_TOOLS: set[str] = set()

# Answers read as approval. Anything else — including silence, an empty string
# and anything unrecognised — is a refusal, so the failure mode of a garbled
# answer is "nothing happened", not "the money moved".
_APPROVALS = frozenset(
    {
        "y",
        "yes",
        "yeah",
        "yep",
        "ok",
        "okay",
        "approve",
        "approved",
        "confirm",
        "confirmed",
        "go ahead",
        "proceed",
        "do it",
        "true",
    }
)


def is_approval(answer) -> bool:
    """Did the customer approve the pending write?

    The answer is whatever the caller resumed with: a string from the CLI, or a
    bool from a programmatic caller. Anything else — including an unrecognised
    shape — is a refusal, so a front end that sends something unexpected leaves
    the money where it is rather than guessing. Normalise at the boundary; this
    is the last check before a write runs and is not the place to interpret.
    """
    if isinstance(answer, bool):
        return answer

    if isinstance(answer, str):
        return answer.strip().lower() in _APPROVALS

    return False


def _ask_human(payload: dict):
    """Halt the graph and return the customer's answer.

    Its own function so tests exercising a tool's body can replace it; outside a
    graph `interrupt()` raises, and the caller below reads that as a refusal.
    """
    return interrupt(payload)


def _request(tool_call: dict, description: str) -> dict:
    """The payload the customer is shown at the halt."""
    return {
        "type": "confirmation_required",
        "message": (
            "Before I change anything on your account, please confirm:\n"
            f"- {description}\n\n"
            "Reply 'yes' to go ahead, or anything else to leave it as it is."
        ),
        "actions": [
            {
                "tool": tool_call["name"],
                "args": tool_call["args"],
                "description": description,
            }
        ],
    }


def _refused(description: str) -> str:
    """The tool's own answer when the customer says no.

    A refusal is a result, not an error: the call is answered, the model is told
    plainly that nothing happened, and the turn carries on.
    """
    refusal = {
        "success": False,
        "executed": False,
        "error": (
            "The customer declined this action, so it was not carried out. "
            "Confirm nothing has changed and ask what they would like to do instead."
        ),
        "declined_action": description,
    }
    return json.dumps(refusal, indent=2)


def _succeeded(result: str) -> bool:
    """Detects if a tool reported success, read from its own payload.

    Tools return a JSON string, so this reads the `success` flag rather than
    matching text: a nested object, or a message that quotes the phrase, must
    not be mistaken for the tool's own verdict.

    A tool could return something that is not JSON. If we cannot read a success
    we assume failure — the trail would rather record a write it could not
    classify than raise a decode error on a turn where the money already moved.
    """
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return False

    if not isinstance(payload, dict):
        return False

    return payload.get("success") is True


def requires_confirmation(describe: Callable[[dict], str]):
    """
    Halt for an explicit yes before this tool runs, so a human can make the decision.
    (human in the loop)

    Args:
        describe: Builds the plain-English line the customer is shown, from the
            tool's arguments. It names the concrete thing being stopped rather
            than the tool being run, because that is what the customer is being
            asked about.

    The wrapped tool should declare `tool_call_id: Annotated[str,
    InjectedToolCallId]`. LangChain fills it in and hides it from the model, and
    the audit trail is keyed by it — without it the trail still records the
    write, but requests and executions can no longer be paired by id.
    """

    def decorator(fn):
        CONFIRMED_TOOLS.add(fn.__name__)
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def confirmed(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            tool_call = {
                "name": fn.__name__,
                "args": {k: v for k, v in arguments.items() if k != "tool_call_id"},
                "id": arguments.get("tool_call_id", ""),
            }
            description = describe(tool_call["args"])

            # log before halt so we always get a trace.
            audit.write_requested(tool_call)
            request = _request(tool_call, description)
            try:
                answer = _ask_human(request)
            except GraphBubbleUp:
                # This is the halt
                raise
            except Exception:
                # Called with no graph behind it so no customer was asked.
                answer = None

            approved = is_approval(answer)
            (audit.write_approved if approved else audit.write_refused)(
                tool_call, answer
            )
            if not approved:
                return _refused(description)

            # Called through __wrapped__ so we can monkeypath this in tests
            result = confirmed.__wrapped__(*args, **kwargs)
            audit.write_executed(
                tool_call, succeeded=_succeeded(result), detail=str(result)
            )
            return result

        confirmed.describe = describe
        return confirmed

    return decorator
