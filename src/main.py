"""CLI entry point for the thinkmoney customer service agent."""

import argparse
import os
import sys
import uuid

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel

from src.config import DEFAULT_MODELS, REQUIRED_ENV_VARS, get_llm
from src.graph import build_graph
from src.models import MOCK_USER
from src.agents.cancellation_research import web_search


def _log_stream_event(console: Console, node_name: str, state_update: dict):
    """Print real-time visibility into graph execution.

    Logs agent activations, tool calls, and routing decisions as they happen.
    Works automatically for any agent or tool — including ones the candidate adds.
    """
    # Log agent/node activation
    if node_name.endswith("_tools"):
        pass  # Tool execution nodes are logged via tool calls on the agent message
    elif node_name not in ("__start__", "__end__"):
        console.print(f"  [dim]> Agent: {node_name}[/]")

    # Inspect messages in the state update for tool calls and routing
    messages = state_update.get("messages", [])
    if not isinstance(messages, list):
        return

    for msg in messages:
        # Log tool calls made by an agent
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name", "unknown")
                args = tc.get("args", {})

                # Special handling for route_to_agent
                if name == "route_to_agent":
                    target = args.get("agent_name", "?")
                    reason = args.get("reason", "")
                    console.print(f"  [dim]> Routing: → {target} ({reason})[/]")
                else:
                    # Format tool args concisely
                    args_parts = []
                    for k, v in args.items():
                        val = str(v)
                        if len(val) > 40:
                            val = val[:37] + "..."
                        args_parts.append(f'{k}="{val}"')
                    args_str = ", ".join(args_parts)
                    console.print(f"  [dim]> Tool: {name}({args_str})[/]")

        # Log tool results (abbreviated)
        if isinstance(msg, ToolMessage):
            content = str(msg.content) if msg.content else ""
            if len(content) > 80:
                content = content[:77] + "..."
            console.print(f"  [dim]> Result: {content}[/]")


def _thread_config(thread_id: str | None = None) -> dict:
    """The graph config for one conversation.

    Every invocation needs it: the checkpointer keys the saved conversation —
    and the halted turn waiting on a confirmation — by thread id.
    """
    return {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}


def _interrupt_payload(state_snapshot):
    """The confirmation request from a halted snapshot, or None.

    With `stream_mode="values"` a halt arrives as an ordinary snapshot carrying
    an extra `__interrupt__` key, not as an exception.
    """
    if not isinstance(state_snapshot, dict):
        return None

    interrupts = state_snapshot.get("__interrupt__") or []
    if not interrupts:
        return None

    return interrupts[0].value


def _confirmation_text(payload) -> str:
    """What the customer reads before answering."""
    if not isinstance(payload, dict):
        return str(payload)

    text = str(payload.get("message", "Please confirm before I continue."))
    for action in payload.get("actions") or []:
        description = str(action.get("description", "")).strip()
        if description and description not in text:
            text += f"\n- {description}"
    return text


def _ask_confirmation(console: Console, payload) -> str:
    """Show the pending action and collect the answer to resume with.

    An abandoned prompt is a refusal — walking away from the question must
    never be read as approval.
    """
    console.print(f"\n[bold yellow]Confirm:[/] {_confirmation_text(payload)}")
    try:
        return console.input("[bold green]You:[/] ").strip()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]No answer given — leaving everything as it is.[/]")
        return "no"


def _checkpointed_message_count(graph, config) -> int:
    """How many messages the checkpointer already holds for this thread."""
    try:
        snapshot = graph.get_state(config)
    except Exception:  # no checkpointer, or nothing saved yet
        return 0
    return len((getattr(snapshot, "values", None) or {}).get("messages", []))


def _stream_segment(graph, console: Console, config: dict, graph_input, logged: int):
    """Stream until the graph finishes or halts for a confirmation.

    Returns the last snapshot seen, the confirmation payload (None if the turn
    ran to completion) and the updated count of messages already logged.
    """
    final_state = None
    payload = None

    with console.status("[bold blue]Thinking...", spinner="dots"):
        for state_snapshot in graph.stream(graph_input, config, stream_mode="values"):
            payload = _interrupt_payload(state_snapshot)

            all_messages = state_snapshot.get("messages", [])
            new_messages = all_messages[logged:]
            logged = max(logged, len(all_messages))

            current_agent = state_snapshot.get("current_agent", "")

            # Build a pseudo state_update for our logger
            if new_messages:
                node_name = current_agent or "unknown"
                # Detect if these are tool result messages (from a _tools node)
                if isinstance(new_messages[0], ToolMessage):
                    node_name = f"{current_agent}_tools"

                _log_stream_event(console, node_name, {"messages": new_messages})

            final_state = state_snapshot

            if payload is not None:
                break

    return final_state, payload, logged


def run_turn(graph, console: Console, config: dict, user_input: str):
    """Run one customer turn, pausing for confirmation as often as needed.

    Only the new HumanMessage is sent: the checkpointer owns the conversation,
    so re-passing accumulated history would duplicate every earlier message.
    """
    graph_input = {
        "messages": [HumanMessage(content=user_input)],
        "user_info": MOCK_USER,
        "current_agent": "triage",
    }
    # The customer's own message needs no echoing back at them.
    logged = _checkpointed_message_count(graph, config) + 1
    final_state = None

    while True:
        final_state, payload, logged = _stream_segment(
            graph, console, config, graph_input, logged
        )
        if payload is None:
            return final_state

        graph_input = Command(resume=_ask_confirmation(console, payload))


def _print_reply(console: Console, final_state, since: int = 0):
    """Print everything the agents said to the customer this turn, in order.

    Printing only the last message showed the wrong one. A subscriptions turn
    ends back at triage, so the final message is triage's closer — while the
    answer it is closing on came from the specialist, whose prompt carries the
    figures computed in `_headline`. The closer is written by a model that was
    never given those figures, so showing it alone put a paraphrase where the
    arithmetic should be, and a turn whose closer was a bare "anything else?"
    reached the customer with the answer missing entirely.

    Printing all of them had its own failure: triage frequently closes by
    restating what the specialist just said, so the customer read the same
    subscription list, or the same question about which one they meant, twice
    in a row. Both are correct messages from different nodes; only one of them
    is worth showing. So each distinct thing said is printed once, and a repeat
    of something already on screen this turn is dropped.

    Args:
        console: Where to print.
        final_state: The state returned by `run_turn`.
        since: Index of the first message belonging to this turn. Earlier
            messages are replies the customer has already been shown, and the
            checkpointer keeps them all, so printing from 0 would repeat the
            whole conversation on every turn.
    """
    messages = (final_state.get("messages", []) if final_state else [])[since:]

    spoken = [
        msg
        for msg in messages
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls
    ]

    # A turn can end without a clean message — content alongside a tool call,
    # for instance. Something was still said, so show the last of it rather
    # than nothing.
    if not spoken:
        spoken = [
            msg
            for msg in reversed(messages)
            if isinstance(msg, AIMessage) and msg.content
        ][:1]

    already_said: set[str] = set()

    for msg in spoken:
        text = msg.content if isinstance(msg.content, str) else str(msg.content)

        # Compared on collapsed whitespace and case: one node's restatement of
        # another's answer is rarely byte-identical, and a difference in
        # wrapping is not a difference in what the customer reads.
        key = " ".join(text.split()).lower()
        if key in already_said:
            continue

        already_said.add(key)
        console.print(f"\n[bold blue]thinkmoney:[/] {text}")


def main():
    parser = argparse.ArgumentParser(description="thinkmoney AI Customer Service Agent")
    parser.add_argument(
        "--provider",
        required=True,
        choices=["ollama", "openai", "anthropic"],
        help="LLM provider to use",
    )
    _defaults = ", ".join(f"{p}={m}" for p, m in DEFAULT_MODELS.items())
    parser.add_argument(
        "--model",
        default=None,
        help=f"Model name override (defaults: {_defaults})",
    )
    args = parser.parse_args()

    console = Console()

    # Checked before the graph is built so a missing key costs one line at
    # startup, not a traceback from inside the first turn. The constructors are
    # happy without a credential; only the request fails, and by then the user
    # has already typed a prompt and the failure surfaces deep in the runtime.
    required_key = REQUIRED_ENV_VARS.get(args.provider)
    if required_key and not os.environ.get(required_key):
        console.print(
            f"[bold red]Error:[/] the {args.provider} provider needs "
            f"[bold]{required_key}[/] and it is not set.\n\n"
            f"  export {required_key}=...\n\n"
            "[dim]Or run against a local model with no key: "
            "uv run thinkmoney --provider ollama[/]"
        )
        sys.exit(1)

    try:
        llm = get_llm(args.provider, args.model)
    except (ImportError, ValueError) as e:
        console.print(f"[bold red]Error:[/] {e}")
        sys.exit(1)

    # The cancellation-guidance fallback uses the same provider's hosted search,
    # so it follows this flag rather than pinning one vendor. Ollama has no
    # hosted search; on it the tool answers from the directory alone and says so.
    web_search.configure(args.provider)

    # The checkpointer is what makes the confirmation gate answerable: it holds
    # the halted turn between `interrupt()` and the customer's reply, and owns
    # the conversation history from here on.
    graph = build_graph(llm, checkpointer=MemorySaver())
    config = _thread_config()

    console.print(
        Panel.fit(
            "[bold]thinkmoney[/] AI Customer Service\n"
            f"Provider: [cyan]{args.provider}[/] | "
            f"Model: [cyan]{args.model or DEFAULT_MODELS[args.provider]}[/]\n"
            f"Customer: [green]{MOCK_USER['name']}[/] ({MOCK_USER['user_id']})\n\n"
            "Type [bold]quit[/] to exit.",
            title="Welcome",
            border_style="blue",
        )
    )

    while True:
        try:
            user_input = console.input("\n[bold green]You:[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/]")
            break

        # Where this turn's messages start, so the reply printer shows what was
        # said now rather than replaying the whole checkpointed conversation.
        turn_start = _checkpointed_message_count(graph, config)

        # Streaming gives real-time visibility into agent activity, and is where
        # a confirmation halt surfaces; run_turn resumes the turn once answered.
        final_state = run_turn(graph, console, config, user_input)
        _print_reply(console, final_state, since=turn_start)


if __name__ == "__main__":
    main()
