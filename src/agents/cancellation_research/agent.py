"""
Cancellation research agent: provide information on how to cancel a subscription.

Frequently used services have a mock/cached version of how to cancel.
These are processed first.  If no mathc if found, it can search the web for cancelation
information.
"""

from langchain_core.messages import SystemMessage, ToolMessage

from src.agents.cancellation_research.guide import find_cancellation_guide

# One tool. Read-only, and the only door to external guidance.
CANCELLATION_RESEARCH_TOOLS = [find_cancellation_guide]


_SYSTEM_PROMPT = """You are thinkmoney's cancellation researcher. You are not talking to the customer \
about their account — the subscriptions specialist owns that, and control returns there when you are \
done. Your one job is to say how a subscription is cancelled with the merchant itself.

You have exactly one tool, find_cancellation_guide. Call it for every merchant you are asked about, and \
answer only from what it returns.

Where guidance comes from:
- thinkmoney's internal cancellation directory is the primary source. It is checked first on every call, \
it works offline, and each entry records the date we verified it.
- A live web search is a fallback, used by the tool only when the directory has no entry for that \
merchant. You cannot run a search yourself and you should not try.

Always state your source. If the result came from our directory, say so and give its verified_on date — \
"our records, verified on 2026-08-04". If it came from a web search, say the guidance is unverified by \
thinkmoney and the customer should confirm it on the merchant's own site. If the entry is marked \
illustrative, say plainly that it is an example rather than a checked record.

If the tool finds nothing, say what we do not know, how the customer can find it, and what their rights \
are. Never invent steps, URLs, notice periods or fees.

Report the steps in order, and call out the details that cost money if missed — notice periods, refund \
windows, early-termination fees, and what happens to the customer's data.

This customer:
- user_id: {user_id}
- name: {name}

Be brief and specific. Steps, deadlines, source. The specialist will handle anything that touches the \
customer's payments.
"""


def _build_system_prompt(user_info: dict) -> str:
    """Build the cancellation research system prompt for one customer.

    Identifiers come from graph state, never from a literal in this module.
    """
    return _SYSTEM_PROMPT.format(
        user_id=user_info.get("user_id", "unknown"),
        name=user_info.get("name", "the customer"),
    )


def _answer_pending_tool_calls(messages: list) -> list[ToolMessage]:
    """Satisfy the dangling tool call that handed control to this agent.

    Cancellation research is entered by a deterministic handoff rather than by executing the
    call on the last message, so that call reaches the provider unanswered and
    the request is rejected with a 400. Mirrors _handle_unavailable_agent in
    src/graph.py.
    """
    if not messages:
        return []

    tool_calls = getattr(messages[-1], "tool_calls", None)
    if not tool_calls:
        return []

    return [
        ToolMessage(
            content=(
                "Handed off to the cancellation research agent for merchant cancellation "
                "guidance. Answer from thinkmoney's cancellation directory."
            ),
            tool_call_id=tool_call["id"],
        )
        for tool_call in tool_calls
    ]


def _normalised_name(name: str) -> str:
    """Lower-cased, with every run of whitespace collapsed to one space.

    "Apple  iCloud+" and "apple icloud+" are the same merchant written two
    ways, so both have to reduce to the same string before comparing.
    """
    return " ".join((name or "").lower().split())


def merchant_matches(queued: str, requested: str) -> bool:
    """
    Check that what is requested actually matches what was identified.
    """
    queued_name = _normalised_name(queued)
    requested_name = _normalised_name(requested)

    # An empty name would be contained in everything, matching the whole queue.
    if not queued_name or not requested_name:
        return False

    return queued_name in requested_name or requested_name in queued_name


def _merchants_asked_about(tool_calls: list[dict]) -> list[str]:
    """The merchants this pass called find_cancellation_guide for.

    Calls to any other tool are skipped, so `args` reliably carries a merchant
    by the time it is read. A call that somehow arrives without one yields the
    empty string, which matches no queue entry.
    """
    merchants = []

    for tool_call in tool_calls:
        if tool_call.get("name") != "find_cancellation_guide":
            continue

        args = tool_call.get("args") or {}
        merchants.append(args.get("merchant", ""))

    return merchants


def _still_needs_guidance(merchant: str, asked_about: list[str]) -> bool:
    """True if the agent has not just looked this merchant up.

    `asked_about` holds the merchants named in this pass's guide calls. If none
    of them is this merchant, it stays on the queue for the next pass.
    """
    for name in asked_about:
        if merchant_matches(merchant, name):
            return False

    return True


def _drain_queue(queue: list, response) -> list[str]:
    """What is left of the work queue after this pass.

    Two rules, and the second is what guarantees the handoff terminates:

    - Still working (the response carries tool calls): drop only the merchants
      those calls name, so anything else stays queued for the next pass.
    - Finished (no tool calls, control returns to the subscriptions agent):
      drop everything handed over. Cancellation research has said all it is
      going to say this turn; re-queueing would send control straight back here
      forever.
    """
    if not queue:
        return []

    tool_calls = getattr(response, "tool_calls", None) or []
    if not tool_calls:
        return []

    asked_about = _merchants_asked_about(tool_calls)

    remaining = []
    for merchant in queue:
        if _still_needs_guidance(merchant, asked_about):
            remaining.append(merchant)

    return remaining


def create_cancellation_research_node(llm):
    """Create the cancellation research agent node for use in a LangGraph StateGraph.

    Args:
        llm: A LangChain chat model instance.

    Returns:
        A callable node function compatible with StateGraph.add_node().
    """
    llm_with_tools = llm.bind_tools(CANCELLATION_RESEARCH_TOOLS)

    def cancellation_research_node(state):
        user_info = state.get("user_info", {}) or {}

        handoff = _answer_pending_tool_calls(state["messages"])
        system = SystemMessage(content=_build_system_prompt(user_info))

        queue = list(state.get("unresolved_card_subs") or [])

        response = llm_with_tools.invoke([system] + state["messages"] + handoff)

        return {
            "messages": handoff + [response],
            "current_agent": "cancellation_research",
            # Draining here is what terminates the handoff — see _drain_queue.
            "unresolved_card_subs": _drain_queue(queue, response),
        }

    return cancellation_research_node
