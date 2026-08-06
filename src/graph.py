"""Main agent graph definition.

The triage agent is fully wired. Sub-agents need to be added by the candidate.
Study the triage wiring to understand the pattern.
"""

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, ToolMessage

from src.models import AgentState
from src.tools.rails import RAIL_DESCRIPTION, RAIL_TOOL, options_for
from src.agents.triage import create_triage_node, TRIAGE_TOOLS
from src.agents.cancellation_research.agent import (
    create_cancellation_research_node,
    merchant_matches,
    CANCELLATION_RESEARCH_TOOLS,
)
from src.agents.subscriptions.agent import (
    card_subs_from_tool_messages,
    card_subs_needing_research,
    create_subscriptions_node,
    merchants_blocked_in,
    merchants_named_in,
    SUBSCRIPTIONS_TOOLS,
)

# agent_map: maps triage routing names → graph node names.
AGENT_MAP: dict[str, str] = {
    "subscriptions": "subscriptions_agent",
}

# available_agents: tells triage which agents exist and what they do
# (populates the system prompt so triage only routes to real agents).
#
# "cancellation_research" is deliberately absent. It holds no customer data and
# performs no action on the account, so there is nothing for triage to route to
# it directly;
# it is reached only by the deterministic rail-driven handoff from the
# subscriptions agent. Listing it here would let the model reach an agent that
# cannot see the subscription it is being asked about.
AVAILABLE_AGENTS: dict[str, str] = {
    "subscriptions": (
        "Recurring payments and subscriptions — find what the customer is paying for each "
        "month, spot waste (duplicates, unused, price rises, converted trials), and cancel "
        "or block a subscription across direct debit, standing order and card-on-file."
    ),
}


def _make_route_target_fn(agent_map: dict[str, str]):
    """Create a routing function that knows about registered agents.

    Args:
        agent_map: Dict mapping agent names to graph node names.

    Returns:
        A routing function for use with add_conditional_edges.
    """

    def _get_route_target(state: AgentState) -> str:
        """Extract the routing target from triage tool calls.

        Returns node name to route to, or END if no tool calls.
        """
        last_message = state["messages"][-1]

        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return END

        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "route_to_agent":
                agent_name = tool_call["args"].get("agent_name", "")

                if agent_name in agent_map:
                    return agent_map[agent_name]

                # Agent not registered — route to unavailable handler
                return "unavailable_agent"

        # Any other tool call (e.g. search_knowledge_base) goes to triage_tools
        return "triage_tools"

    return _get_route_target


def _handle_unavailable_agent(state: AgentState) -> dict:
    """Handle routing to an agent that hasn't been implemented yet.

    Creates a ToolMessage telling triage the requested agent is not available,
    so triage can inform the customer honestly.
    """
    last_message = state["messages"][-1]

    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "route_to_agent":
            agent_name = tool_call["args"].get("agent_name", "unknown")
            return {
                "messages": [
                    ToolMessage(
                        content=f"Error: The '{agent_name}' agent is not available. "
                        "This specialist capability has not been implemented yet. "
                        "Please let the customer know and offer to help with what you can do "
                        "(knowledge base search and account lookup).",
                        tool_call_id=tool_call["id"],
                    )
                ],
            }

    return {"messages": []}


def _route_from_subagent(state: AgentState) -> str:
    """Route after a sub-agent responds.

    If the sub-agent made tool calls, route to its tool node.
    Otherwise, return to triage.

    This is a helper you can use (or adapt) for your sub-agents.
    """
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        current = state.get("current_agent", "triage")
        return f"{current}_tools"

    return "triage"


def _merchants_already_researched(messages: list) -> list[str]:
    """Merchants a find_cancellation_guide call has already been made for.

    Derived from the message history rather than a second state field, so the
    subscriptions tool node can refuse to re-queue a merchant cancellation research has
    already dealt with — a second detection call mid-turn must not resurrect a
    drained queue entry.
    """
    merchants: list[str] = []

    for message in messages:
        for tool_call in getattr(message, "tool_calls", None) or []:
            if tool_call.get("name") != "find_cancellation_guide":
                continue
            merchant = (tool_call.get("args") or {}).get("merchant")
            if merchant:
                merchants.append(merchant)

    return merchants


def _latest_customer_text(messages: list) -> str:
    """What the customer said most recently, as plain text.

    The queue is scoped against this, so it has to be the customer's own words:
    an agent's summary of the report mentions every merchant in it, which would
    put the data-shaped trigger straight back.
    """
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)

    return ""


# The write tools that name a merchant. Each changes the account, so each is
# worth checking against what the customer actually asked for.
_MERCHANT_WRITE_TOOLS = {
    "cancel_direct_debit",
    "cancel_standing_order",
    "block_merchant_on_card",
}

# Which rail each write acts on. Inverted from RAIL_TOOL rather than restated,
# so a rail and its tool cannot drift apart in two places.
_TOOL_RAIL = {tool: rail for rail, tool in RAIL_TOOL.items()}


# Words customers put in front of a choice. Stripped before looking for the
# number, because "cancel 1" is how people answer a numbered list and refusing
# it sends them round the question again.
_CHOICE_PREFIXES = (
    "cancel",
    "block",
    "stop",
    "number",
    "option",
    "the",
    "please",
)


def _choice_from_offer(text: str, offered: list) -> list[str]:
    """The subscription an answer to "which one?" picks out.

    Customers answer a numbered list with a number. Nothing in the account data
    says what "2" means, so the list that produced the question is kept in state
    and the answer is resolved against it. Without this the reply identifies no
    merchant, the write is refused for having no referent, and the same question
    is asked again — a customer asking to cancel Netflix could not get there.

    What must remain true is that the answer *is* a choice and nothing else.
    Leading words like "cancel" or "option" are stripped, so "cancel 1" selects;
    but anything left over means it was not a selection, so "cancel 2 of these"
    and "£2.99" still select nothing rather than acting on a subscription nobody
    pointed at.
    """
    if not offered:
        return []

    words = (text or "").lower().replace(".", " ").replace(",", " ").split()
    while words and words[0] in _CHOICE_PREFIXES:
        words.pop(0)

    if len(words) != 1 or not words[0].isdigit():
        return []

    index = int(words[0])
    if 1 <= index <= len(offered):
        return [offered[index - 1]]

    return []


def _offer_for(proposed_merchant: str, rail: str) -> tuple[str, list[str]]:
    """The choice to put to the customer: numbered text, and the same order.

    Numbered because the answer has to be bindable. The list is returned
    alongside the text so state holds exactly what was offered — the numbering
    the customer replies to and the numbering the reply resolves against are
    then the same list, not two that happen to agree.

    Amounts lose the statement's minus sign and card rows have no cadence to
    quote, so both are normalised here: this is read to someone choosing between
    them, not dumped from the fixture.
    """
    lines = []
    merchants = []

    for position, row in enumerate(options_for(proposed_merchant, rail), start=1):
        detail = row["amount"].lstrip("-")
        if row["frequency"]:
            detail = f"{detail} {row['frequency']}"
        # The rail matters here: two Netflix entries differ only by how
        # they bill, and the customer is choosing between them.
        detail = f"{detail}, {RAIL_DESCRIPTION[row['rail']]}"
        lines.append(f"{position}. {row['merchant']} ({detail})")
        merchants.append(row["merchant"])

    return "; ".join(lines), merchants


def _writes_without_a_confirmed_referent(
    messages: list, offered: list | None = None
) -> tuple[dict[str, str], list[str]]:
    """Pending writes not bound to a subscription the customer identified.

    Returns `({tool_call_id: refusal}, merchants_offered_this_time)`. The second
    value is what the customer is now being asked to choose between; it goes
    into state so their answer can be resolved against the same list that
    produced the question.

    `resolve_rail` makes the mechanism and
    the reference correct for whatever merchant is named; it cannot know whether
    that is the merchant the customer meant. Which subscription someone means by
    "cancel it" is not in the data — it is in their head — so the only honest
    source for it is the customer.

    Hence the rule: a write proceeds when the customer named the subscription,
    and otherwise the turn goes back to them with the list. Two things are
    refused, for different reasons:

    * **Contradiction** — they named a subscription and the call names a
      different one. Acting would cancel the wrong thing.
    * **No referent** — they named none, so nothing binds this call to anything.
      This used to be allowed, which made the whole check fail open: "cancel it"
      let the model pick freely and a wrong pick went through silently.

    Asking is not a degraded outcome here. It is the correct one: the bank
    cannot know which subscription was meant, and the customer can answer in a
    word.
    """
    customer_text = _latest_customer_text(messages)
    named = merchants_named_in(customer_text) or _choice_from_offer(
        customer_text, offered or []
    )

    last_message = messages[-1] if messages else None
    refusals: dict[str, str] = {}
    now_offered: list[str] = []

    for tool_call in getattr(last_message, "tool_calls", None) or []:
        name = tool_call.get("name")
        if name not in _MERCHANT_WRITE_TOOLS:
            continue

        merchant = (tool_call.get("args") or {}).get("merchant") or ""

        if not named:
            options, merchants = _offer_for(merchant, _TOOL_RAIL[name])
            now_offered = merchants
            refusals[tool_call["id"]] = (
                "Refused, and nothing was changed: the customer has not said "
                "which subscription they mean, so this call is not bound to "
                "anything they asked for. Put this list to them exactly as "
                f"written, keeping the numbers: {options}. They can reply with "
                "a number or a name. Do not renumber it, do not add or remove "
                "entries, and do not choose for them."
            )
        elif not any(merchant_matches(merchant, expected) for expected in named):
            refusals[tool_call["id"]] = (
                f"Refused, and nothing was changed: this call names "
                f"{merchant or '(no merchant)'}, but the customer asked about "
                f"{' or '.join(named)}. Act on what they asked for, or ask them "
                "which one they mean — do not proceed on a different "
                "subscription."
            )

    return refusals, now_offered


def _make_subscriptions_tool_node():
    """The subscriptions tool node, plus the work queue.

    Running the tools is ToolNode's job. The confirmation gate is not here: it
    rides on the write tools themselves (`src/confirmation.py:1`), so a refused
    call answers itself and this node needs no knowledge of which tools move
    money. What is left is the delegation trigger — still no model decision
    anywhere in it, but scoped to the merchants this turn is actually about.

    The rail label alone used to be the trigger, which made the handoff fire on
    the shape of the report rather than on the request: holding any card-billed
    subscription sent every one of them to cancellation research, so "what am I
    paying for?" came back with unasked-for cancellation steps. A merchant now
    reaches the queue only when the customer named it or a block is being placed
    on it.
    """
    tool_node = ToolNode(SUBSCRIPTIONS_TOOLS)

    def _run_tools(state: AgentState, skip: dict[str, str]) -> list:
        """Run the pending tool calls, answering any refused ones without running.

        A refused call still needs a ToolMessage: an unanswered tool_call_id is
        a provider 400 on the next turn, the same reason the handoff stubs one
        in. So the refusals are answered here and only the rest reach ToolNode.
        """
        messages = state["messages"]
        last_message = messages[-1]

        answered = [
            ToolMessage(
                content=refusal,
                tool_call_id=call_id,
                name=next(
                    tc["name"] for tc in last_message.tool_calls if tc["id"] == call_id
                ),
            )
            for call_id, refusal in skip.items()
        ]

        allowed = [tc for tc in last_message.tool_calls if tc["id"] not in skip]
        if not allowed:
            return answered

        trimmed = last_message.model_copy(update={"tool_calls": allowed})
        result = tool_node.invoke(
            {**state, "messages": list(messages[:-1]) + [trimmed]}
        )
        ran = result["messages"] if isinstance(result, dict) else result

        return answered + list(ran)

    def subscriptions_tool_node(state: AgentState) -> dict:
        refused, now_offered = _writes_without_a_confirmed_referent(
            state["messages"], state.get("offered_subscriptions")
        )

        if refused:
            new_messages = _run_tools(state, refused)
        else:
            result = tool_node.invoke(state)
            new_messages = result["messages"] if isinstance(result, dict) else result

        queue = list(state.get("unresolved_card_subs") or [])
        already_researched = _merchants_already_researched(state["messages"])

        candidates = card_subs_from_tool_messages(new_messages)
        wanted = card_subs_needing_research(
            candidates,
            _latest_customer_text(state["messages"]),
            merchants_blocked_in(state["messages"]),
        )

        for merchant in wanted:
            if merchant in queue:
                continue
            if any(merchant_matches(merchant, done) for done in already_researched):
                continue
            queue.append(merchant)

        return {
            "messages": new_messages,
            "unresolved_card_subs": queue,
            # Carried forward while a choice is outstanding, cleared once one is
            # made: a stale list would let a later "2" pick from a question the
            # customer was never asked.
            "offered_subscriptions": now_offered,
        }

    return subscriptions_tool_node


def _route_from_subscriptions(state: AgentState) -> str:
    """Route after the subscriptions agent responds.

    Its own tool work comes first; then any card-on-file subscription still
    needing guidance goes to cancellation research; then control returns to triage. The
    middle branch is the deterministic handoff — it depends only on the rail
    label the detection tool reported, never on the model choosing to delegate.
    """
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "subscriptions_tools"

    if state.get("unresolved_card_subs"):
        return "cancellation_research_agent"

    return "triage"


def _route_from_cancellation_research(state: AgentState) -> str:
    """Route after the cancellation research agent responds.

    Cancellation research either calls its one tool or hands control back to the
    specialist — never to triage. It holds no account data, so it cannot close
    out a turn.
    """
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "cancellation_research_tools"

    return "subscriptions_agent"


def build_graph(llm, checkpointer=None):
    """Build and compile the thinkmoney customer service agent graph.

    Args:
        llm: A LangChain chat model instance.
        checkpointer: Optional checkpointer. Required for the confirmation gate
            to be answerable — `interrupt()` needs somewhere to save the halted
            turn — but left optional so callers that never reach a write tool
            (and the graph-shape tests) can build the graph without one.

    Returns:
        A compiled LangGraph that can be invoked with AgentState.
    """
    agent_map = dict(AGENT_MAP)
    available_agents = dict(AVAILABLE_AGENTS)

    graph = StateGraph(AgentState)

    # --- Triage agent (PROVIDED) ---
    triage_node = create_triage_node(llm, available_agents=available_agents or None)
    graph.add_node("triage", triage_node)

    # --- Triage tool execution ---
    triage_tool_node = ToolNode(TRIAGE_TOOLS)
    graph.add_node("triage_tools", triage_tool_node)

    # --- Unavailable agent handler ---
    graph.add_node("unavailable_agent", _handle_unavailable_agent)

    # --- Subscriptions agent (specialist: the customer's recurring payments) ---
    graph.add_node("subscriptions_agent", create_subscriptions_node(llm))
    graph.add_node("subscriptions_tools", _make_subscriptions_tool_node())

    # --- Cancellation research agent (specialist: how to cancel with the merchant) ---
    graph.add_node(
        "cancellation_research_agent", create_cancellation_research_node(llm)
    )
    graph.add_node("cancellation_research_tools", ToolNode(CANCELLATION_RESEARCH_TOOLS))

    # --- Entry point ---
    graph.set_entry_point("triage")

    # --- Triage routing ---
    _get_route_target = _make_route_target_fn(agent_map)
    graph.add_conditional_edges(
        "triage",
        _get_route_target,
        # Spelled out rather than left implicit so the drawn graph shows exactly
        # which nodes triage can reach — cancellation_research_agent is not
        # among them.
        list(agent_map.values()) + ["triage_tools", "unavailable_agent", END],
    )
    graph.add_edge("triage_tools", "triage")
    graph.add_edge("unavailable_agent", "triage")

    # --- Sub-agent routing ---
    # Subscriptions loops through its own tools, then hands off to cancellation
    # research for as long as the work queue holds a card-on-file merchant, then
    # returns to triage. Cancellation research loops through its one tool and
    # always comes back to the specialist, draining what it handled so the
    # handoff terminates.
    graph.add_conditional_edges(
        "subscriptions_agent",
        _route_from_subscriptions,
        ["subscriptions_tools", "cancellation_research_agent", "triage"],
    )
    graph.add_edge("subscriptions_tools", "subscriptions_agent")

    graph.add_conditional_edges(
        "cancellation_research_agent",
        _route_from_cancellation_research,
        ["cancellation_research_tools", "subscriptions_agent"],
    )
    graph.add_edge("cancellation_research_tools", "cancellation_research_agent")

    return graph.compile(checkpointer=checkpointer)
