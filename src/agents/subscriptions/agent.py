"""Subscriptions agent — the specialist for recurring payments.

Detection and savings arithmetic and every write are done by the tools not by the LLM.
This module decides what the agent is allowed to say and which subscriptions it
may say it about. The headline figures are calculated here, before the LLM runs.
"""

import json

from langchain_core.messages import SystemMessage, ToolMessage

from src.agents.cancellation_research.agent import merchant_matches
from src.agents.cancellation_research.directory import CANCELLATION_DIRECTORY
from src.agents.triage import search_knowledge_base
from src.tools.cards import list_cards
from src.tools.payments import cancel_standing_order, get_payees
from src.agents.subscriptions.detection import find_recurring_payments
from src.tools.cards import block_merchant_on_card
from src.tools.payments import cancel_direct_debit
from src.tools.rails import all_rails
from src.tools.transactions import get_transaction_history

SUBSCRIPTIONS_TOOLS = [
    find_recurring_payments,
    get_transaction_history,
    cancel_standing_order,
    cancel_direct_debit,
    block_merchant_on_card,
    list_cards,
    get_payees,
    search_knowledge_base,
]


# The rail label find_recurring_payments uses for a subscription billed against
# a card. It is the only rail the bank cannot cancel on the customer's behalf,
# which is why it — and nothing else — feeds the cancellation research work
# queue.
CARD_RAIL = "card_on_file"


def card_subs_from_tool_messages(messages: list) -> list[str]:
    """Read the card-on-file merchants out of find_recurring_payments results.

    The rail label is the only thing consulted, so this is plain parsing: no
    judgement, same answer on every provider. These are *candidates* for
    cancellation research, not the queue itself — every card subscription the
    customer holds appears here whether or not they asked about any of them.
    `card_subs_needing_research` decides which of them are actually wanted.
    """
    merchants: list[str] = []

    for message in messages:
        if getattr(message, "name", None) != "find_recurring_payments":
            continue
        try:
            report = json.loads(message.content)
        except (TypeError, ValueError):
            # A tool that errored returns prose, not JSON. Nothing to queue.
            continue

        for subscription in report.get("subscriptions", []):
            merchant = subscription.get("merchant")
            if (
                subscription.get("rail") == CARD_RAIL
                and merchant
                and merchant not in merchants
            ):
                merchants.append(merchant)

    return merchants


def merchants_named_in(text: str) -> list[str]:
    """Subscriptions the customer's own words identify, by name or alias.

    Used to catch a write aimed at the wrong merchant. Deliberately answers
    "which did they name", not "did they name this one": a request naming no
    merchant at all ("cancel it", "cancel my gym membership") returns an empty
    list and blocks nothing. Only a customer who named something specific can
    contradict the model.

    Aliases come from the cancellation directory, so "fitlife" and "fit life
    gym" read as the same subscription rather than as a different one.
    """
    haystack = " ".join((text or "").lower().split())
    if not haystack:
        return []

    named: list[str] = []

    for row in all_rails():
        merchant = row["merchant"]
        if merchant in named:
            continue

        candidates = [merchant.lower()]
        for entry in CANCELLATION_DIRECTORY.values():
            if merchant_matches(merchant, entry["merchant"]):
                candidates.extend(alias.lower() for alias in entry.get("aliases", []))

        if any(candidate and candidate in haystack for candidate in candidates):
            named.append(merchant)

    return named


def merchants_blocked_in(messages: list) -> list[str]:
    """Merchants a block_merchant_on_card call names in these messages.

    A block is the point at which cancellation guidance stops being optional:
    the charge stops but the contract does not, so the customer needs the
    merchant's own cancellation steps alongside it.
    """
    merchants: list[str] = []

    for message in messages:
        for tool_call in getattr(message, "tool_calls", None) or []:
            if tool_call.get("name") != "block_merchant_on_card":
                continue
            merchant = (tool_call.get("args") or {}).get("merchant")
            if merchant and merchant not in merchants:
                merchants.append(merchant)

    return merchants


def card_subs_needing_research(
    candidates: list[str], customer_text: str, blocked: list[str]
) -> list[str]:
    """Narrow every card-on-file subscription down to the ones actually wanted.

    The handoff used to fire on the shape of the report: hold a card-billed
    subscription and cancellation research ran, even for "what am I paying
    for?". The customer got cancellation steps for a gym they had only asked
    the price of, and every card merchant was looked up to produce them.

    The trigger is the request, not the data. A merchant earns research when
    the customer names it, or when a block is being placed on it — both read
    off the turn rather than guessed at, so the handoff stays deterministic and
    a read-only question stays read-only.
    """
    wanted: list[str] = []

    for merchant in candidates:
        named_by_customer = merchant_matches(merchant, customer_text)
        being_blocked = any(merchant_matches(merchant, name) for name in blocked)
        if (named_by_customer or being_blocked) and merchant not in wanted:
            wanted.append(merchant)

    return wanted


def _headline(user_id: str) -> dict:
    """The figures the agent opens with, taken straight from the detection tool.

    Computing them here rather than letting the model read them out of a tool
    result means the first sentence of the conversation cannot be wrong.
    """
    report = json.loads(find_recurring_payments.invoke({"user_id": user_id}))
    totals = report["totals"]
    savings = report["savings"]

    return {
        "subscription_count": totals["subscription_count"],
        "discretionary_count": totals["discretionary_count"],
        "monthly_total": totals["monthly_total"],
        "annualised_total": totals["annualised_total"],
        "identified_saving": savings["identified_saving"],
        "potential_saving": savings["potential_saving"],
        "strategy_count": savings["strategy_count"],
    }


_SYSTEM_PROMPT = """You are thinkmoney's subscriptions specialist. {first_name} has been routed to \
you because they want to understand or act on their recurring payments.

Open with the headline, in your own words but with these exact figures: {first_name} is committed to \
£{annualised_total} a year across {discretionary_count} discretionary subscriptions (£{monthly_total} a \
month on the monthly ones), and across {strategy_count} changes you can identify £{identified_saving} a \
year from their account alone, plus a further £{potential_saving} a year that depends on a question only \
they can answer. Keep those two savings figures separate — never add them together and never present the \
potential figure as money already found.

The split is not presentation, it is honesty about what thinkmoney can see. The bank observes payments \
leaving the account: amounts, dates, cadence, rail. It cannot observe whether a service is being used — \
gym visits, streams, logins and syncs are all held by the merchant, not by us. So for any strategy \
carrying confirmation_required, state the cost as fact, say plainly you have no way of knowing whether \
they still use it, and ask. Never tell a customer they have not used something.

Every figure you quote must come from a tool result. Call find_recurring_payments to get the full \
breakdown — it returns each subscription with its cadence, rail and annualised cost, plus a savings block \
listing every change worth making, largest first. Never calculate a total yourself.

This customer:
- user_id: {user_id} — pass this to every tool that takes one
- primary card: {primary_card_id}
{account_line}
Use these values exactly as given. Do not invent or assume an identifier; if a tool needs one you have \
not been given, ask.

How to act:
- **A question is not an instruction.** When {first_name} asks what they pay for, where they could save, \
what something costs or how a process works, answer it in words. Do not call a write tool. The savings \
block find_recurring_payments returns is a report to relay and offer — it is not a list of work to carry \
out, and its wording describes what the customer could choose to do, not what you should go and do. \
Cancelling or blocking something nobody asked you to cancel is a serious error even though the \
confirmation gate would catch it: it puts a change to their account in front of them that they never \
requested, instead of the answer they did ask for.
- **Once they have asked, call the write tool. Do not ask again.** cancel_standing_order, \
cancel_direct_debit and block_merchant_on_card all move money, and every one of them is gated by the \
system: when you call it, the turn halts and the customer is asked to confirm before anything executes. \
That gate is the confirmation. Asking "shall I proceed?" in your reply instead of calling the tool does \
not pause anything — it just ends your turn, and the customer's "yes" never reaches you, so the action \
they asked for silently never happens. So when the customer has asked for a specific change, call the \
tool on that same turn and let the gate do the asking. Call one write tool at a time so each is \
confirmed separately.
- **Name the merchant, not a reference.** Every write tool takes the merchant the customer asked about \
and looks the rail and reference up itself, so pass the name as it appears in the subscription list and \
leave mandate_id, order_id and card_id alone unless you are deliberately asserting which one you expect. \
If you pass a reference that does not belong to that merchant, the tool refuses and nothing happens. A \
saving often names two subscriptions — the one to drop and the one being kept — so take the merchant from \
the strategy you are acting on, not from the sentence around it.
- **A refusal is information, not an obstacle.** If a write tool comes back with success false, it is \
telling you the remedy does not match the rail — a card-billed subscription has no mandate to cancel, for \
instance. Read what it says, tell the customer, and use the tool it names. Do not retry the same call with \
a different identifier.
- **Never choose which subscription they meant.** A write only goes through when {first_name} has said \
which one. If they have not — "cancel it", "stop the expensive one", or a name too garbled to match — the \
write is refused and comes back with a numbered list. Put that list to them exactly as it was given to \
you, with the same numbers and the same entries: they can answer with a number, and the number is matched \
against the list you were handed, not against one you composed. Renumbering it, reordering it, merging it \
with the savings report or adding a subscription of your own makes their "2" select something they did \
not point at. Do not pick the likeliest for them. Which subscription somebody meant is the one thing here \
you cannot look up.
- **Blocking is not cancelling.** Blocking a merchant on a card stops the charge, but it does not cancel \
the contract — the customer may still owe the merchant, and can still be chased for late fees or sent to \
debt collection. Say this every time you propose or perform a block, and recommend cancelling with the \
merchant as well.
- **Do not touch essential payments.** Rent, utilities and insurance are detected and reported but are \
excluded from the savings figures. Never propose cancelling them as part of a cost-cutting sweep, and if \
the customer asks to cancel one, check they mean it before acting.
- **Match the remedy to the rail.** Direct debits are cancelled with cancel_direct_debit using the \
mandate_id, standing orders with cancel_standing_order using the order_id, and card-on-file subscriptions \
can only be blocked with block_merchant_on_card — the customer has to cancel those with the merchant.
- **Ground policy in the knowledge base.** Before stating anything about thinkmoney's own rules — whether \
there is a fee for cancelling a direct debit or standing order, what we do and do not support, what a \
process involves — call search_knowledge_base and answer from what it returns. Do not state a fee or a \
policy from memory. If the knowledge base does not cover it, say so rather than guessing.
- Use list_cards to name a card by its last four digits rather than reading out a card ID, and get_payees \
to put a name to a standing order's payee.

Be direct and concrete. The customer is here to spend less, so quantify everything in pounds per year and \
say which single change saves the most.
"""


def _build_system_prompt(user_info: dict, headline: dict) -> str:
    """Build the subscriptions system prompt for one customer.

    Every identifier comes from `user_info` (i.e. from graph state); nothing
    about the mock customer is baked into the prompt text.
    """
    name = user_info.get("name") or "Customer"
    first_name = name.split()[0]
    account_id = user_info.get("account_id")

    return _SYSTEM_PROMPT.format(
        first_name=first_name,
        user_id=user_info.get("user_id", "unknown — ask the customer"),
        primary_card_id=user_info.get("primary_card_id", "unknown — call list_cards"),
        account_line=f"- account_id: {account_id}\n" if account_id else "",
        **headline,
    )


def _answer_pending_tool_calls(messages: list) -> list[ToolMessage]:
    """Satisfy the dangling tool call that handed control to this agent.

    Triage's route_to_agent call is never executed — the router jumps straight
    here — so the history reaches the provider with an unanswered tool call and
    is rejected with a 400. Mirrors _handle_unavailable_agent in src/graph.py.
    """
    if not messages:
        return []

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    if not tool_calls:
        return []

    answers = []
    for tool_call in tool_calls:
        if tool_call["name"] == "route_to_agent":
            content = (
                "Handed off to the subscriptions agent. You now have the customer's "
                "recurring payments, cancellation and card-blocking tools. Answer the "
                "customer directly."
            )
        else:
            # Not expected — the router only sends route_to_agent calls here —
            # but an unanswered call of any name invalidates the history.
            content = (
                f"The '{tool_call['name']}' call was not executed: the conversation "
                "was handed to the subscriptions agent instead."
            )
        answers.append(ToolMessage(content=content, tool_call_id=tool_call["id"]))

    return answers


def create_subscriptions_node(llm):
    """Create the subscriptions agent node for use in a LangGraph StateGraph.

    Args:
        llm: A LangChain chat model instance.

    Returns:
        A callable node function compatible with StateGraph.add_node().
    """
    llm_with_tools = llm.bind_tools(SUBSCRIPTIONS_TOOLS)

    def subscriptions_node(state):
        user_info = state.get("user_info", {}) or {}
        user_id = user_info.get("user_id", "")

        handoff = _answer_pending_tool_calls(state["messages"])
        system = SystemMessage(
            content=_build_system_prompt(user_info, _headline(user_id))
        )

        response = llm_with_tools.invoke([system] + state["messages"] + handoff)

        return {
            "messages": handoff + [response],
            "current_agent": "subscriptions",
        }

    return subscriptions_node
