"""Shared state definitions and mock data for the thinkmoney customer service agent."""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State schema shared across all agent nodes in the graph.

    Attributes:
        messages: Conversation message history (auto-appended via add_messages reducer).
        current_agent: Name of the currently active agent node.
        user_info: Mock customer context (user_id, name, account_id, etc.).
        unresolved_card_subs: WORK QUEUE — the card-on-file subscriptions
            still needing guidance on how to cancel with the merchant. It is
            deliberately not an inventory of the card subscriptions that
            exist. The router re-reads it every
            time control leaves the subscriptions agent, so the cancellation
            research node must remove the entries it handled as part of its
            state update. Left undrained, the router sends control back to
            cancellation research forever and the turn aborts at LangGraph's
            recursion limit of 25.
        offered_subscriptions: the merchants, in order, last offered to the
            customer when a write named none they had asked for. Held so the
            answer can be bound to the question: told "1. Netflix 2. Shared
            Netflix", customers reply "2", and without the list that answer
            identifies nothing and the same question gets asked again forever.
    """

    messages: Annotated[list, add_messages]
    current_agent: str
    user_info: dict
    unresolved_card_subs: list
    offered_subscriptions: list


# Mock customer identity used throughout the exercise.
# All mock tools reference this same customer for consistency.
MOCK_USER = {
    "user_id": "USR-2847",
    "name": "Sarah Johnson",
    "email": "sarah.j@email.com",
    "phone": "+44 7700 900123",
    "account_id": "ACC-9182",
    "primary_card_id": "CARD-5521",
    "address": "42 Kings Road, London, E1 7AZ",
    "account_type": "Premium",
    "member_since": "2023-03-15",
}
