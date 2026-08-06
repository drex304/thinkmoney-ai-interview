"""The subscriptions agent, and the detection it is built on.

Everything here is specific to this agent: the node, the deterministic
recurring-payment detection, and the plan/billing reference data. Bank records
it reads — transactions, mandates, cards — stay in `src/tools/`, because they
belong to the bank rather than to any one agent.
"""

from src.agents.subscriptions.agent import (
    CARD_RAIL,
    SUBSCRIPTIONS_TOOLS,
    card_subs_from_tool_messages,
    create_subscriptions_node,
)
from src.agents.subscriptions.detection import find_recurring_payments

__all__ = [
    "CARD_RAIL",
    "SUBSCRIPTIONS_TOOLS",
    "card_subs_from_tool_messages",
    "create_subscriptions_node",
    "find_recurring_payments",
]
