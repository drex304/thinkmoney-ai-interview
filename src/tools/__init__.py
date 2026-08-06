"""Mock tools for thinkmoney customer service agents.

Tools are grouped by domain for easy binding to sub-agents.
Import the group constant (e.g. ACCOUNT_TOOLS) and pass it to llm.bind_tools().
"""

from src.tools.account import (
    get_account_details,
    get_account_limits,
    update_contact_info,
    get_account_status,
)
from src.tools.cards import (
    list_cards,
    freeze_card,
    unfreeze_card,
    order_replacement_card,
    get_card_status,
    block_merchant_on_card,
)
from src.tools.transactions import (
    get_transaction_history,
    get_transaction_details,
    dispute_transaction,
    get_dispute_status,
)
from src.tools.payments import (
    get_payees,
    initiate_transfer,
    get_exchange_rate,
    get_standing_orders,
    cancel_standing_order,
    cancel_direct_debit,
)
from src.tools.kyc import (
    get_kyc_status,
    get_required_documents,
    submit_document,
    get_account_restrictions,
)

# Grouped tool lists for binding to sub-agents
ACCOUNT_TOOLS = [
    get_account_details,
    get_account_limits,
    update_contact_info,
    get_account_status,
]
CARD_TOOLS = [
    list_cards,
    freeze_card,
    unfreeze_card,
    order_replacement_card,
    get_card_status,
    block_merchant_on_card,
]
TRANSACTION_TOOLS = [
    get_transaction_history,
    get_transaction_details,
    dispute_transaction,
    get_dispute_status,
]
PAYMENT_TOOLS = [
    get_payees,
    initiate_transfer,
    get_exchange_rate,
    get_standing_orders,
    cancel_standing_order,
    cancel_direct_debit,
]
KYC_TOOLS = [
    get_kyc_status,
    get_required_documents,
    submit_document,
    get_account_restrictions,
]

ALL_TOOLS = ACCOUNT_TOOLS + CARD_TOOLS + TRANSACTION_TOOLS + PAYMENT_TOOLS + KYC_TOOLS
