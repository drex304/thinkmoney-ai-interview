"""Mock transaction query tools for thinkmoney customer service."""

import json
from datetime import date, datetime, time, timedelta

from langchain_core.tools import tool

# The corpus is dated relative to today so the demo never goes stale: recurring
# patterns stay inside the get_transaction_history window and "renews in 11 days"
# stays true whenever the exercise is opened.
_TODAY = date.today()


def _days_ago(days: int, hour: int = 9, minute: int = 0) -> str:
    """ISO-8601 timestamp for a point `days` days before today."""
    stamp = datetime.combine(_TODAY, time(hour, minute)) - timedelta(days=days)
    return stamp.isoformat() + "Z"


def _txn(
    transaction_id: str,
    days: int,
    description: str,
    category: str,
    amount: str,
    *,
    rail: str,
    txn_type: str,
    card: str | None = None,
    **rail_reference: str,
) -> dict:
    """Build a transaction record.

    `rail` is how the money leaves the account and therefore which remedy applies:
    direct_debit, standing_order and card_on_file are the three subscription rails.
    """
    return {
        "transaction_id": transaction_id,
        "date": _days_ago(days),
        "description": description,
        "category": category,
        "amount": amount,
        "currency": "GBP",
        "status": "completed",
        "card_used": card,
        "type": txn_type,
        "rail": rail,
        **rail_reference,
    }


def _card(transaction_id, days, description, category, amount, card):
    """A one-off card purchase — a negative case for subscription detection."""
    return _txn(
        transaction_id,
        days,
        description,
        category,
        amount,
        rail="card_payment",
        txn_type="card_payment",
        card=card,
    )


def _subscription_card(transaction_id, days, description, category, amount, card):
    """A recurring charge billed to a card on file."""
    return _txn(
        transaction_id,
        days,
        description,
        category,
        amount,
        rail="card_on_file",
        txn_type="card_payment",
        card=card,
    )


MOCK_TRANSACTIONS = [
    # --- Netflix: direct debit, monthly. Duplicated by SO-102 below. ---
    _txn(
        "TXN-90101",
        65,
        "Netflix",
        "Entertainment",
        "-£15.99",
        rail="direct_debit",
        txn_type="direct_debit",
        mandate_id="DD-4471",
    ),
    _txn(
        "TXN-90102",
        35,
        "Netflix",
        "Entertainment",
        "-£15.99",
        rail="direct_debit",
        txn_type="direct_debit",
        mandate_id="DD-4471",
    ),
    _txn(
        "TXN-90103",
        5,
        "Netflix",
        "Entertainment",
        "-£15.99",
        rail="direct_debit",
        txn_type="direct_debit",
        mandate_id="DD-4471",
    ),
    # --- Shared Netflix: standing order to a friend — the other half of the duplicate. ---
    _txn(
        "TXN-90111",
        65,
        "James Wilson - Shared Netflix",
        "Entertainment",
        "-£25.00",
        rail="standing_order",
        txn_type="standing_order",
        order_id="SO-102",
    ),
    _txn(
        "TXN-90112",
        35,
        "James Wilson - Shared Netflix",
        "Entertainment",
        "-£25.00",
        rail="standing_order",
        txn_type="standing_order",
        order_id="SO-102",
    ),
    _txn(
        "TXN-90113",
        5,
        "James Wilson - Shared Netflix",
        "Entertainment",
        "-£25.00",
        rail="standing_order",
        txn_type="standing_order",
        order_id="SO-102",
    ),
    # --- Spotify: card on file, price rose from £11.99 to £12.99. ---
    _subscription_card(
        "TXN-90121", 92, "Spotify", "Entertainment", "-£11.99", "CARD-8834"
    ),
    _subscription_card(
        "TXN-90122", 62, "Spotify", "Entertainment", "-£11.99", "CARD-8834"
    ),
    _subscription_card(
        "TXN-90123", 32, "Spotify", "Entertainment", "-£12.99", "CARD-8834"
    ),
    _subscription_card(
        "TXN-90124", 2, "Spotify", "Entertainment", "-£12.99", "CARD-8834"
    ),
    # --- Adobe: free trial converted to a paid plan. ---
    _subscription_card(
        "TXN-90131", 70, "Adobe Creative Cloud", "Software", "-£0.00", "CARD-8834"
    ),
    _subscription_card(
        "TXN-90132", 40, "Adobe Creative Cloud", "Software", "-£9.99", "CARD-8834"
    ),
    _subscription_card(
        "TXN-90133", 10, "Adobe Creative Cloud", "Software", "-£9.99", "CARD-8834"
    ),
    # --- FitLife Gym: card on file, and the largest discretionary commitment.
    # Whether it is still being used is something the bank cannot see, so the
    # usage_unknown strategy asks rather than assumes. ---
    _subscription_card(
        "TXN-90141", 63, "FitLife Gym", "Fitness", "-£44.00", "CARD-5521"
    ),
    _subscription_card(
        "TXN-90142", 33, "FitLife Gym", "Fitness", "-£44.00", "CARD-5521"
    ),
    _subscription_card(
        "TXN-90143", 3, "FitLife Gym", "Fitness", "-£44.00", "CARD-5521"
    ),
    # --- Apple iCloud+ and Google One: overlapping cloud storage on different cards. ---
    _subscription_card(
        "TXN-90151", 66, "Apple iCloud+", "Cloud Storage", "-£2.99", "CARD-5521"
    ),
    _subscription_card(
        "TXN-90152", 36, "Apple iCloud+", "Cloud Storage", "-£2.99", "CARD-5521"
    ),
    _subscription_card(
        "TXN-90153", 6, "Apple iCloud+", "Cloud Storage", "-£2.99", "CARD-5521"
    ),
    _subscription_card(
        "TXN-90161", 74, "Google One", "Cloud Storage", "-£7.99", "CARD-8834"
    ),
    _subscription_card(
        "TXN-90162", 44, "Google One", "Cloud Storage", "-£7.99", "CARD-8834"
    ),
    _subscription_card(
        "TXN-90163", 14, "Google One", "Cloud Storage", "-£7.99", "CARD-8834"
    ),
    # --- Domain renewal: annual direct debit, next one lands in ~11 days. ---
    _txn(
        "TXN-90171",
        719,
        "Namecheap Domain Renewal",
        "Software",
        "-£79.99",
        rail="direct_debit",
        txn_type="direct_debit",
        mandate_id="DD-4472",
    ),
    _txn(
        "TXN-90172",
        354,
        "Namecheap Domain Renewal",
        "Software",
        "-£79.99",
        rail="direct_debit",
        txn_type="direct_debit",
        mandate_id="DD-4472",
    ),
    # --- Rent: correctly detected, but must never be offered for cancellation. ---
    _txn(
        "TXN-90181",
        64,
        "Landlord - Premier Properties",
        "Housing",
        "-£1,200.00",
        rail="standing_order",
        txn_type="standing_order",
        order_id="SO-101",
    ),
    _txn(
        "TXN-90182",
        34,
        "Landlord - Premier Properties",
        "Housing",
        "-£1,200.00",
        rail="standing_order",
        txn_type="standing_order",
        order_id="SO-101",
    ),
    _txn(
        "TXN-90183",
        4,
        "Landlord - Premier Properties",
        "Housing",
        "-£1,200.00",
        rail="standing_order",
        txn_type="standing_order",
        order_id="SO-101",
    ),
    # --- One-offs carried over from the original fixture: negative cases. ---
    _card("TXN-90001", 3, "Tesco Express", "Groceries", "-£12.45", "CARD-5521"),
    _card("TXN-89992", 4, "TfL - Contactless", "Transport", "-£4.80", "CARD-5521"),
    _card("TXN-89985", 6, "Amazon.co.uk", "Shopping", "-£67.99", "CARD-8834"),
    _txn(
        "TXN-89960",
        7,
        "Acme Corp - Salary",
        "Income",
        "+£3,250.00",
        rail="bank_transfer",
        txn_type="bank_transfer",
    ),
    _card("TXN-89945", 8, "Pret A Manger", "Eating Out", "-£6.50", "CARD-5521"),
    _txn(
        "TXN-89930",
        10,
        "ATM Withdrawal - Barclays",
        "Cash",
        "-£100.00",
        rail="atm_withdrawal",
        txn_type="atm_withdrawal",
        card="CARD-5521",
    ),
    _txn(
        "TXN-89920",
        12,
        "Transfer to James Wilson",
        "Transfer",
        "-£50.00",
        rail="bank_transfer",
        txn_type="bank_transfer",
    ),
    # --- Everyday spending: volume and variety, none of it recurring. ---
    _card("TXN-90201", 15, "Sainsbury's Local", "Groceries", "-£23.10", "CARD-5521"),
    _card("TXN-90202", 17, "Costa Coffee", "Eating Out", "-£3.95", "CARD-5521"),
    _card("TXN-90203", 19, "Boots", "Health", "-£8.99", "CARD-5521"),
    _card("TXN-90204", 21, "Uber", "Transport", "-£14.20", "CARD-8834"),
    _card("TXN-90205", 23, "Deliveroo", "Eating Out", "-£28.45", "CARD-8834"),
    _card("TXN-90206", 25, "Screwfix", "Home", "-£56.30", "CARD-8834"),
    _card("TXN-90207", 27, "Waterstones", "Shopping", "-£19.99", "CARD-5521"),
    _card("TXN-90208", 29, "Shell Petrol", "Transport", "-£62.00", "CARD-8834"),
    _card("TXN-90209", 31, "Argos", "Shopping", "-£45.50", "CARD-5521"),
    _card("TXN-90210", 37, "The Crown Inn", "Eating Out", "-£34.80", "CARD-5521"),
    _card("TXN-90211", 39, "Zara", "Shopping", "-£79.00", "CARD-8834"),
    _txn(
        "TXN-90212",
        41,
        "British Gas - Account Top Up",
        "Utilities",
        "-£85.00",
        rail="bank_transfer",
        txn_type="bank_transfer",
    ),
    _card("TXN-90213", 43, "Vue Cinema", "Entertainment", "-£11.50", "CARD-5521"),
    _card("TXN-90214", 45, "Holland & Barrett", "Health", "-£16.75", "CARD-5521"),
    _card("TXN-90215", 47, "Trainline", "Transport", "-£48.60", "CARD-8834"),
    _card("TXN-90216", 49, "Morrisons", "Groceries", "-£41.20", "CARD-5521"),
    _card("TXN-90217", 51, "Etsy", "Shopping", "-£24.99", "CARD-8834"),
    _card("TXN-90218", 53, "Nando's", "Eating Out", "-£31.40", "CARD-5521"),
    _card("TXN-90219", 55, "IKEA", "Home", "-£132.75", "CARD-8834"),
    _card("TXN-90220", 57, "Halfords", "Transport", "-£27.30", "CARD-5521"),
    _card("TXN-90221", 59, "Currys", "Shopping", "-£249.99", "CARD-8834"),
    _card("TXN-90222", 61, "Greggs", "Eating Out", "-£4.15", "CARD-5521"),
    _card("TXN-90223", 69, "National Trust", "Leisure", "-£13.00", "CARD-5521"),
    _card("TXN-90224", 72, "Wickes", "Home", "-£38.90", "CARD-8834"),
    _card("TXN-90225", 78, "Superdrug", "Health", "-£9.45", "CARD-5521"),
    _card("TXN-90226", 85, "Cineworld", "Entertainment", "-£12.99", "CARD-8834"),
]


def is_debit(txn: dict) -> bool:
    """Money leaving the account. The sign is on the formatted amount."""
    return txn["amount"].lstrip().startswith("-")


def txn_date(txn: dict) -> date:
    """The transaction's calendar date, dropping the wall-clock time."""
    return datetime.fromisoformat(txn["date"].replace("Z", "+00:00")).date()


@tool
def get_transaction_history(user_id: str, days: int = 30) -> str:
    """Get recent transaction history for a customer.

    Args:
        user_id: The customer's user ID.
        days: Number of days to look back (default 30, max 90).
    """
    return json.dumps(
        {
            "user_id": user_id,
            "period": f"Last {min(days, 90)} days",
            "transaction_count": len(MOCK_TRANSACTIONS),
            "transactions": MOCK_TRANSACTIONS,
        }
    )


@tool
def get_transaction_details(transaction_id: str) -> str:
    """Get full details of a specific transaction.

    Args:
        transaction_id: The transaction ID (e.g. TXN-90001).
    """
    for txn in MOCK_TRANSACTIONS:
        if txn["transaction_id"] == transaction_id:
            detail = {**txn}
            detail.update(
                {
                    "merchant_category_code": (
                        "5411" if txn["category"] == "Groceries" else "0000"
                    ),
                    "merchant_country": "GB",
                    "exchange_rate": None,
                    "fee": "£0.00",
                    "reference": f"REF-{transaction_id}",
                    "settlement_date": txn["date"][:10],
                }
            )
            return json.dumps(detail)

    return json.dumps(
        {
            "error": f"Transaction {transaction_id} not found.",
            "suggestion": "Use get_transaction_history to list recent transactions.",
        }
    )


@tool
def dispute_transaction(transaction_id: str, reason: str) -> str:
    """File a dispute for a transaction. Starts the chargeback investigation process.

    Args:
        transaction_id: The transaction ID to dispute.
        reason: Reason for the dispute — e.g. 'unauthorised', 'goods_not_received',
                'duplicate_charge', 'incorrect_amount', 'other'.
    """
    valid_reasons = {
        "unauthorised",
        "goods_not_received",
        "duplicate_charge",
        "incorrect_amount",
        "other",
    }
    if reason not in valid_reasons:
        return json.dumps(
            {
                "success": False,
                "error": f"Invalid reason '{reason}'. Must be one of: {', '.join(sorted(valid_reasons))}",
            }
        )

    return json.dumps(
        {
            "success": True,
            "dispute_id": "DSP-44210",
            "transaction_id": transaction_id,
            "reason": reason,
            "status": "under_review",
            "estimated_resolution": "5-10 business days",
            "provisional_credit": reason == "unauthorised",
            "created_at": _days_ago(0, hour=10),
            "message": "Dispute filed successfully. You will receive updates via the app. "
            "If the transaction was unauthorised, a provisional credit may be applied within 24 hours.",
        }
    )


@tool
def get_dispute_status(dispute_id: str) -> str:
    """Check the status of an existing transaction dispute.

    Args:
        dispute_id: The dispute ID (e.g. DSP-44210).
    """
    return json.dumps(
        {
            "dispute_id": dispute_id,
            "transaction_id": "TXN-89985",
            "status": "under_review",
            "reason": "goods_not_received",
            "filed_date": _days_ago(6, hour=10),
            "estimated_resolution": (_TODAY + timedelta(days=5)).isoformat(),
            "provisional_credit_applied": False,
            "updates": [
                {"date": _days_ago(6, hour=10), "note": "Dispute received and logged."},
                {
                    "date": _days_ago(5, hour=14),
                    "note": "Merchant contacted for evidence.",
                },
            ],
            "message": "Your dispute is being reviewed. The merchant has been contacted.",
        }
    )
