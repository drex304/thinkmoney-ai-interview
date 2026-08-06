"""Mock payment and transfer tools for thinkmoney customer service."""

import json
from datetime import date, datetime, timedelta
from typing import Annotated

from langchain_core.tools import InjectedToolCallId, tool

from src.confirmation import requires_confirmation
from src.tools.rails import resolve_rail


def _days_from_now(days: int) -> str:
    """ISO date `days` days ahead — keeps scheduled payments coherent with the corpus."""
    return (date.today() + timedelta(days=days)).isoformat()


# The bank's own view of the two non-card subscription rails. Subscription
# detection reads these directly so a live mandate is discoverable even when the
# corpus window is too short to show it charging.
MOCK_STANDING_ORDERS = [
    {
        "order_id": "SO-101",
        "payee": "Landlord - Premier Properties",
        "payee_id": "PAY-002",
        "merchant": "Landlord - Premier Properties",
        "category": "Housing",
        "amount": "£1,200.00",
        "frequency": "monthly",
        "next_payment": _days_from_now(26),
        "reference": "Rent - Flat 4B",
        "status": "active",
        "created": "2023-06-01",
    },
    {
        "order_id": "SO-102",
        "payee": "James Wilson",
        "payee_id": "PAY-001",
        "merchant": "James Wilson - Shared Netflix",
        "category": "Entertainment",
        "amount": "£25.00",
        "frequency": "monthly",
        "next_payment": _days_from_now(25),
        "reference": "Shared Netflix",
        "status": "active",
        "created": "2024-01-10",
    },
]

MOCK_DIRECT_DEBITS = [
    {
        "mandate_id": "DD-4471",
        "merchant": "Netflix",
        "category": "Entertainment",
        "amount": "£15.99",
        "frequency": "monthly",
        "next_payment": _days_from_now(25),
        "reference": "NETFLIX.COM",
        "status": "active",
        "created": "2023-11-02",
    },
    {
        "mandate_id": "DD-4472",
        "merchant": "Namecheap Domain Renewal",
        "category": "Software",
        "amount": "£79.99",
        "frequency": "annual",
        "next_payment": _days_from_now(11),
        "reference": "NAMECHEAP RENEWAL",
        "status": "active",
        "created": "2022-08-14",
    },
]


@tool
def get_payees(user_id: str) -> str:
    """List all saved payees for a customer's account.

    Args:
        user_id: The customer's user ID.
    """
    return json.dumps(
        {
            "user_id": user_id,
            "payees": [
                {
                    "payee_id": "PAY-001",
                    "name": "James Wilson",
                    "sort_code": "20-30-40",
                    "account_number": "12345678",
                    "type": "personal",
                    "last_paid": _days_from_now(-5),
                },
                {
                    "payee_id": "PAY-002",
                    "name": "Landlord - Premier Properties",
                    "sort_code": "11-22-33",
                    "account_number": "87654321",
                    "type": "business",
                    "last_paid": _days_from_now(-4),
                },
                {
                    "payee_id": "PAY-003",
                    "name": "Maria Garcia",
                    "iban": "ES91 2100 0418 4502 0005 1332",
                    "type": "international",
                    "currency": "EUR",
                    "last_paid": _days_from_now(-48),
                },
            ],
        }
    )


@tool
def initiate_transfer(
    user_id: str,
    payee_id: str,
    amount: float,
    currency: str = "GBP",
    reference: str = "",
) -> str:
    """Initiate a money transfer to a saved payee.

    Args:
        user_id: The customer's user ID.
        payee_id: The payee ID to send money to.
        amount: Amount to transfer (must be positive).
        currency: Currency code (default GBP).
        reference: Optional payment reference.
    """
    if amount <= 0:
        return json.dumps(
            {
                "success": False,
                "error": "Amount must be a positive number.",
            }
        )

    if amount > 25000:
        return json.dumps(
            {
                "success": False,
                "error": "Amount exceeds single transfer limit of £25,000. "
                "Please contact support for higher-value transfers.",
            }
        )

    is_international = payee_id == "PAY-003"

    return json.dumps(
        {
            "success": True,
            "payment_id": "PMT-78432",
            "user_id": user_id,
            "payee_id": payee_id,
            "amount": f"{amount:.2f}",
            "currency": currency,
            "reference": reference or "thinkmoney transfer",
            "type": "SWIFT" if is_international else "Faster Payments",
            "fee": "£3.00" if is_international else "£0.00",
            "estimated_arrival": (
                "2-5 business days" if is_international else "Within minutes"
            ),
            "initiated_at": datetime.now().isoformat(),
            "status": "processing",
            "message": "Transfer initiated successfully.",
        }
    )


@tool
def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """Get the current exchange rate between two currencies.

    Args:
        from_currency: Source currency code (e.g. GBP).
        to_currency: Target currency code (e.g. EUR).
    """
    rates = {
        ("GBP", "EUR"): 1.1650,
        ("GBP", "USD"): 1.2710,
        ("GBP", "PLN"): 5.0820,
        ("GBP", "RON"): 5.8140,
        ("EUR", "GBP"): 0.8584,
        ("EUR", "USD"): 1.0910,
        ("USD", "GBP"): 0.7868,
        ("USD", "EUR"): 0.9166,
        ("PLN", "GBP"): 0.1968,
        ("RON", "GBP"): 0.1720,
    }

    pair = (from_currency.upper(), to_currency.upper())
    rate = rates.get(pair)

    if rate is None:
        return json.dumps(
            {
                "error": f"Exchange rate for {pair[0]}/{pair[1]} not available.",
                "supported_currencies": ["GBP", "EUR", "USD", "PLN", "RON"],
            }
        )

    # Simulate weekend markup
    is_weekend = datetime.now().weekday() >= 5
    markup = 0.015 if is_weekend else 0.005

    return json.dumps(
        {
            "from": pair[0],
            "to": pair[1],
            "rate": round(rate, 4),
            "markup": f"{markup * 100:.1f}%",
            "effective_rate": round(rate * (1 - markup), 4),
            "weekend_rate": is_weekend,
            "timestamp": datetime.now().isoformat(),
            "note": (
                "Weekend rates include a higher markup due to FX market closure."
                if is_weekend
                else "Standard weekday rate."
            ),
        }
    )


@tool
def get_standing_orders(user_id: str) -> str:
    """List all active standing orders for a customer.

    Args:
        user_id: The customer's user ID.
    """
    return json.dumps(
        {
            "user_id": user_id,
            "standing_orders": MOCK_STANDING_ORDERS,
        }
    )


def _describe_cancel_standing_order(args: dict) -> str:
    """What the customer is asked to confirm, in terms they can check.

    Named payee and amount, not just the reference. See
    `_describe_cancel_direct_debit` for why an opaque identifier makes the gate
    unable to do its job.
    """
    resolved, error = resolve_rail(args.get("merchant", ""), "standing_order")

    if resolved is None:
        return f"Cancel a standing order — but {error} Nothing will be cancelled."

    # The resolved name, not the payee: SO-102 is paid to "James Wilson", which
    # says who is paid but not what for. "James Wilson - Shared Netflix" is what
    # the customer would recognise as the one they meant, or did not.
    return (
        f"Cancel the standing order {resolved['reference']} — {resolved['amount']} "
        f"{resolved['frequency']} to {resolved['merchant']}. No further payments "
        "will be sent."
    )


def _describe_cancel_direct_debit(args: dict) -> str:
    """What the customer is asked to confirm, in terms they can check.

    This used to name the mandate and nothing else — "Cancel the direct debit
    DD-4472". A customer asked to cancel Google One approved exactly that, and
    DD-4472 is Namecheap's domain renewal: Google One is card-on-file and has no
    mandate at all. The agent had picked the wrong rail and a merchant's
    reference that was not theirs, and the gate — the one safeguard positioned
    to catch it — showed the customer an identifier they had no way to
    recognise as wrong.

    So the description resolves the reference back to the merchant and amount it
    belongs to. A mismatch between what was asked for and what is about to
    happen becomes visible at the moment of the decision, which is the only
    moment it can still be stopped. The contract caveat is here for the same
    reason: it changes whether cancelling is the right call, so it has to arrive
    before the customer answers, not in the confirmation afterwards.
    """
    merchant = args.get("merchant", "")
    resolved, error = resolve_rail(merchant, "direct_debit")

    if resolved is None:
        return f"Cancel a direct debit — but {error} Nothing will be cancelled."

    claimed = args.get("mandate_id")
    if claimed and claimed != resolved["reference"]:
        return (
            f"Cancel {merchant}'s direct debit — but {claimed} was named and "
            f"that reference belongs to a different subscription. {merchant} is "
            f"billed on {resolved['reference']}. Nothing will be cancelled."
        )

    return (
        f"Cancel the direct debit {resolved['reference']} — {resolved['amount']} "
        f"{resolved['frequency']} to {resolved['merchant']}. No further payments "
        f"will be taken from your account for it. This stops the payment, not "
        f"the contract: if you are signed up with {resolved['merchant']}, you may "
        "still owe them until you cancel with them directly."
    )


@tool
@requires_confirmation(_describe_cancel_standing_order)
def cancel_standing_order(
    merchant: str,
    order_id: str = "",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> str:
    """Cancel an active standing order.

    Name the merchant or payee the customer asked about. The order is looked up
    from that name, so a subscription billed on another rail is refused rather
    than matched to an unrelated order.

    Halts for the customer's explicit confirmation before it runs. Call it when
    they ask — do not ask them in prose first.

    Args:
        merchant: The payee whose standing order is being cancelled, as it
            appears in the subscription list (e.g. "James Wilson - Shared
            Netflix").
        order_id: Optional. The expected standing order ID. Supplying one that
            does not belong to `merchant` refuses the cancellation.
    """
    resolved, error = resolve_rail(merchant, "standing_order")

    if resolved is None:
        return json.dumps(
            {
                "success": False,
                "error": error,
                "suggestion": "Use find_recurring_payments to list every "
                "recurring payment with its rail and reference.",
            },
            indent=2,
        )

    if order_id and order_id != resolved["reference"]:
        return json.dumps(
            {
                "success": False,
                "error": (
                    f"{order_id} is not {merchant}'s standing order — "
                    f"{merchant} is billed on {resolved['reference']}. "
                    "Nothing was cancelled."
                ),
            },
            indent=2,
        )

    order_id = resolved["reference"]

    return json.dumps(
        {
            "success": True,
            "order_id": order_id,
            "merchant": resolved["merchant"],
            "status": "cancelled",
            "cancelled_at": datetime.now().isoformat(),
            "message": f"Standing order {order_id} to {resolved['merchant']} "
            "has been cancelled. No further payments will be made.",
        }
    )


@tool
@requires_confirmation(_describe_cancel_direct_debit)
def cancel_direct_debit(
    user_id: str,
    merchant: str,
    mandate_id: str = "",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> str:
    """Cancel a direct debit mandate at the bank, stopping all future collections.

    The customer has a right to cancel any direct debit through their bank, so
    this needs no involvement from the merchant. It stops the money leaving the
    account; it does not terminate the underlying contract with the merchant.

    Name the merchant the customer asked about. The mandate is looked up from
    that name, so a merchant billed on another rail is refused rather than
    matched to somebody else's mandate.

    Halts for the customer's explicit confirmation before it runs. Call it when
    they ask — do not ask them in prose first.

    Args:
        user_id: The customer's user ID.
        merchant: The merchant whose direct debit is being cancelled, as it
            appears in the subscription list (e.g. "Netflix").
        mandate_id: Optional. The expected mandate ID. Supplying one that does
            not belong to `merchant` refuses the cancellation rather than
            performing it.
    """
    resolved, error = resolve_rail(merchant, "direct_debit")

    if resolved is None:
        result = {
            "success": False,
            "error": error,
            "suggestion": "Use find_recurring_payments to list every recurring "
            "payment with its rail and reference, and act on the "
            "one whose merchant the customer actually named.",
        }
        return json.dumps(result, indent=2)

    # A supplied identifier is treated as a claim to check, never as the
    # instruction. This is the mismatch that cancelled Namecheap's domain
    # renewal when the customer had asked about Google One.
    if mandate_id and mandate_id != resolved["reference"]:
        return json.dumps(
            {
                "success": False,
                "error": (
                    f"{mandate_id} is not {merchant}'s direct debit — "
                    f"{merchant} is billed on {resolved['reference']}. "
                    "Nothing was cancelled."
                ),
                "suggestion": "Re-read the subscription list. Acting on a "
                "reference belonging to another merchant cancels "
                "the wrong subscription.",
            },
            indent=2,
        )

    mandate_id = resolved["reference"]
    mandate = next(
        (dd for dd in MOCK_DIRECT_DEBITS if dd["mandate_id"] == mandate_id), None
    )

    if mandate is None:
        valid_mandate_ids = [dd["mandate_id"] for dd in MOCK_DIRECT_DEBITS]

        result = {
            "success": False,
            "error": f"Direct debit mandate '{mandate_id}' not found. "
            f"Must be one of: {', '.join(valid_mandate_ids)}",
            "valid_mandate_ids": valid_mandate_ids,
            "suggestion": "Use find_recurring_payments to list every recurring "
            "payment with its rail and mandate ID. Standing orders "
            "(SO-…) are cancelled with cancel_standing_order, and "
            "card subscriptions cannot be cancelled by the bank.",
        }
    else:
        # Deliberately does not mutate MOCK_DIRECT_DEBITS: the fixtures are the
        # bank's read model and detection is asserted to be repeatable, so a
        # cancellation here must not silently rewrite a later report.
        result = {
            "success": True,
            "user_id": user_id,
            "mandate_id": mandate_id,
            "merchant": mandate["merchant"],
            "amount": mandate["amount"],
            "frequency": mandate["frequency"],
            "status": "cancelled",
            "cancelled_at": date.today().isoformat(),
            "cancelled_next_payment": mandate["next_payment"],
            "message": f"Direct debit {mandate_id} to {mandate['merchant']} "
            f"({mandate['amount']} {mandate['frequency']}) has been "
            "cancelled. No further payments will be taken. The collection "
            f"due on {mandate['next_payment']} will not go out.",
            "note": "Cancelling the mandate stops the payment, not the contract. "
            f"If you are still signed up with {mandate['merchant']} they may "
            "chase the balance or pursue it as a missed payment, so cancel "
            "the service with them as well.",
        }

    return json.dumps(result)
