"""Mock card management tools for thinkmoney customer service."""

import json
from datetime import date, datetime
from typing import Annotated

from langchain_core.tools import InjectedToolCallId, tool

from src.confirmation import requires_confirmation
from src.tools.rails import resolve_rail
from src.tools.transactions import MOCK_TRANSACTIONS, is_debit, txn_date

# The cards on the account. Lifted out of list_cards so other tools can
# validate a card ID against the same source of truth instead of restating it.
MOCK_CARDS = [
    {
        "card_id": "CARD-5521",
        "type": "physical",
        "scheme": "Visa Debit",
        "variant": "Metal (Premium)",
        "last_four": "4821",
        "status": "active",
        "frozen": False,
        "expiry": "09/28",
        "contactless_enabled": True,
        "daily_limit": "£15,000",
    },
    {
        "card_id": "CARD-8834",
        "type": "virtual",
        "scheme": "Visa Debit",
        "variant": "Standard",
        "last_four": "7203",
        "status": "active",
        "frozen": False,
        "expiry": "12/27",
        "contactless_enabled": False,
        "daily_limit": "£5,000",
    },
]


@tool
def list_cards(user_id: str) -> str:
    """List all cards (physical and virtual) associated with a customer's account.

    Args:
        user_id: The customer's user ID.
    """
    return json.dumps(
        {
            "user_id": user_id,
            "cards": MOCK_CARDS,
        }
    )


@tool
def freeze_card(card_id: str) -> str:
    """Freeze a card immediately, blocking all transactions. Can be unfrozen later.

    Args:
        card_id: The card ID to freeze (e.g. CARD-5521).
    """
    return json.dumps(
        {
            "success": True,
            "card_id": card_id,
            "status": "frozen",
            "frozen_at": datetime.now().isoformat(),
            "message": "Card has been frozen. No transactions will be authorised until the card is unfrozen.",
        }
    )


@tool
def unfreeze_card(card_id: str) -> str:
    """Unfreeze a previously frozen card, restoring normal transaction capability.

    Args:
        card_id: The card ID to unfreeze.
    """
    return json.dumps(
        {
            "success": True,
            "card_id": card_id,
            "status": "active",
            "unfrozen_at": datetime.now().isoformat(),
            "message": "Card has been unfrozen and is now active for transactions.",
        }
    )


@tool
def order_replacement_card(
    user_id: str, card_id: str, reason: str, delivery: str = "standard"
) -> str:
    """Order a replacement for a lost, stolen, or damaged card.

    Args:
        user_id: The customer's user ID.
        card_id: The card ID to replace.
        reason: Reason for replacement — one of 'lost', 'stolen', 'damaged', 'expired'.
        delivery: Delivery speed — 'standard' (5-7 days, free) or 'express' (1-2 days, £10).
    """
    fee = "£0.00" if delivery == "standard" else "£10.00"
    days = "5-7 business days" if delivery == "standard" else "1-2 business days"

    result = {
        "success": True,
        "user_id": user_id,
        "old_card_id": card_id,
        "new_card_id": "CARD-9917",
        "reason": reason,
        "delivery": delivery,
        "fee": fee,
        "estimated_delivery": days,
        "tracking_ref": "TM-REPL-20260324-001",
        "message": f"Replacement card ordered ({delivery} delivery, {days}). "
        f"Old card has been cancelled. Fee: {fee}.",
    }

    if reason in ("lost", "stolen"):
        result["old_card_status"] = "cancelled"
        result[
            "message"
        ] += " We recommend monitoring your recent transactions for any unauthorised activity."

    return json.dumps(result)


@tool
def get_card_status(card_id: str) -> str:
    """Get detailed status information for a specific card.

    Args:
        card_id: The card ID to check.
    """
    cards = {
        "CARD-5521": {
            "card_id": "CARD-5521",
            "type": "physical",
            "scheme": "Visa Debit",
            "variant": "Metal (Premium)",
            "last_four": "4821",
            "status": "active",
            "frozen": False,
            "expiry": "09/28",
            "contactless_enabled": True,
            "pin_set": True,
            "apple_pay_enrolled": True,
            "google_pay_enrolled": False,
            "last_used": "2026-03-23T18:42:00Z",
            "last_used_merchant": "Tesco Express",
        },
        "CARD-8834": {
            "card_id": "CARD-8834",
            "type": "virtual",
            "scheme": "Visa Debit",
            "variant": "Standard",
            "last_four": "7203",
            "status": "active",
            "frozen": False,
            "expiry": "12/27",
            "contactless_enabled": False,
            "pin_set": False,
            "apple_pay_enrolled": False,
            "google_pay_enrolled": False,
            "last_used": "2026-03-20T11:05:00Z",
            "last_used_merchant": "Amazon.co.uk",
        },
    }

    if card_id in cards:
        return json.dumps(cards[card_id])

    return json.dumps(
        {
            "error": f"Card {card_id} not found.",
            "suggestion": "Use list_cards to see available cards.",
        }
    )


def _describe_block_merchant_on_card(args: dict) -> str:
    """What the customer is asked to confirm, with the card resolved.

    The card comes from the merchant rather than from the caller, so the
    confirmation shows the card that will actually be blocked — not the one the
    model believed was involved.
    """
    merchant = args.get("merchant") or "(unknown merchant)"
    resolved, error = resolve_rail(merchant, "card_on_file")

    if resolved is None:
        return f"Block {merchant} — but {error} Nothing will be blocked."

    return (
        f"Block {resolved['merchant']} from charging card "
        f"{resolved['reference']}. This stops the payment, not the "
        "contract — you may still owe the merchant until you cancel with them."
    )


@tool
@requires_confirmation(_describe_block_merchant_on_card)
def block_merchant_on_card(
    user_id: str,
    merchant: str,
    card_id: str = "",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> str:
    """Block future card payments to a merchant on one of the customer's cards.

    This is the only lever the bank has over a subscription billed to a card:
    there is no mandate to cancel, so the bank can refuse the charges but cannot
    end the agreement. Use it when the customer wants the money to stop now, and
    always pass on the caveat — the contract survives the block.

    The card is looked up from the merchant, so a subscription the bank could
    actually cancel is refused here rather than blocked instead.

    Halts for the customer's explicit confirmation before it runs. Call it when
    they ask — do not ask them in prose first.

    Args:
        user_id: The customer's user ID.
        merchant: The merchant name as it appears on the statement.
        card_id: Optional. The expected card. Supplying one the merchant does
            not bill refuses the block rather than performing it.
    """
    resolved, error = resolve_rail(merchant, "card_on_file")

    if resolved is None:
        return json.dumps(
            {
                "success": False,
                "error": error,
                "suggestion": "Use find_recurring_payments to see which card "
                "each subscription bills, and which rail it is on.",
            },
            indent=2,
        )

    # A supplied card is honoured as long as it is a real card on the account.
    # Unlike a mandate, a card block is forward-looking: blocking a merchant on
    # a card it has not charged yet is a legitimate instruction, so this must
    # not insist on the card the merchant currently bills. The rail check above
    # is what stops a cancellable subscription being blocked instead.
    card_id = card_id or resolved["reference"]
    card = next((c for c in MOCK_CARDS if c["card_id"] == card_id), None)

    if card is None:
        valid_card_ids = [c["card_id"] for c in MOCK_CARDS]

        result = {
            "success": False,
            "error": f"Card '{card_id}' not found on this account. "
            f"Must be one of: {', '.join(valid_card_ids)}",
            "valid_card_ids": valid_card_ids,
            "suggestion": "Use list_cards to see the customer's cards, or "
            "find_recurring_payments to see which card each "
            "subscription bills. Direct debits are stopped with "
            "cancel_direct_debit and standing orders with "
            "cancel_standing_order — neither needs a card block.",
        }
    else:
        charges = [
            txn
            for txn in MOCK_TRANSACTIONS
            if txn["description"] == merchant
            and txn.get("card_used") == card_id
            and is_debit(txn)
        ]
        charges.sort(key=txn_date)

        if charges:
            known_charges = {
                "occurrences": len(charges),
                "amount": charges[-1]["amount"].lstrip("-"),
                "last_charged": txn_date(charges[-1]).isoformat(),
            }
        else:
            # A block is forward-looking, so an unrecognised merchant is not an
            # error — but say so, in case the name is wrong and the real charges
            # would sail straight through.
            known_charges = {
                "occurrences": 0,
                "note": f"No charges from '{merchant}' found on {card_id}. The block "
                "is still in place, but check the merchant name on the "
                "statement — payments billed under a different name will "
                "not be caught by it.",
            }

        # Stateless, like every other money-moving tool here: the fixtures are
        # the bank's read model and detection output is asserted to be
        # repeatable.
        result = {
            "success": True,
            "user_id": user_id,
            "card_id": card_id,
            "last_four": card["last_four"],
            "merchant": merchant,
            "status": "blocked",
            "blocked_at": date.today().isoformat(),
            "known_charges": known_charges,
            "message": f"Future payments to {merchant} on card ending "
            f"{card['last_four']} ({card_id}) will be declined from now on.",
            "caveat": f"Blocking the card does not cancel your {merchant} "
            "subscription — it stops the payment, not the contract. The "
            "contract is still running, so you may still owe them, and "
            "they can chase the balance, add late fees or pass it to a "
            "debt collector.",
            "recommended_next_step": f"Cancel directly with {merchant} as well, so "
            "the contract ends rather than just the "
            "payments failing.",
        }

    return json.dumps(result)
