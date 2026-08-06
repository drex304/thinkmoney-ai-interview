"""Which rail a merchant is billed on, and the reference that identifies it.

The single source of truth for merchant → (rail, reference). It exists because
that mapping was previously derived twice: once here, correctly, from the
fixtures, and once again by the model, reading the JSON report and choosing a
tool and an identifier from it. The second derivation is the one that failed —
asked to cancel Google One, which is billed to a card and has no mandate at
all, the model called `cancel_direct_debit` on `DD-4472`, Namecheap's domain
renewal.

A model asked *what* to stop cannot get this wrong. A model asked *how* can, so
it is no longer asked. The write tools take a merchant name and resolve the
rail and reference through here, which makes "cancel Google One" incapable of
naming a direct debit.

`find_rails` returns every match rather than a best guess: "Netflix" genuinely
identifies two subscriptions on this account (the direct debit and the shared
standing order), and picking one silently is the failure this module exists to
prevent.
"""

from src.tools.transactions import MOCK_TRANSACTIONS, is_debit, txn_date

# The transaction field carrying each rail's identifier. Mirrors the detection
# tool's own mapping — the same three rails a subscription can leave by.
_RAIL_REFERENCE_FIELD = {
    "direct_debit": "mandate_id",
    "standing_order": "order_id",
    "card_on_file": "card_used",
}

# The only tool that can act on each rail. A direct debit and a standing order
# are cancellable at the bank; a card-on-file subscription is not, because there
# is no mandate to cancel — the charge can only be blocked.
RAIL_TOOL = {
    "direct_debit": "cancel_direct_debit",
    "standing_order": "cancel_standing_order",
    "card_on_file": "block_merchant_on_card",
}

# Said to the customer when the rail they asked about cannot do what they want.
RAIL_DESCRIPTION = {
    "direct_debit": "a direct debit",
    "standing_order": "a standing order",
    "card_on_file": "billed to a card",
}

# Why the wrong rail cannot simply be retried with the same intent.
_RAIL_CONSEQUENCE = {
    "card_on_file": "There is no mandate to cancel — the charge can only be blocked.",
    "direct_debit": "It is cancellable at the bank as a mandate.",
    "standing_order": "It is cancellable at the bank as an order.",
}


def _normalised(name: str) -> str:
    """Lower-cased with runs of whitespace collapsed, for comparing names."""
    return " ".join((name or "").lower().split())


def _names_match(candidate: str, requested: str) -> bool:
    """Whether a merchant name refers to what the customer asked about."""
    left, right = _normalised(candidate), _normalised(requested)
    if not left or not right:
        return False
    return left in right or right in left


def _active_fixtures() -> list[dict]:
    """Live mandates and standing orders, normalised to one shape.

    Imported inside the function rather than at module scope: `payments` imports
    this module for its own validation, so a top-level import here would close
    the cycle. Nothing calls this at import time.
    """
    from src.tools.payments import MOCK_DIRECT_DEBITS, MOCK_STANDING_ORDERS

    fixtures = []

    for mandate in MOCK_DIRECT_DEBITS:
        if mandate.get("status") != "active":
            continue
        fixtures.append(
            {
                "merchant": mandate["merchant"],
                "category": mandate.get("category", ""),
                "rail": "direct_debit",
                "reference": mandate["mandate_id"],
                "amount": mandate.get("amount", ""),
                "frequency": mandate.get("frequency", ""),
            }
        )

    for order in MOCK_STANDING_ORDERS:
        if order.get("status") != "active":
            continue
        fixtures.append(
            {
                # The statement description, not the payee: SO-102 is paid to
                # "James Wilson" but is the customer's "Shared Netflix".
                "merchant": order.get("merchant") or order["payee"],
                "category": order.get("category", ""),
                "rail": "standing_order",
                "reference": order["order_id"],
                "amount": order.get("amount", ""),
                "frequency": order.get("frequency", ""),
            }
        )

    return fixtures


def _card_subscriptions() -> list[dict]:
    """Card-billed merchants, taken from the most recent charge of each.

    Card-on-file has no mandate fixture to read — the card the charge lands on
    is the only reference there is, and it can change, so the latest charge is
    the authoritative one.
    """
    latest: dict[str, dict] = {}

    for txn in MOCK_TRANSACTIONS:
        if not is_debit(txn) or txn.get("rail") != "card_on_file":
            continue
        merchant = txn["description"]
        if merchant not in latest or txn_date(txn) > txn_date(latest[merchant]):
            latest[merchant] = txn

    return [
        {
            "merchant": merchant,
            "category": txn.get("category", ""),
            "rail": "card_on_file",
            "reference": txn.get(_RAIL_REFERENCE_FIELD["card_on_file"]) or "",
            "amount": txn.get("amount", ""),
            "frequency": "",
        }
        for merchant, txn in latest.items()
    ]


def all_rails() -> list[dict]:
    """Every subscription on the account, in the same shape as `find_rails`."""
    return _active_fixtures() + _card_subscriptions()


# Commitments never offered as something to stop. Detection already keeps these
# out of the savings figures; a cancellation menu is a far worse place for them.
# Asked about Netflix, the standing-order list offered the customer their rent.
_ESSENTIAL_CATEGORIES = {"Housing", "Utilities", "Insurance"}


def options_for_rail(rail: str) -> list[dict]:
    """The subscriptions a given tool could offer to act on, in a stable order.

    Sorted by reference so the same account always produces the same list. This
    is what the customer is offered when a write names no subscription they
    asked for: the choice has to come from a list the code generated, or it is
    another guess wearing a different hat.

    Essentials are excluded. Offering to cancel the rent alongside a streaming
    subscription invites exactly the mistake this whole path exists to prevent,
    and a customer who genuinely wants to stop a standing order to their
    landlord can say so by name.
    """
    return sorted(
        (
            row
            for row in all_rails()
            if row["rail"] == rail and row["category"] not in _ESSENTIAL_CATEGORIES
        ),
        key=lambda row: row["reference"],
    )


def options_for(merchant: str, rail: str) -> list[dict]:
    """What to offer the customer when a write could not be bound to a choice.

    Scoped by the merchant the model proposed. That name is not authority — it
    is the thing being checked — but it is a good hint about what the
    conversation is about, and using it as a hint is safe because the customer
    still makes the choice. Asked about Netflix, the customer was offered every
    direct debit on the account, which put their domain renewal in a list about
    streaming.

    Falls back to everything on the rail when the proposed name matches nothing,
    since an empty question is worse than a broad one. Essentials are excluded
    either way.
    """
    # Every fuzzy match, not `find_rails` — its exact-name short-circuit is
    # right for resolving one subscription and wrong for offering a choice
    # between them. Scoped by "Netflix", the customer needs to see both the
    # Netflix direct debit and the shared standing order; that is the question.
    scoped = [
        row
        for row in all_rails()
        if _names_match(row["merchant"], merchant)
        and row["category"] not in _ESSENTIAL_CATEGORIES
    ]

    if scoped:
        return sorted(scoped, key=lambda row: row["reference"])

    return options_for_rail(rail)


def find_rails(merchant: str) -> list[dict]:
    """Every subscription on this account matching `merchant`.

    Exact name matches win outright. Without that, "Netflix" would return both
    the Netflix direct debit and the "James Wilson - Shared Netflix" standing
    order, and a customer naming Netflix exactly means the former.

    Returns dicts of merchant, rail, reference, amount and frequency. An empty
    list means nothing on the account bills under that name — which is itself
    worth reporting rather than proceeding on.
    """
    everything = all_rails()

    exact = [
        row
        for row in everything
        if _normalised(row["merchant"]) == _normalised(merchant)
    ]
    if exact:
        return exact

    return [row for row in everything if _names_match(row["merchant"], merchant)]


def resolve_rail(merchant: str, expected_rail: str) -> tuple[dict | None, str]:
    """Resolve `merchant` to the one subscription on `expected_rail`.

    Returns `(match, error)`. Exactly one of them is meaningful: a match with an
    empty error, or None with a sentence explaining what to do instead. The
    error text is written for the customer, because it is what the agent will
    pass on when a write is refused.
    """
    matches = find_rails(merchant)

    if not matches:
        return None, (
            f"No recurring payment to '{merchant}' was found on this account, "
            "so there is nothing to act on. Check the name against the "
            "subscription list before trying again."
        )

    on_rail = [row for row in matches if row["rail"] == expected_rail]

    if not on_rail:
        wrong = matches[0]
        return None, (
            f"{wrong['merchant']} is {RAIL_DESCRIPTION[wrong['rail']]} "
            f"({wrong['reference']}), not {RAIL_DESCRIPTION[expected_rail]}. "
            f"{_RAIL_CONSEQUENCE[wrong['rail']]} "
            f"Use {RAIL_TOOL[wrong['rail']]} instead."
        )

    if len(on_rail) > 1:
        listed = ", ".join(f"{row['merchant']} ({row['reference']})" for row in on_rail)
        return None, (
            f"'{merchant}' matches more than one subscription: {listed}. "
            "Ask the customer which one they mean rather than choosing."
        )

    return on_rail[0], ""
