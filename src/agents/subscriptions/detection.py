"""Recurring-payment detection and the savings arithmetic built on it.

Detection is plain deterministic Python: the same statement always produces the
same report, so the agent can quote figures without inventing them. The LLM's
job is to explain the output, never to compute it.

It reads the bank's shared records — transactions, mandates, standing orders —
rather than owning them. Only the plan and billing reference data in `data.py`
belongs to this agent.
"""

import json
from datetime import date, timedelta
from decimal import Decimal

from langchain_core.tools import tool

from src.agents.subscriptions.data import (
    MOCK_BILLING_OPTIONS,
    MOCK_PLAN_TIERS,
)
from src.tools.payments import MOCK_DIRECT_DEBITS, MOCK_STANDING_ORDERS
from src.tools.transactions import MOCK_TRANSACTIONS, is_debit, txn_date

# matched to the cadence whose window contains it; anything outside every window
# is "irregular" and is not treated as a subscription.
_CADENCES = [
    ("weekly", 7, 52, range(5, 10)),
    ("fortnightly", 14, 26, range(11, 19)),
    ("monthly", 30, 12, range(24, 46)),
    ("quarterly", 91, 4, range(80, 100)),
    ("annual", 365, 1, range(330, 400)),
]

_CADENCE_BY_NAME = {name: (days, per_year) for name, days, per_year, _ in _CADENCES}

# The three rails a subscription can leave the account by. Each implies a
# different remedy, so the rail is carried through to the report.
_SUBSCRIPTION_RAILS = {
    "direct_debit": "mandate_id",
    "standing_order": "order_id",
    "card_on_file": "card_used",
}

# Commitments the customer cannot simply cancel. They are still detected — the
# report would be misleading without them — but they are kept out of the
# headline "what you could save" figures.
_ESSENTIAL_CATEGORIES = {"Housing", "Utilities", "Insurance"}


def _parse_amount(amount: str) -> Decimal:
    """'-£1,200.00' -> Decimal('1200.00'). Sign is dropped; direction is `type`."""
    return Decimal(amount.replace("£", "").replace(",", "").lstrip("+-").strip())


def _days_between_charges(charge_dates: list[date]) -> list[int]:
    """The gap in days between each charge and the one before it.

    Expects the dates in ascending order, which is how the caller sorts them.
    One charge yields no gaps, and so no cadence to infer.
    """
    previous_dates = charge_dates[:-1]
    following_dates = charge_dates[1:]

    return [
        (later - earlier).days
        for earlier, later in zip(previous_dates, following_dates, strict=True)
    ]


def _median(values: list[int]) -> int:
    """The middle value once the list is in order.

    An even-length list takes the upper of the two middle values rather than
    averaging them, so the result stays a whole number of days.
    """
    ordered = sorted(values)
    middle_index = len(ordered) // 2
    return ordered[middle_index]


def _cadence_for_gap(gap_in_days: int) -> str | None:
    """The cadence whose window contains this gap, or None if none does."""
    for cadence_name, _days, _per_year, window in _CADENCES:
        if gap_in_days in window:
            return cadence_name

    return None


def _infer_cadence(charge_dates: list[date]) -> str:
    """Get the cadence from the median gap between charges.

    The median rather than the mean so one late payment cannot drag a monthly
    subscription into another bucket. A gap that falls in no cadence window
    makes the merchant "irregular", which the caller drops rather than
    reporting as a subscription.
    """
    gaps = _days_between_charges(charge_dates)
    if not gaps:
        return "irregular"

    typical_gap = _median(gaps)
    cadence = _cadence_for_gap(typical_gap)

    return cadence if cadence is not None else "irregular"


def _annualised(amount: Decimal, cadence: str) -> Decimal:
    per_year = _CADENCE_BY_NAME[cadence][1]
    return amount * per_year


def _group_by_merchant(transactions: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for txn in transactions:
        if not is_debit(txn) or txn["rail"] not in _SUBSCRIPTION_RAILS:
            continue
        groups.setdefault(txn["description"], []).append(txn)
    return groups


def _rail_reference(txn: dict) -> str:
    return txn.get(_SUBSCRIPTION_RAILS[txn["rail"]]) or ""


def _fixtures_by_reference() -> dict[str, dict]:
    """Active mandates and standing orders keyed by their rail identifier.

    Keyed by identifier rather than merchant name because the fixture payee and
    the statement description differ (SO-102 is "James Wilson" on the standing
    order and "James Wilson - Shared Netflix" on the statement).
    """
    fixtures = {}
    for mandate in MOCK_DIRECT_DEBITS:
        if mandate["status"] == "active":
            fixtures[mandate["mandate_id"]] = {**mandate, "rail": "direct_debit"}
    for order in MOCK_STANDING_ORDERS:
        if order["status"] == "active":
            fixtures[order["order_id"]] = {**order, "rail": "standing_order"}
    return fixtures


def _detect_from_transactions(fixtures: dict[str, dict]) -> list[dict]:
    """Build one record per merchant that charges the account more than once."""
    detected = []

    for merchant, charges in _group_by_merchant(MOCK_TRANSACTIONS).items():
        # A single charge is a purchase, not a subscription — there is no
        # interval to infer and no future payment to avoid.
        if len(charges) < 2:
            continue

        charges = sorted(charges, key=txn_date)
        cadence = _infer_cadence([txn_date(c) for c in charges])
        if cadence == "irregular":
            continue

        latest = charges[-1]
        amount = _parse_amount(latest["amount"])
        cadence_days = _CADENCE_BY_NAME[cadence][0]
        reference = _rail_reference(latest)
        fixture = fixtures.get(reference)

        # The bank's own mandate record beats an extrapolated date when we have it.
        next_expected = (txn_date(latest) + timedelta(days=cadence_days)).isoformat()
        next_expected_source = "inferred_from_cadence"
        if fixture and fixture.get("next_payment"):
            next_expected = fixture["next_payment"]
            next_expected_source = (
                "direct_debit_mandate"
                if fixture["rail"] == "direct_debit"
                else "standing_order"
            )

        detected.append(
            {
                "merchant": merchant,
                "category": latest["category"],
                "amount": latest["amount"],
                "cadence": cadence,
                "rail": latest["rail"],
                "rail_reference": reference,
                "occurrences": len(charges),
                "first_charged": txn_date(charges[0]).isoformat(),
                "last_charged": txn_date(latest).isoformat(),
                "next_expected": next_expected,
                "next_expected_source": next_expected_source,
                "annualised_cost": f"{_annualised(amount, cadence):.2f}",
                "essential": latest["category"] in _ESSENTIAL_CATEGORIES,
                "source": "transactions",
                "charges": [
                    {"date": txn_date(c).isoformat(), "amount": c["amount"]}
                    for c in charges
                ],
            }
        )

    return detected


def _detect_from_fixtures(fixtures: dict[str, dict], seen: set[str]) -> list[dict]:
    """Report live mandates the corpus never shows charging.

    An active mandate is a recurring commitment whether or not it happens to
    have billed inside the statement window, so it belongs in the report.
    """
    detected = []

    for reference, fixture in fixtures.items():
        if reference in seen:
            continue
        cadence = fixture.get("frequency", "monthly")
        if cadence not in _CADENCE_BY_NAME:
            continue
        amount = _parse_amount(fixture["amount"])
        category = fixture.get("category", "Uncategorised")

        detected.append(
            {
                "merchant": fixture.get("merchant") or fixture.get("payee", reference),
                "category": category,
                "amount": f"-£{amount:,.2f}",
                "cadence": cadence,
                "rail": fixture["rail"],
                "rail_reference": reference,
                "occurrences": 0,
                "first_charged": None,
                "last_charged": None,
                "next_expected": fixture.get("next_payment"),
                "next_expected_source": (
                    "direct_debit_mandate"
                    if fixture["rail"] == "direct_debit"
                    else "standing_order"
                ),
                "annualised_cost": f"{_annualised(amount, cadence):.2f}",
                "essential": category in _ESSENTIAL_CATEGORIES,
                "source": "mandate_only",
                "charges": [],
            }
        )

    return detected


# A renewal further out than this is not worth interrupting the customer over;
# anything sooner is still cancellable before the money leaves.
_RENEWAL_WARNING_DAYS = 30

# Cadences whose next charge is worth flagging. Monthly and shorter always fall
# inside the warning window, so flagging them would bury the one renewal that
# genuinely needs a decision this month.
_INFREQUENT_CADENCES = {"quarterly", "annual"}


def _service_name(merchant: str) -> str:
    """cleanup merchant name"""
    return merchant.lower().replace("-", " ")


def _detect_duplicates(subscriptions: list[dict]) -> list[dict]:
    """Flag two subscriptions that pay for the same service.

    A match is one merchant's whole name appearing inside another's, which is
    what a shared or resold subscription looks like on a statement. The more
    expensive of the pair is the one worth dropping, since the cheaper one buys
    the same thing.
    """
    strategies = []

    for candidate in subscriptions:
        candidate_words = _service_name(candidate["merchant"]).split()
        for other in subscriptions:
            if other is candidate:
                continue
            other_words = _service_name(other["merchant"]).split()
            if len(other_words) >= len(candidate_words):
                continue
            if not _is_subsequence(other_words, candidate_words):
                continue
            dearer, cheaper = sorted(
                (candidate, other),
                key=lambda s: Decimal(s["annualised_cost"]),
                reverse=True,
            )
            strategies.append(
                {
                    "strategy": "duplicate",
                    "merchant": dearer["merchant"],
                    "duplicate_of": cheaper["merchant"],
                    "rail": dearer["rail"],
                    "rail_reference": dearer["rail_reference"],
                    # Phrased as an option, not an order. This string reaches the
                    # model alongside both identifiers, and an imperative here
                    # reads to it as an instruction: one provider acted on it
                    # unprompted and cancelled the kept subscription rather than
                    # the duplicate.
                    "recommended_action": (
                        f"Cancelling {dearer['rail_reference']} would stop paying "
                        f"twice — {cheaper['merchant']} already covers this at "
                        f"{cheaper['amount'].lstrip('-')} a month. "
                        f"{dearer['rail_reference']} is the one to drop; "
                        f"{cheaper['rail_reference']} is the one to keep."
                    ),
                    "annual_saving": dearer["annualised_cost"],
                    "evidence": (
                        f"{dearer['merchant']} ({dearer['rail_reference']}) and "
                        f"{cheaper['merchant']} ({cheaper['rail_reference']}) both "
                        f"bill monthly for the same service."
                    ),
                }
            )

    return strategies


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    """True when every word of `needle` appears in `haystack`, in order."""
    remaining = iter(haystack)
    return all(word in remaining for word in needle)


def _detect_converted_trials(subscriptions: list[dict]) -> list[dict]:
    """Flag a free trial that has quietly started charging.

    The signature is a £0.00 charge followed by a real one: the customer agreed
    to the trial, not necessarily to the year that follows it.
    """
    strategies = []

    for sub in subscriptions:
        amounts = [_parse_amount(c["amount"]) for c in sub["charges"]]
        if not amounts or amounts[0] != 0:
            continue
        paid = [amount for amount in amounts if amount > 0]
        if not paid:
            continue
        strategies.append(
            {
                "strategy": "converted_trial",
                "merchant": sub["merchant"],
                "rail": sub["rail"],
                "rail_reference": sub["rail_reference"],
                "recommended_action": (
                    f"Decide whether you still want {sub['merchant']} now the free "
                    f"trial has ended — cancelling stops the full "
                    f"£{Decimal(sub['annualised_cost']):,.2f} a year."
                ),
                "annual_saving": sub["annualised_cost"],
                "evidence": (
                    f"First charge was £0.00 on {sub['charges'][0]['date']}, then "
                    f"£{paid[0]:,.2f} from "
                    f"{sub['charges'][amounts.index(paid[0])]['date']}."
                ),
            }
        )

    return strategies


def _detect_price_rises(subscriptions: list[dict]) -> list[dict]:
    """
    Flag a subscription that costs more than it used to.
    The customer need to know the price went up.
    """
    strategies = []

    for sub in subscriptions:
        amounts = [_parse_amount(c["amount"]) for c in sub["charges"]]
        paid = [amount for amount in amounts if amount > 0]
        # A £0.00 opening charge is a trial ending, not a price rise; calling it
        # one would misdescribe what happened.
        if len(paid) < 2 or len(paid) != len(amounts):
            continue
        was, now = paid[0], paid[-1]
        if now <= was:
            continue
        delta = now - was
        strategies.append(
            {
                "strategy": "price_rise",
                "merchant": sub["merchant"],
                "rail": sub["rail"],
                "rail_reference": sub["rail_reference"],
                "recommended_action": (
                    f"{sub['merchant']} has put its price up. Challenge it, switch "
                    f"plan, or cancel to avoid the extra "
                    f"£{_annualised(delta, sub['cadence']):,.2f} a year."
                ),
                "annual_saving": f"{_annualised(delta, sub['cadence']):.2f}",
                "evidence": (
                    f"Charge rose from £{was:,.2f} to £{now:,.2f} "
                    f"(£{delta:,.2f} more per {sub['cadence']} charge)."
                ),
            }
        )

    return strategies


def _detect_upcoming_renewals(subscriptions: list[dict]) -> list[dict]:
    """Flag an infrequent renewal about to take a large one-off payment."""
    strategies = []
    today = date.today()

    for sub in subscriptions:
        if sub["cadence"] not in _INFREQUENT_CADENCES or not sub["next_expected"]:
            continue
        days_until = (date.fromisoformat(sub["next_expected"]) - today).days
        if not 0 <= days_until <= _RENEWAL_WARNING_DAYS:
            continue
        amount = _parse_amount(sub["amount"])
        strategies.append(
            {
                "strategy": "upcoming_renewal",
                "merchant": sub["merchant"],
                "rail": sub["rail"],
                "rail_reference": sub["rail_reference"],
                "days_until_charge": days_until,
                "recommended_action": (
                    f"Cancel before {sub['next_expected']} if you no longer need "
                    f"{sub['merchant']} — after that the £{amount:,.2f} is gone for "
                    f"another year."
                ),
                "annual_saving": f"{amount:.2f}",
                "evidence": (
                    f"{sub['cadence'].title()} renewal of £{amount:,.2f} due "
                    f"{sub['next_expected']}, {days_until} days from now."
                ),
            }
        )

    return strategies


# Categories where two subscriptions genuinely do the same job, so keeping both
# is paying twice for one outcome. Entertainment is deliberately absent:
# Netflix's catalogue is not Spotify's, and telling someone to drop one because
# they have the other would be wrong.
_INTERCHANGEABLE_CATEGORIES = {"Cloud Storage"}


def _rail_remedy(sub: dict) -> str:
    """
    What thinkmoney can actually do about this subscription, by rail.
    Some types of subscriptions can be cancelled and others can only be blocked,
    but the risk is that the block happens but the you will still owe.
    """
    reference = sub["rail_reference"]
    merchant = sub["merchant"]

    if sub["rail"] == "standing_order":
        return f"I can cancel standing order {reference} for you."
    if sub["rail"] == "direct_debit":
        return f"I can cancel direct debit {reference} for you."
    return (
        f"thinkmoney cannot cancel this one — the mandate is held by {merchant}, "
        f"not by us. I can block future charges on {reference} and give you "
        f"{merchant}'s own cancellation steps. Blocking stops the payment, not "
        f"the contract, so you may still owe them."
    )


# This is an example of things that could be removed.
# We cant know if the customer is using the gym regularly enough to justify it
# but we can define the kind of things that should be considered.
_USAGE_DEPENDENT_CATEGORIES = {"Fitness"}

# Only worth interrupting the customer for a commitment of real size. Below this
# the question is noise; the gym clears it comfortably and nothing else does.
_REVIEW_THRESHOLD_ANNUAL = Decimal("400.00")


def _detect_usage_unknown(subscriptions: list[dict]) -> list[dict]:
    """Flag an expensive subscription whose usage the bank cannot see.

    thinkmoney knows only what leaves the account: amount, cadence, rail, dates.
    Whether the customer still attends the gym is provider/merchant data.
    So this strategy only detects these kind of recurring subscriptions and asks
    is never added to a total presented as money already found.
    """
    strategies = []

    for sub in subscriptions:
        if sub["category"] not in _USAGE_DEPENDENT_CATEGORIES:
            continue
        annualised = Decimal(sub["annualised_cost"])
        if annualised < _REVIEW_THRESHOLD_ANNUAL:
            continue
        strategies.append(
            {
                "strategy": "usage_unknown",
                "merchant": sub["merchant"],
                "rail": sub["rail"],
                "rail_reference": sub["rail_reference"],
                "confirmation_required": True,
                "question": f"Are you still using {sub['merchant']}?",
                "recommended_action": (
                    f"{sub['merchant']} has taken £{Decimal(sub['amount'].lstrip('-£').replace(',', '')):,.2f} "
                    f"a month since {sub['first_charged']} — £{annualised:,.2f} a year, "
                    f"your largest discretionary subscription. Are you still using it? "
                    f"I have no way of knowing that from your account. "
                    f"If you are not, {_rail_remedy(sub)}"
                ),
                "annual_saving": sub["annualised_cost"],
                "evidence": (
                    f"{sub['occurrences']} payments of {sub['amount']} between "
                    f"{sub['first_charged']} and {sub['last_charged']}. thinkmoney can "
                    f"see the payments leaving, but not whether the membership is "
                    f"being used — only you can confirm that."
                ),
            }
        )

    return strategies


def _detect_category_overlap(subscriptions: list[dict]) -> list[dict]:
    """Flag two subscriptions buying the same thing under different names.

    The cheaper of the pair is the one recommended for dropping: it is the
    smaller saving, but it is the one whose loss the customer will notice
    least, and the bigger plan already covers the need.
    """
    strategies = []
    by_category: dict[str, list[dict]] = {}

    for sub in subscriptions:
        if sub["category"] not in _INTERCHANGEABLE_CATEGORIES:
            continue
        by_category.setdefault(sub["category"], []).append(sub)

    for category, group in by_category.items():
        if len(group) < 2:
            continue
        ranked = sorted(group, key=lambda s: Decimal(s["annualised_cost"]))
        cheaper, dearer = ranked[0], ranked[-1]
        strategies.append(
            {
                "strategy": "category_overlap",
                "merchant": cheaper["merchant"],
                "overlaps_with": dearer["merchant"],
                "rail": cheaper["rail"],
                "rail_reference": cheaper["rail_reference"],
                "recommended_action": (
                    f"Keep {dearer['merchant']} and drop {cheaper['merchant']} — "
                    f"move anything stored there across first, then cancel."
                ),
                "annual_saving": cheaper["annualised_cost"],
                "evidence": (
                    f"{cheaper['merchant']} and {dearer['merchant']} are both "
                    f"{category} plans, billed separately at "
                    f"£{Decimal(cheaper['annualised_cost']):,.2f} and "
                    f"£{Decimal(dearer['annualised_cost']):,.2f} a year."
                ),
            }
        )

    return strategies


def _detect_downgrades(subscriptions: list[dict]) -> list[dict]:
    """Offer the cheapest tier of a service the customer wants to keep.

    Cancelling is not the only answer, and for a service in daily use it is
    usually the wrong one.
    """
    strategies = []

    for sub in subscriptions:
        tiers = MOCK_PLAN_TIERS.get(sub["merchant"])
        if not tiers or not tiers["alternatives"]:
            continue
        cheapest = min(tiers["alternatives"], key=lambda tier: tier["monthly_price"])
        current = Decimal(str(tiers["current_monthly_price"]))
        alternative = Decimal(str(cheapest["monthly_price"]))
        if alternative >= current:
            continue
        saving = _annualised(current - alternative, "monthly")
        strategies.append(
            {
                "strategy": "downgrade",
                "merchant": sub["merchant"],
                "rail": sub["rail"],
                "rail_reference": sub["rail_reference"],
                "current_plan": tiers["current_plan"],
                "alternative_plan": cheapest["plan"],
                "recommended_action": (
                    f"Move {sub['merchant']} from {tiers['current_plan']} to "
                    f"{cheapest['plan']} at £{alternative:,.2f} a month and keep the "
                    f"service — £{saving:,.2f} a year less."
                ),
                "annual_saving": f"{saving:.2f}",
                "evidence": (
                    f"{tiers['current_plan']} costs £{current:,.2f} a month; "
                    f"{cheapest['plan']} costs £{alternative:,.2f}. "
                    f"{cheapest['trade_off']}"
                ),
            }
        )

    return strategies


def _detect_annual_billing(subscriptions: list[dict]) -> list[dict]:
    """Offer the same plan on annual billing where that is cheaper.

    The only saving here that costs the customer nothing they currently have —
    except flexibility, which the caveat spells out.
    """
    strategies = []

    for sub in subscriptions:
        billing = MOCK_BILLING_OPTIONS.get(sub["merchant"])
        if not billing:
            continue
        monthly = Decimal(str(billing["monthly_price"]))
        annual = Decimal(str(billing["annual_price"]))
        saving = _annualised(monthly, "monthly") - annual
        if saving <= 0:
            continue
        strategies.append(
            {
                "strategy": "annual_billing",
                "merchant": sub["merchant"],
                "rail": sub["rail"],
                "rail_reference": sub["rail_reference"],
                "recommended_action": (
                    f"Switch {sub['merchant']} to annual billing at "
                    f"£{annual:,.2f} up front for the same {billing['plan']} — "
                    f"£{saving:,.2f} a year less."
                ),
                "annual_saving": f"{saving:.2f}",
                "evidence": (
                    f"£{monthly:,.2f} a month is £{_annualised(monthly, 'monthly'):,.2f} "
                    f"a year; the same plan billed annually is £{annual:,.2f}."
                ),
                "caveat": billing["notes"],
            }
        )

    return strategies


def _service_keys(subscriptions: list[dict], duplicates: list[dict]) -> dict[str, str]:
    """Map each merchant to the service it belongs to.

    Usually a merchant is its own service. The exception is a duplicate pair:
    the Netflix direct debit and the shared-Netflix standing order are one
    service bought twice, so a saving banked on either side is the saving for
    both. Without this the report would offer to cancel the duplicate *and*
    downgrade the copy that gets kept, and count both.
    """
    keys = {sub["merchant"]: sub["merchant"] for sub in subscriptions}
    for duplicate in duplicates:
        keys[duplicate["merchant"]] = duplicate["duplicate_of"]
    return keys


def _savings(subscriptions: list[dict]) -> dict:
    """Every applicable saving, largest first, one per merchant.

    A merchant can qualify for more than one strategy — Spotify has both risen
    in price and has a cheaper tier — but only one of them can actually be
    taken, so the total keeps the largest and drops the rest. Without that the
    headline would promise savings the customer cannot collect twice.
    """
    # Essential commitments are deliberately out of scope: nobody is served by
    # a report suggesting they save money by not paying rent.
    candidates = [s for s in subscriptions if not s["essential"]]

    duplicates = _detect_duplicates(candidates)
    strategies = (
        duplicates
        + _detect_converted_trials(candidates)
        + _detect_price_rises(candidates)
        + _detect_upcoming_renewals(candidates)
        + _detect_usage_unknown(candidates)
        + _detect_category_overlap(candidates)
        + _detect_downgrades(candidates)
        + _detect_annual_billing(candidates)
    )

    service_keys = _service_keys(candidates, duplicates)
    for strategy in strategies:
        strategy["service"] = service_keys.get(
            strategy["merchant"], strategy["merchant"]
        )

    best_per_service: dict[str, dict] = {}
    for strategy in strategies:
        incumbent = best_per_service.get(strategy["service"])
        if incumbent is None or Decimal(strategy["annual_saving"]) > Decimal(
            incumbent["annual_saving"]
        ):
            best_per_service[strategy["service"]] = strategy

    kept = sorted(
        best_per_service.values(),
        key=lambda s: (-Decimal(s["annual_saving"]), s["merchant"]),
    )
    # Everything else is still real and still worth showing — it just cannot be
    # added to a total the customer is meant to be able to collect.
    superseded = sorted(
        (
            {**s, "superseded_by": best_per_service[s["service"]]["strategy"]}
            for s in strategies
            if s is not best_per_service[s["service"]]
        ),
        key=lambda s: (-Decimal(s["annual_saving"]), s["merchant"]),
    )
    # Two tiers, because they are two different kinds of claim. "Identified"
    # is provable from the account: a service billed twice, a trial that
    # converted, a renewal dated in the future. "Potential" depends on an
    # answer only the customer has — thinkmoney can see a gym payment leave,
    # never a gym visit. Adding the two together would present a question as
    # a finding.
    identified = sum(
        (
            Decimal(s["annual_saving"])
            for s in kept
            if not s.get("confirmation_required")
        ),
        Decimal("0"),
    )
    potential = sum(
        (Decimal(s["annual_saving"]) for s in kept if s.get("confirmation_required")),
        Decimal("0"),
    )

    return {
        "strategies": kept,
        "strategy_count": len(kept),
        "superseded": superseded,
        "identified_saving": f"{identified:.2f}",
        "potential_saving": f"{potential:.2f}",
        "combined_saving": f"{identified + potential:.2f}",
        "note": "Each service is counted once, under its largest saving, so the "
        "totals are money that can actually be taken. 'identified' is "
        "provable from the account alone. 'potential' depends on the "
        "customer answering a question thinkmoney cannot answer for "
        "them, and must never be presented as money already found. "
        "Strategies listed under 'superseded' are real alternatives to "
        "the one counted, not additional savings.",
    }


def _totals(subscriptions: list[dict]) -> dict:
    """Headline figures.

    Monthly total counts monthly-cadence subscriptions only; annualising every
    cadence and dividing back down would quietly fold the annual domain renewal
    into a "monthly" number the customer would never recognise on a statement.
    """
    discretionary = [s for s in subscriptions if not s["essential"]]

    monthly = sum(
        (
            _parse_amount(s["amount"])
            for s in discretionary
            if s["cadence"] == "monthly"
        ),
        Decimal("0"),
    )
    annualised = sum(
        (Decimal(s["annualised_cost"]) for s in discretionary), Decimal("0")
    )
    essential_monthly = sum(
        (
            _parse_amount(s["amount"])
            for s in subscriptions
            if s["essential"] and s["cadence"] == "monthly"
        ),
        Decimal("0"),
    )

    return {
        "subscription_count": len(subscriptions),
        "discretionary_count": len(discretionary),
        "monthly_total": f"{monthly:.2f}",
        "annualised_total": f"{annualised:.2f}",
        "essential_monthly_total": f"{essential_monthly:.2f}",
        "note": "Totals cover discretionary subscriptions only. Essential "
        "commitments such as rent are listed but excluded.",
    }


@tool
def find_recurring_payments(user_id: str) -> str:
    """Find every recurring payment on a customer's account across all rails.

    Groups the transaction history by merchant and cross-references the
    direct-debit and standing-order mandates, so card, direct-debit and
    standing-order subscriptions all come back in one call. Returns each
    subscription with its cadence, rail, next expected charge and annualised
    cost, plus headline monthly and annual totals and a savings block naming
    every way the customer could pay less.

    Args:
        user_id: The customer's user ID.
    """
    fixtures = _fixtures_by_reference()

    subscriptions = _detect_from_transactions(fixtures)
    seen = {sub["rail_reference"] for sub in subscriptions}
    subscriptions += _detect_from_fixtures(fixtures, seen)

    # Most expensive first — that is the order the customer cares about.
    subscriptions.sort(key=lambda s: Decimal(s["annualised_cost"]), reverse=True)
    subscriptions.sort(key=lambda s: s["essential"])

    report = {
        "user_id": user_id,
        # Date, not timestamp: the corpus is dated to the day, so a wall-clock
        # time would be the only thing that differed between two identical runs.
        "generated_at": date.today().isoformat(),
        "subscriptions": subscriptions,
        "totals": _totals(subscriptions),
        "savings": _savings(subscriptions),
    }

    return json.dumps(report)
