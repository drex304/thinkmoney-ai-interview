"""Mock data that would help with saving stratergies.

Two questions a statement alone cannot answer:

* "Could I keep it and pay less?"    -> MOCK_PLAN_TIERS
* "Would annual billing be cheaper?" -> MOCK_BILLING_OPTIONS

Both are facts about what the merchant offers, which a bank could have access too.

The questions of: "am I actually using this?" — deliberately has no fixture.
The usage_unknown strategy in subscriptions.py now states the cost, which is a fact, and asks the question,
which only the customer can answer.
"""

from datetime import date, timedelta

_TODAY = date.today()


def _days_ago(days: int) -> str:
    """ISO date `days` days before today."""
    return (_TODAY - timedelta(days=days)).isoformat()


# Cheaper tiers of the same service — the evidence for recommending a downgrade instead
# of a cancellation. `current_monthly_price` matches what the corpus actually charges.
MOCK_PLAN_TIERS = {
    "Spotify": {
        "merchant": "Spotify",
        "current_plan": "Premium Individual",
        "current_monthly_price": 12.99,
        "alternatives": [
            {
                "plan": "Free",
                "monthly_price": 0.00,
                "trade_off": "Adverts between tracks and no offline downloads.",
            },
            {
                "plan": "Premium Duo",
                "monthly_price": 8.75,
                "trade_off": "Priced per person — needs a second listener at the same address.",
            },
        ],
    },
    "Netflix": {
        "merchant": "Netflix",
        "current_plan": "Standard",
        "current_monthly_price": 15.99,
        "alternatives": [
            {
                "plan": "Standard with adverts",
                "monthly_price": 4.99,
                "trade_off": "Adverts, and a few titles are unavailable.",
            },
            {
                "plan": "Basic",
                "monthly_price": 7.99,
                "trade_off": "One screen at a time and no 1080p.",
            },
        ],
    },
}


# Monthly versus annual pricing for the same plan — a saving that requires no change
# to what the customer actually gets.
MOCK_BILLING_OPTIONS = {
    "Adobe Creative Cloud": {
        "merchant": "Adobe Creative Cloud",
        "plan": "Photography Plan",
        "monthly_price": 9.99,
        "annual_price": 95.88,
        "annual_equivalent_monthly": 7.99,
        "annual_saving": 24.00,
        "notes": (
            "Paid annually up front. Roughly two months free versus rolling monthly, "
            "but the year is committed — Adobe charges 50% of the remaining balance to "
            "leave early."
        ),
    },
}
