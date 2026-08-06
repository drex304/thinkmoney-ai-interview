"""Tests for the plan-tier and billing-option fixtures.

These two fixtures are the evidence behind the saving strategies the transaction
corpus cannot prove on its own: "you could downgrade" and "annual billing is
cheaper". Both are facts about what the merchant offers, which a bank can hold.

A third fixture used to live here — MOCK_MERCHANT_ENGAGEMENT, carrying gym
check-ins and streaming counts — and was deleted. No e-money provider holds
merchant-side usage data, and while it existed the agent asserted "you have not
used FitLife Gym in 4 months" as fact. TestNoUsageFixture below is the guard
against it coming back.
"""

import re
from datetime import date, datetime
from pathlib import Path

import src.agents.subscriptions.data as subscription_data
from src.agents.subscriptions.data import (
    MOCK_BILLING_OPTIONS,
    MOCK_PLAN_TIERS,
)
from src.tools.transactions import MOCK_TRANSACTIONS


def _as_date(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date()


def _days_since(value):
    return (date.today() - _as_date(value)).days


CORPUS_MERCHANTS = {t["description"] for t in MOCK_TRANSACTIONS}
MODULE_PATH = str(subscription_data.__file__)


class TestRelativeDating:
    def test_source_file_contains_no_hardcoded_date_strings(self):
        source = Path(MODULE_PATH).read_text()
        assert not re.search(r'"\d{4}-\d{2}-\d{2}', source)


class TestNoUsageFixture:
    """The bank must not hold, or invent, merchant-side usage data.

    Deleting MOCK_MERCHANT_ENGAGEMENT was the fix for the agent claiming it knew
    whether a gym membership was being used. These tests make the deletion
    permanent: reintroducing usage data should fail loudly here rather than
    quietly restore a confident lie.
    """

    def test_the_engagement_fixture_is_gone(self):
        assert not hasattr(subscription_data, "MOCK_MERCHANT_ENGAGEMENT")

    def test_no_usage_signal_is_exported_under_another_name(self):
        exported = [n for n in dir(subscription_data) if n.isupper()]
        for name in exported:
            lowered = name.lower()
            for banned in ("engagement", "usage", "interaction", "visit", "checkin"):
                assert banned not in lowered, (
                    f"{name} looks like merchant-side usage data, which a bank "
                    f"cannot observe"
                )

    def test_the_module_declares_no_usage_vocabulary(self):
        source = Path(MODULE_PATH).read_text()
        body = source.split('"""', 2)[
            -1
        ]  # skip the module docstring, which explains the removal
        for banned in ("last_interaction", "interactions_last_90_days", "check-in"):
            assert banned not in body, f"{banned!r} is merchant-side data"


class TestPlanTiers:
    def test_spotify_and_netflix_both_offer_cheaper_alternatives(self):
        assert {"Spotify", "Netflix"} <= set(MOCK_PLAN_TIERS)

    def test_every_alternative_is_cheaper_than_the_current_plan(self):
        for merchant, entry in MOCK_PLAN_TIERS.items():
            assert entry["merchant"] == merchant
            assert entry["alternatives"], f"{merchant} has no alternatives"
            for alternative in entry["alternatives"]:
                assert alternative["monthly_price"] < entry["current_monthly_price"]
                assert alternative["trade_off"]

    def test_spotify_premium_can_drop_to_free(self):
        spotify = MOCK_PLAN_TIERS["Spotify"]
        assert spotify["current_plan"] == "Premium Individual"
        assert spotify["current_monthly_price"] == 12.99
        free = next(a for a in spotify["alternatives"] if a["plan"] == "Free")
        assert free["monthly_price"] == 0.00
        # The headline saving in the coverage contract: £12.99 x 12.
        assert (
            round((spotify["current_monthly_price"] - free["monthly_price"]) * 12, 2)
            == 155.88
        )

    def test_netflix_standard_can_drop_to_basic(self):
        netflix = MOCK_PLAN_TIERS["Netflix"]
        assert netflix["current_plan"] == "Standard"
        assert netflix["current_monthly_price"] == 15.99
        basic = next(a for a in netflix["alternatives"] if a["plan"] == "Basic")
        assert basic["monthly_price"] < 15.99

    def test_current_prices_match_the_transaction_corpus(self):
        charged = {
            "Spotify": 12.99,  # after the price rise
            "Netflix": 15.99,
        }
        for merchant, amount in charged.items():
            assert MOCK_PLAN_TIERS[merchant]["current_monthly_price"] == amount


class TestBillingOptions:
    def test_adobe_annual_billing_is_cheaper_by_roughly_two_months(self):
        adobe = MOCK_BILLING_OPTIONS["Adobe Creative Cloud"]
        assert adobe["monthly_price"] == 9.99
        yearly_on_monthly = round(adobe["monthly_price"] * 12, 2)
        assert yearly_on_monthly == 119.88
        saving = round(yearly_on_monthly - adobe["annual_price"], 2)
        assert saving == 24.00
        assert 1.5 <= saving / adobe["monthly_price"] <= 2.5

    def test_annual_saving_is_recorded_and_consistent(self):
        for merchant, entry in MOCK_BILLING_OPTIONS.items():
            assert entry["merchant"] == merchant
            assert entry["annual_price"] < round(entry["monthly_price"] * 12, 2)
            assert entry["annual_saving"] == round(
                entry["monthly_price"] * 12 - entry["annual_price"], 2
            )

    def test_every_merchant_is_present_in_the_transaction_corpus(self):
        assert set(MOCK_BILLING_OPTIONS) <= CORPUS_MERCHANTS


class TestFixturesLiveOutsideAgentCode:
    def test_fixtures_are_importable_from_a_tools_data_module(self):
        assert MODULE_PATH.replace("\\", "/").endswith(
            "src/agents/subscriptions/data.py"
        )


class TestDatesStayCoherent:
    def test_plan_and_billing_prices_match_what_the_corpus_charges(self):
        # The fixtures describe alternatives to what is actually being paid, so a
        # drifting corpus price would silently invalidate every saving figure.
        spotify = MOCK_PLAN_TIERS["Spotify"]["current_monthly_price"]
        adobe = MOCK_BILLING_OPTIONS["Adobe Creative Cloud"]["monthly_price"]
        charged = {t["description"]: t["amount"] for t in MOCK_TRANSACTIONS}
        assert f"{spotify:.2f}" in charged["Spotify"]
        assert f"{adobe:.2f}" in charged["Adobe Creative Cloud"]
