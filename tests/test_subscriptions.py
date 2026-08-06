"""Tests for deterministic recurring-payment detection."""

import json
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from src.tools.payments import MOCK_DIRECT_DEBITS, MOCK_STANDING_ORDERS
from src.agents.cancellation_research.directory import CANCELLATION_DIRECTORY

from src.agents.cancellation_research.guide import (
    _search_available,
    _search_web,
    find_cancellation_guide,
)
from src.agents.subscriptions import find_recurring_payments
from src.tools.cards import block_merchant_on_card
from src.tools.payments import cancel_direct_debit


@pytest.fixture
def report():
    return json.loads(find_recurring_payments.invoke({"user_id": "USR-12345"}))


def _unguarded(tool, **kwargs):
    """Run a write tool's body, skipping the confirmation gate.

    These tests are about what the tool does once it runs. That it does not run
    without an explicit yes is asserted in `tests/test_confirmation_gate.py:1`,
    against the real graph.
    """
    return json.loads(tool.func.__wrapped__(**kwargs))


def _by_merchant(report, merchant):
    for sub in report["subscriptions"]:
        if sub["merchant"] == merchant:
            return sub
    raise AssertionError(f"{merchant} was not detected as a subscription")


def _merchants(report):
    return {sub["merchant"] for sub in report["subscriptions"]}


class TestCadenceDetection:
    def test_monthly_cadence_inferred_from_three_charges(self, report):
        netflix = _by_merchant(report, "Netflix")
        assert netflix["cadence"] == "monthly"
        assert netflix["occurrences"] == 3

    def test_annual_cadence_inferred_for_the_domain_renewal(self, report):
        domain = _by_merchant(report, "Namecheap Domain Renewal")
        assert domain["cadence"] == "annual"
        assert domain["occurrences"] == 2

    def test_every_recurring_merchant_in_the_corpus_is_detected(self, report):
        assert _merchants(report) == {
            "Netflix",
            "James Wilson - Shared Netflix",
            "Spotify",
            "Adobe Creative Cloud",
            "FitLife Gym",
            "Apple iCloud+",
            "Google One",
            "Namecheap Domain Renewal",
            "Landlord - Premier Properties",
        }


class TestNegativeCases:
    @pytest.mark.parametrize(
        "merchant",
        ["Acme Corp - Salary", "ATM Withdrawal - Barclays", "Tesco Express"],
    )
    def test_one_offs_are_not_subscriptions(self, report, merchant):
        assert merchant not in _merchants(report)

    def test_no_single_occurrence_merchant_is_ever_classified(self, report):
        assert all(sub["occurrences"] >= 2 for sub in report["subscriptions"])

    def test_incoming_money_is_never_a_subscription(self, report):
        assert all(not sub["amount"].startswith("+") for sub in report["subscriptions"])


class TestSubscriptionShape:
    REQUIRED_FIELDS = [
        "merchant",
        "category",
        "amount",
        "cadence",
        "rail",
        "rail_reference",
        "occurrences",
        "last_charged",
        "next_expected",
        "annualised_cost",
    ]

    def test_every_result_carries_all_required_fields(self, report):
        for sub in report["subscriptions"]:
            for field in self.REQUIRED_FIELDS:
                assert sub[field], f"{sub['merchant']} is missing {field}"

    def test_every_result_carries_a_rail_and_a_category(self, report):
        for sub in report["subscriptions"]:
            assert sub["rail"] in {"direct_debit", "standing_order", "card_on_file"}
            assert sub["category"]

    def test_rail_reference_matches_the_rail(self, report):
        expected_prefix = {
            "direct_debit": "DD-",
            "standing_order": "SO-",
            "card_on_file": "CARD-",
        }
        for sub in report["subscriptions"]:
            assert sub["rail_reference"].startswith(expected_prefix[sub["rail"]])

    def test_amount_is_the_most_recent_charge(self, report):
        # Spotify rose from £11.99 to £12.99; the current price is what the
        # customer is actually paying.
        assert _by_merchant(report, "Spotify")["amount"] == "-£12.99"
        assert _by_merchant(report, "Adobe Creative Cloud")["amount"] == "-£9.99"

    def test_next_expected_is_one_cadence_after_the_last_charge(self, report):
        netflix = _by_merchant(report, "Netflix")
        last = date.fromisoformat(netflix["last_charged"])
        assert date.fromisoformat(netflix["next_expected"]) == last + timedelta(days=30)

    def test_annualised_cost_multiplies_the_cadence(self, report):
        assert _by_merchant(report, "Netflix")["annualised_cost"] == "191.88"
        assert (
            _by_merchant(report, "Namecheap Domain Renewal")["annualised_cost"]
            == "79.99"
        )


class TestRailCoverage:
    def test_all_three_subscription_rails_are_discoverable_in_one_call(self, report):
        assert {sub["rail"] for sub in report["subscriptions"]} == {
            "direct_debit",
            "standing_order",
            "card_on_file",
        }

    def test_every_active_direct_debit_fixture_is_accounted_for(self, report):
        references = {sub["rail_reference"] for sub in report["subscriptions"]}
        for mandate in MOCK_DIRECT_DEBITS:
            assert mandate["mandate_id"] in references

    def test_every_active_standing_order_fixture_is_accounted_for(self, report):
        references = {sub["rail_reference"] for sub in report["subscriptions"]}
        for order in MOCK_STANDING_ORDERS:
            assert order["order_id"] in references

    def test_fixture_supplies_the_authoritative_next_payment_date(self, report):
        domain = _by_merchant(report, "Namecheap Domain Renewal")
        mandate = next(m for m in MOCK_DIRECT_DEBITS if m["mandate_id"] == "DD-4472")
        assert domain["next_expected"] == mandate["next_payment"]
        assert domain["next_expected_source"] == "direct_debit_mandate"

    def test_a_mandate_with_no_transaction_history_is_still_reported(self, monkeypatch):
        import src.agents.subscriptions.detection as subscriptions_module

        unseen = {
            "mandate_id": "DD-9999",
            "merchant": "Hidden Insurer",
            "category": "Insurance",
            "amount": "£18.50",
            "frequency": "monthly",
            "next_payment": (date.today() + timedelta(days=9)).isoformat(),
            "status": "active",
        }
        monkeypatch.setattr(
            subscriptions_module,
            "MOCK_DIRECT_DEBITS",
            list(MOCK_DIRECT_DEBITS) + [unseen],
        )
        patched = json.loads(find_recurring_payments.invoke({"user_id": "USR-12345"}))
        hidden = _by_merchant(patched, "Hidden Insurer")
        assert hidden["rail"] == "direct_debit"
        assert hidden["rail_reference"] == "DD-9999"
        assert hidden["occurrences"] == 0
        assert hidden["source"] == "mandate_only"


class TestEssentials:
    def test_rent_is_detected_but_flagged_as_essential(self, report):
        rent = _by_merchant(report, "Landlord - Premier Properties")
        assert rent["essential"] is True

    def test_discretionary_subscriptions_are_not_flagged_essential(self, report):
        assert _by_merchant(report, "Netflix")["essential"] is False


class TestHeadlineTotals:
    def test_monthly_total_covers_the_discretionary_monthly_subscriptions(self, report):
        assert report["totals"]["monthly_total"] == "118.95"

    def test_annualised_total_is_monthly_times_twelve_plus_the_annual_renewal(
        self, report
    ):
        assert report["totals"]["annualised_total"] == "1507.39"
        assert float(report["totals"]["annualised_total"]) == pytest.approx(
            118.95 * 12 + 79.99
        )

    def test_essential_commitments_are_reported_separately(self, report):
        assert report["totals"]["essential_monthly_total"] == "1200.00"

    def test_totals_report_the_subscription_count(self, report):
        assert report["totals"]["subscription_count"] == len(report["subscriptions"])


class TestDeterminism:
    def test_the_tool_makes_no_llm_call(self):
        from pathlib import Path

        import src.agents.subscriptions.detection as subscriptions_module

        source = Path(subscriptions_module.__file__).read_text()
        for banned in (
            "ChatAnthropic",
            "invoke_model",
            "get_llm",
            "langchain_anthropic",
        ):
            assert banned not in source

    def test_repeated_calls_return_identical_results(self):
        first = find_recurring_payments.invoke({"user_id": "USR-12345"})
        second = find_recurring_payments.invoke({"user_id": "USR-12345"})
        assert first == second

    def test_the_report_echoes_the_user_id(self, report):
        assert report["user_id"] == "USR-12345"

    def test_generated_at_is_todays_date(self, report):
        assert report["generated_at"].startswith(date.today().isoformat())


def test_subscriptions_are_ordered_by_annual_cost_descending(report):
    costs = [
        float(sub["annualised_cost"])
        for sub in report["subscriptions"]
        if not sub["essential"]
    ]
    assert costs == sorted(costs, reverse=True)


def test_last_charged_is_a_plain_iso_date(report):
    for sub in report["subscriptions"]:
        if sub["occurrences"]:
            assert datetime.fromisoformat(sub["last_charged"]).date() <= date.today()


def _strategies(report, name):
    # Superseded strategies are still detected and still reported — they are
    # only left out of the total so a merchant is not counted twice.
    detected = report["savings"]["strategies"] + report["savings"]["superseded"]
    return [s for s in detected if s["strategy"] == name]


def _by_strategy_merchant(report, name, merchant):
    for strategy in _strategies(report, name):
        if strategy["merchant"] == merchant:
            return strategy
    raise AssertionError(f"no {name} strategy for {merchant}")


def _only_strategy(report, name):
    matches = _strategies(report, name)
    assert len(matches) == 1, f"expected exactly one {name}, got {len(matches)}"
    return matches[0]


class TestSavingsBlockShape:
    def test_the_report_carries_a_savings_block(self, report):
        assert report["savings"]["strategies"]

    def test_every_strategy_names_a_merchant_an_action_and_a_saving(self, report):
        for strategy in report["savings"]["strategies"]:
            assert strategy["merchant"]
            assert strategy["recommended_action"]
            assert Decimal(strategy["annual_saving"]) > 0
            assert strategy["evidence"]

    def test_every_strategy_points_at_a_detected_subscription(self, report):
        merchants = _merchants(report)
        for strategy in report["savings"]["strategies"]:
            assert strategy["merchant"] in merchants

    def test_essential_commitments_are_never_targeted(self, report):
        essential = {
            sub["merchant"] for sub in report["subscriptions"] if sub["essential"]
        }
        for strategy in report["savings"]["strategies"]:
            assert strategy["merchant"] not in essential

    def test_strategies_are_ordered_by_saving_descending(self, report):
        savings = [Decimal(s["annual_saving"]) for s in report["savings"]["strategies"]]
        assert savings == sorted(savings, reverse=True)


class TestDuplicateStrategy:
    def test_the_two_netflix_payments_are_flagged_as_one_service_paid_twice(
        self, report
    ):
        duplicate = _only_strategy(report, "duplicate")
        assert duplicate["merchant"] == "James Wilson - Shared Netflix"
        assert duplicate["duplicate_of"] == "Netflix"

    def test_the_saving_is_the_more_expensive_of_the_two(self, report):
        assert _only_strategy(report, "duplicate")["annual_saving"] == "300.00"

    def test_the_cheaper_netflix_payment_is_the_one_kept(self, report):
        duplicate = _only_strategy(report, "duplicate")
        assert duplicate["rail"] == "standing_order"
        assert duplicate["rail_reference"] == "SO-102"

    def test_two_unrelated_merchants_are_not_flagged_as_duplicates(self, report):
        flagged = {s["merchant"] for s in _strategies(report, "duplicate")}
        assert "Spotify" not in flagged
        assert "FitLife Gym" not in flagged


class TestConvertedTrialStrategy:
    def test_the_adobe_free_trial_conversion_is_flagged(self, report):
        trial = _only_strategy(report, "converted_trial")
        assert trial["merchant"] == "Adobe Creative Cloud"

    def test_the_saving_is_a_full_year_at_the_post_trial_price(self, report):
        assert _only_strategy(report, "converted_trial")["annual_saving"] == "119.88"

    def test_the_evidence_cites_the_zero_charge_and_the_first_real_charge(self, report):
        trial = _only_strategy(report, "converted_trial")
        assert "0.00" in trial["evidence"]
        assert "9.99" in trial["evidence"]

    def test_a_merchant_that_never_charged_zero_is_not_a_converted_trial(self, report):
        flagged = {s["merchant"] for s in _strategies(report, "converted_trial")}
        assert "Netflix" not in flagged
        assert "Google One" not in flagged


class TestPriceRiseStrategy:
    def test_the_spotify_price_rise_is_flagged(self, report):
        rise = _only_strategy(report, "price_rise")
        assert rise["merchant"] == "Spotify"

    def test_the_saving_is_the_annualised_delta_not_the_whole_price(self, report):
        assert _only_strategy(report, "price_rise")["annual_saving"] == "12.00"

    def test_the_evidence_carries_both_prices(self, report):
        evidence = _only_strategy(report, "price_rise")["evidence"]
        assert "11.99" in evidence
        assert "12.99" in evidence

    def test_a_flat_priced_subscription_is_not_flagged(self, report):
        flagged = {s["merchant"] for s in _strategies(report, "price_rise")}
        assert "Netflix" not in flagged
        assert "FitLife Gym" not in flagged

    def test_the_adobe_trial_conversion_is_not_reported_as_a_price_rise(self, report):
        # £0.00 -> £9.99 is a trial ending, and saying "your price went up by
        # £9.99" would misdescribe it.
        flagged = {s["merchant"] for s in _strategies(report, "price_rise")}
        assert "Adobe Creative Cloud" not in flagged


class TestUpcomingRenewalStrategy:
    def test_the_domain_renewal_due_within_thirty_days_is_flagged(self, report):
        renewal = _only_strategy(report, "upcoming_renewal")
        assert renewal["merchant"] == "Namecheap Domain Renewal"

    def test_the_whole_renewal_amount_is_avoidable(self, report):
        assert _only_strategy(report, "upcoming_renewal")["annual_saving"] == "79.99"

    def test_the_evidence_gives_the_renewal_date_and_days_remaining(self, report):
        renewal = _only_strategy(report, "upcoming_renewal")
        domain = _by_merchant(report, "Namecheap Domain Renewal")
        assert domain["next_expected"] in renewal["evidence"]
        assert (
            renewal["days_until_charge"]
            == (date.fromisoformat(domain["next_expected"]) - date.today()).days
        )

    def test_a_monthly_subscription_is_not_an_upcoming_renewal(self, report):
        # Every monthly subscription bills within 30 days by definition, so
        # flagging them would drown the one renewal that is worth warning about.
        flagged = {s["merchant"] for s in _strategies(report, "upcoming_renewal")}
        assert flagged == {"Namecheap Domain Renewal"}


class TestSavingsTotal:
    def test_the_combined_total_sums_the_strategies(self, report):
        total = sum(
            (Decimal(s["annual_saving"]) for s in report["savings"]["strategies"]),
            Decimal("0"),
        )
        assert report["savings"]["combined_saving"] == f"{total:.2f}"

    def test_identified_and_potential_split_on_confirmation_required(self, report):
        savings = report["savings"]
        identified = sum(
            (
                Decimal(s["annual_saving"])
                for s in savings["strategies"]
                if not s.get("confirmation_required")
            ),
            Decimal("0"),
        )
        potential = sum(
            (
                Decimal(s["annual_saving"])
                for s in savings["strategies"]
                if s.get("confirmation_required")
            ),
            Decimal("0"),
        )
        assert savings["identified_saving"] == f"{identified:.2f}"
        assert savings["potential_saving"] == f"{potential:.2f}"
        assert identified + potential == Decimal(savings["combined_saving"])

    def test_the_gym_is_not_counted_as_money_already_found(self, report):
        # £528 depends on an answer the bank does not have, so it must sit
        # outside the identified figure however tempting the headline is.
        savings = report["savings"]
        assert savings["identified_saving"] == "691.63"
        assert savings["potential_saving"] == "528.00"

    def test_no_merchant_is_counted_under_two_strategies(self, report):
        merchants = [s["merchant"] for s in report["savings"]["strategies"]]
        assert len(merchants) == len(set(merchants))

    def test_the_transaction_derived_strategies_are_all_present(self, report):
        # Detected, not necessarily counted: Spotify's price rise is real but
        # its downgrade saves more, so only the downgrade banks the money.
        detected = {
            s["strategy"]
            for s in report["savings"]["strategies"] + report["savings"]["superseded"]
        }
        assert detected >= {
            "duplicate",
            "converted_trial",
            "price_rise",
            "upcoming_renewal",
        }


class TestUsageUnknownStrategy:
    """The bank sees payments, never usage.

    thinkmoney can prove £44 leaves the account every month. It has no way to
    know whether Sarah went to the gym — that is merchant-side data no e-money
    provider holds. So this strategy must state the cost and ask the question,
    never assert disuse.
    """

    def test_the_largest_usage_dependent_subscription_is_raised(self, report):
        strategy = _only_strategy(report, "usage_unknown")
        assert strategy["merchant"] == "FitLife Gym"

    def test_the_whole_subscription_is_avoidable_if_it_is_unused(self, report):
        assert _only_strategy(report, "usage_unknown")["annual_saving"] == "528.00"

    def test_it_asks_rather_than_asserts(self, report):
        strategy = _only_strategy(report, "usage_unknown")
        assert strategy["question"] == "Are you still using FitLife Gym?"
        assert "Are you still using it?" in strategy["recommended_action"]

    def test_it_never_claims_to_know_whether_the_service_was_used(self, report):
        strategy = _only_strategy(report, "usage_unknown")
        prose = f"{strategy['recommended_action']} {strategy['evidence']}"
        for claim in (
            "you have not used",
            "you haven't used",
            "usage_unknown",
            "idle",
            "last interaction",
            "check-in",
        ):
            assert claim not in prose.lower(), f"asserts unknowable usage: {claim!r}"

    def test_it_says_plainly_that_the_bank_cannot_know(self, report):
        strategy = _only_strategy(report, "usage_unknown")
        assert "no way of knowing" in strategy["recommended_action"]
        assert "only you can confirm" in strategy["evidence"]

    def test_the_evidence_cites_only_bank_visible_facts(self, report):
        # Payment count and dates are on the statement; usage is not.
        evidence = _only_strategy(report, "usage_unknown")["evidence"]
        assert "3 payments" in evidence
        assert "-£44.00" in evidence

    def test_the_saving_is_flagged_as_needing_confirmation(self, report):
        assert _only_strategy(report, "usage_unknown")["confirmation_required"] is True

    def test_cheaper_subscriptions_are_not_queried(self, report):
        # Interrupting someone about £12.99 is noise, not service.
        flagged = {s["merchant"] for s in _strategies(report, "usage_unknown")}
        assert "Netflix" not in flagged
        assert "Spotify" not in flagged
        assert "Google One" not in flagged

    def test_no_other_strategy_requires_confirmation(self, report):
        # Everything else is provable from the account alone.
        needing = {
            s["strategy"]
            for s in report["savings"]["strategies"]
            if s.get("confirmation_required")
        }
        assert needing == {"usage_unknown"}


class TestCategoryOverlapStrategy:
    def test_the_two_cloud_storage_plans_are_flagged_as_overlapping(self, report):
        overlap = _only_strategy(report, "category_overlap")
        assert {overlap["merchant"], overlap["overlaps_with"]} == {
            "Apple iCloud+",
            "Google One",
        }

    def test_the_cheaper_to_lose_plan_is_the_one_recommended_for_dropping(self, report):
        overlap = _only_strategy(report, "category_overlap")
        assert overlap["merchant"] == "Apple iCloud+"
        assert overlap["annual_saving"] == "35.88"

    def test_the_evidence_names_the_shared_category(self, report):
        assert "Cloud Storage" in _only_strategy(report, "category_overlap")["evidence"]

    def test_subscriptions_in_different_categories_are_not_an_overlap(self, report):
        # Adobe (Software) and FitLife (Fitness) buy unrelated things.
        flagged = {s["merchant"] for s in _strategies(report, "category_overlap")}
        assert "Adobe Creative Cloud" not in flagged
        assert "FitLife Gym" not in flagged


class TestDowngradeStrategy:
    def test_spotify_is_offered_a_cheaper_tier_instead_of_cancellation(self, report):
        downgrade = _by_strategy_merchant(report, "downgrade", "Spotify")
        assert downgrade["current_plan"] == "Premium Individual"
        assert downgrade["alternative_plan"] == "Free"

    def test_the_saving_is_the_full_gap_between_the_tiers(self, report):
        downgrade = _by_strategy_merchant(report, "downgrade", "Spotify")
        assert downgrade["annual_saving"] == "155.88"

    def test_the_action_frames_the_downgrade_as_keeping_the_service(self, report):
        downgrade = _by_strategy_merchant(report, "downgrade", "Spotify")
        assert "cancel" not in downgrade["recommended_action"].lower()

    def test_a_merchant_with_no_cheaper_tier_is_not_offered_a_downgrade(self, report):
        flagged = {s["merchant"] for s in _strategies(report, "downgrade")}
        assert "FitLife Gym" not in flagged
        assert "Apple iCloud+" not in flagged


class TestAnnualBillingStrategy:
    def test_adobe_is_offered_the_annual_plan(self, report):
        switch = _by_strategy_merchant(report, "annual_billing", "Adobe Creative Cloud")
        assert switch["annual_saving"] == "24.00"

    def test_the_evidence_carries_both_prices(self, report):
        switch = _by_strategy_merchant(report, "annual_billing", "Adobe Creative Cloud")
        assert "9.99" in switch["evidence"]
        assert "95.88" in switch["evidence"]

    def test_the_commitment_is_disclosed(self, report):
        # Annual billing is cheaper but locks the customer in — saying so is the
        # difference between advice and a sales pitch.
        switch = _by_strategy_merchant(report, "annual_billing", "Adobe Creative Cloud")
        assert "50%" in switch["caveat"]

    def test_a_merchant_with_no_annual_option_is_not_offered_one(self, report):
        flagged = {s["merchant"] for s in _strategies(report, "annual_billing")}
        assert flagged == {"Adobe Creative Cloud"}


class TestTotalIdentifiableSaving:
    def test_the_total_is_the_pinned_figure(self, report):
        assert report["savings"]["combined_saving"] == "1219.63"

    def test_the_total_is_the_sum_of_the_counted_strategies(self, report):
        counted = sum(
            (Decimal(s["annual_saving"]) for s in report["savings"]["strategies"]),
            Decimal("0"),
        )
        assert report["savings"]["combined_saving"] == f"{counted:.2f}"

    def test_every_strategy_kind_is_represented(self, report):
        detected = {
            s["strategy"]
            for s in report["savings"]["strategies"] + report["savings"]["superseded"]
        }
        assert detected == {
            "duplicate",
            "converted_trial",
            "price_rise",
            "upcoming_renewal",
            "usage_unknown",
            "category_overlap",
            "downgrade",
            "annual_billing",
        }

    def test_a_superseded_strategy_never_counts_towards_the_total(self, report):
        counted = {s["service"] for s in report["savings"]["strategies"]}
        for strategy in report["savings"]["superseded"]:
            assert strategy["superseded_by"]
            assert strategy["service"] in counted

    def test_spotify_banks_its_downgrade_not_its_price_rise(self, report):
        # Both are real, but the customer can only take one of them.
        counted = {
            s["merchant"]: s["strategy"] for s in report["savings"]["strategies"]
        }
        assert counted["Spotify"] == "downgrade"
        assert "price_rise" in {s["strategy"] for s in report["savings"]["superseded"]}

    def test_adobe_banks_its_trial_conversion_not_its_annual_switch(self, report):
        counted = {
            s["merchant"]: s["strategy"] for s in report["savings"]["strategies"]
        }
        assert counted["Adobe Creative Cloud"] == "converted_trial"

    def test_the_two_sides_of_a_duplicate_pair_bank_one_saving_between_them(
        self, report
    ):
        # Netflix and the shared Netflix standing order are one service, so the
        # 300.00 duplicate and Netflix's own downgrade cannot both be collected.
        counted = {s["merchant"] for s in report["savings"]["strategies"]}
        assert "James Wilson - Shared Netflix" in counted
        assert "Netflix" not in counted

    def test_no_service_is_counted_twice(self, report):
        services = [s["service"] for s in report["savings"]["strategies"]]
        assert len(services) == len(set(services))


class TestCancelDirectDebit:
    """The tool takes the merchant and derives the mandate.

    It used to take the mandate directly, which made it only as accurate as the
    model's reading of the report: asked to cancel Google One — card-billed,
    with no mandate at all — it passed DD-4472 and cancelled Namecheap's domain
    renewal. The reference is now looked up from the name, so a request for one
    merchant cannot reach another's mandate.
    """

    def test_a_valid_mandate_is_cancelled(self):
        result = _unguarded(
            cancel_direct_debit, user_id="USR-12345", merchant="Netflix"
        )
        assert result["success"] is True
        assert result["mandate_id"] == "DD-4471"
        assert result["merchant"] == "Netflix"
        assert result["status"] == "cancelled"
        assert result["cancelled_at"] == date.today().isoformat()

    def test_the_confirmation_states_no_further_payments_will_be_taken(self):
        result = _unguarded(
            cancel_direct_debit, user_id="USR-12345", merchant="Netflix"
        )
        assert "no further payments will be taken" in result["message"].lower()
        assert "DD-4471" in result["message"]

    def test_the_confirmation_names_the_payment_that_will_not_be_taken(self):
        result = _unguarded(
            cancel_direct_debit,
            user_id="USR-12345",
            merchant="Namecheap Domain Renewal",
        )
        assert result["merchant"] == "Namecheap Domain Renewal"
        assert result["amount"] == "£79.99"
        mandate = next(d for d in MOCK_DIRECT_DEBITS if d["mandate_id"] == "DD-4472")
        assert result["cancelled_next_payment"] == mandate["next_payment"]

    def test_cancelling_the_mandate_does_not_end_the_merchant_contract(self):
        # Honest framing: the bank can stop paying, it cannot cancel the service.
        result = _unguarded(
            cancel_direct_debit, user_id="USR-12345", merchant="Netflix"
        )
        assert "contract" in result["note"].lower()

    def test_a_card_billed_merchant_is_refused_not_matched_to_a_mandate(self):
        """The exact failure this signature exists to prevent."""
        result = _unguarded(
            cancel_direct_debit, user_id="USR-12345", merchant="Google One"
        )
        assert result["success"] is False
        assert "block_merchant_on_card" in result["error"]
        assert "DD-4472" not in result["error"]

    def test_a_foreign_mandate_id_refuses_rather_than_cancelling_it(self):
        """Passing DD-4472 while naming Google One must stop, not proceed."""
        result = _unguarded(
            cancel_direct_debit,
            user_id="USR-12345",
            merchant="Google One",
            mandate_id="DD-4472",
        )
        assert result["success"] is False
        assert all(dd["status"] == "active" for dd in MOCK_DIRECT_DEBITS)

    def test_a_mandate_id_belonging_to_another_merchant_is_refused(self):
        result = _unguarded(
            cancel_direct_debit,
            user_id="USR-12345",
            merchant="Netflix",
            mandate_id="DD-4472",
        )
        assert result["success"] is False
        assert "DD-4471" in result["error"]

    def test_an_unknown_merchant_is_rejected(self):
        result = _unguarded(
            cancel_direct_debit, user_id="USR-12345", merchant="Nonsense Ltd"
        )
        assert result["success"] is False
        assert "Nonsense Ltd" in result["error"]

    def test_a_standing_order_merchant_is_rejected_rather_than_silently_cancelled(self):
        result = _unguarded(
            cancel_direct_debit,
            user_id="USR-12345",
            merchant="James Wilson - Shared Netflix",
        )
        assert result["success"] is False
        assert "cancel_standing_order" in result["error"]

    def test_cancellation_does_not_mutate_the_mandate_fixtures(self, report):
        _unguarded(cancel_direct_debit, user_id="USR-12345", merchant="Netflix")
        after = json.loads(find_recurring_payments.invoke({"user_id": "USR-12345"}))
        assert after == report
        assert all(dd["status"] == "active" for dd in MOCK_DIRECT_DEBITS)


def _block(card_id="CARD-8834", merchant="Spotify", user_id="USR-12345"):
    return _unguarded(
        block_merchant_on_card, user_id=user_id, card_id=card_id, merchant=merchant
    )


class TestBlockMerchantOnCard:
    def test_a_merchant_is_blocked_on_a_valid_card(self):
        result = _block()
        assert result["success"] is True
        assert result["card_id"] == "CARD-8834"
        assert result["merchant"] == "Spotify"
        assert result["status"] == "blocked"
        assert result["blocked_at"] == date.today().isoformat()

    def test_the_block_is_reported_against_the_right_card(self):
        result = _block(card_id="CARD-5521", merchant="FitLife Gym")
        assert result["success"] is True
        assert result["card_id"] == "CARD-5521"
        assert result["last_four"] == "4821"
        assert "4821" in result["message"]

    def test_the_response_states_the_contract_is_not_cancelled(self):
        caveat = _block()["caveat"].lower()
        assert "not" in caveat
        assert "contract" in caveat
        assert "owe" in caveat

    def test_the_caveat_text_is_present_in_the_payload(self):
        # The honest framing must survive anywhere the payload is read, not just
        # in one field the model might skip over.
        payload = json.dumps(_block()).lower()
        assert "does not cancel" in payload
        assert "still owe" in payload

    def test_the_response_recommends_cancelling_with_the_merchant(self):
        result = _block()
        recommendation = result["recommended_next_step"].lower()
        assert "cancel" in recommendation
        assert "spotify" in recommendation

    def test_an_unknown_card_is_rejected_with_the_valid_options(self):
        result = _block(card_id="CARD-0000")
        assert result["success"] is False
        assert "CARD-0000" in result["error"]
        assert "CARD-5521" in result["error"]
        assert "CARD-8834" in result["error"]
        assert result["valid_card_ids"] == ["CARD-5521", "CARD-8834"]

    def test_a_mandate_id_is_rejected_rather_than_treated_as_a_card(self):
        result = _block(card_id="DD-4471")
        assert result["success"] is False
        assert "cancel_direct_debit" in result["suggestion"]

    def test_card_ids_are_matched_exactly_not_case_insensitively(self):
        # Card IDs come from list_cards verbatim; a fuzzy match would let the
        # model invent one that happens to normalise onto a real card.
        assert _block(card_id="card-8834")["success"] is False

    def test_the_charges_being_stopped_are_named(self):
        result = _block(card_id="CARD-5521", merchant="FitLife Gym")
        assert result["known_charges"]["occurrences"] == 3
        assert result["known_charges"]["amount"] == "£44.00"

    def test_a_merchant_with_no_history_on_the_card_is_still_blocked(self):
        # A block is a forward-looking instruction, so it must work for a
        # merchant the customer has not been charged by on that card yet.
        result = _block(card_id="CARD-5521", merchant="Spotify")
        assert result["success"] is True
        assert result["known_charges"]["occurrences"] == 0
        assert "no charges" in result["known_charges"]["note"].lower()

    def test_blocking_does_not_mutate_the_fixtures(self, report):
        _block(card_id="CARD-8834", merchant="Spotify")
        after = json.loads(find_recurring_payments.invoke({"user_id": "USR-12345"}))
        assert after == report

    def test_the_user_id_is_echoed_from_the_caller(self):
        assert _block(user_id="USR-99999")["user_id"] == "USR-99999"


def _guide(merchant):
    return json.loads(find_cancellation_guide.invoke({"merchant": merchant}))


class _RecordingSearch:
    """Stand-in for a live search backend that records every call it receives."""

    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def __call__(self, merchant):
        self.calls.append(merchant)
        return self.result


def _with_search(monkeypatch, search):
    """Wire in a stand-in backend and declare search available, as a configured
    machine would be. Availability is a separate switch from the backend itself,
    so a run with no credentials never claims it searched."""
    monkeypatch.setattr(
        "src.agents.cancellation_research.guide._search_available", lambda: True
    )
    monkeypatch.setattr("src.agents.cancellation_research.guide._search_web", search)
    return search


def _without_search(monkeypatch):
    monkeypatch.setattr(
        "src.agents.cancellation_research.guide._search_available", lambda: False
    )


_SEARCH_RESULT = {
    "url": "https://obscure-saas.example/account/billing",
    "steps": [
        "Sign in at obscure-saas.example.",
        "Open Billing and choose Cancel plan.",
    ],
    "notice_period": "30 days, according to the vendor's published terms.",
    "gotchas": "The vendor bills annually in advance.",
    "results": [
        {
            "title": "Cancelling your Obscure SaaS plan",
            "url": "https://obscure-saas.example/help/cancel",
        }
    ],
}


class TestCancellationGuideDirectoryHit:
    def test_a_covered_merchant_returns_the_directory_entry(self):
        guide = _guide("Netflix")
        assert guide["found"] is True
        assert guide["merchant"] == "Netflix"
        assert guide["url"] == "https://www.netflix.com/cancelplan"
        assert len(guide["steps"]) == 3
        assert guide["notice_period"]
        assert guide["gotchas"]

    def test_the_source_is_the_directory(self):
        assert _guide("Netflix")["source"] == "directory"

    def test_a_researched_entry_carries_its_verification_date_and_source_url(self):
        guide = _guide("Spotify")
        assert guide["verified"] is True
        assert guide["verified_on"] == "2026-08-06"
        assert guide["directory_source"].startswith("https://")

    def test_an_illustrative_entry_is_not_presented_as_verified_fact(self):
        # FitLife Gym is invented. The steps are plausible, not researched, and
        # the response has to say so rather than borrowing the directory's
        # authority for a merchant nobody checked.
        guide = _guide("FitLife Gym")
        assert guide["found"] is True
        assert guide["illustrative"] is True
        assert guide["verified"] is False
        assert guide["verified_on"] is None
        assert "illustrative" in guide["note"].lower()

    def test_a_directory_hit_performs_zero_search_calls(self, monkeypatch):
        # The whole point of directory-first: a merchant we already know about
        # must never reach the network, under any circumstances.
        search = _with_search(monkeypatch, _RecordingSearch(result=_SEARCH_RESULT))

        guide = _guide("Netflix")

        assert search.calls == []
        assert guide["source"] == "directory"

    def test_every_covered_merchant_is_found_without_search(self, monkeypatch):
        search = _with_search(monkeypatch, _RecordingSearch(result=_SEARCH_RESULT))

        for merchant in CANCELLATION_DIRECTORY:
            assert _guide(merchant)["found"] is True

        assert search.calls == []

    def test_the_lookup_is_deterministic(self):
        assert _guide("Netflix") == _guide("Netflix")


class TestCancellationGuideMatching:
    def test_the_lookup_is_case_insensitive(self):
        assert _guide("netflix")["merchant"] == "Netflix"
        assert _guide("NETFLIX")["merchant"] == "Netflix"
        assert _guide("nEtFlIx")["merchant"] == "Netflix"

    def test_surrounding_whitespace_is_tolerated(self):
        assert _guide("  Spotify  ")["merchant"] == "Spotify"

    def test_a_transaction_description_form_matches(self):
        # "Spotify UK" and "Adobe Creative Cloud" are how these merchants appear
        # on the statement — the agent will pass them through verbatim.
        assert _guide("Spotify UK")["merchant"] == "Spotify"
        assert _guide("Adobe Creative Cloud")["merchant"] == "Adobe"
        assert _guide("Namecheap Domain Renewal")["merchant"] == "Namecheap"

    def test_the_shared_netflix_description_matches_netflix(self):
        assert _guide("James Wilson - Shared Netflix")["merchant"] == "Netflix"

    def test_every_held_subscription_description_resolves_to_a_guide(self, report):
        for sub in report["subscriptions"]:
            if sub["essential"]:
                continue
            assert _guide(sub["merchant"])["found"] is True

    def test_the_matched_lookup_form_is_reported(self):
        assert _guide("Spotify UK")["matched_on"] == "spotify uk"


class TestCancellationGuideSearchFallback:
    def test_an_unknown_merchant_reaches_the_search_fallback(self, monkeypatch):
        search = _with_search(monkeypatch, _RecordingSearch(result=_SEARCH_RESULT))

        guide = _guide("Obscure SaaS Ltd")

        assert search.calls == ["Obscure SaaS Ltd"]
        assert guide["found"] is True
        assert guide["source"] == "web_search"

    def test_search_results_are_labelled_unverified_with_no_verified_date(
        self, monkeypatch
    ):
        _with_search(monkeypatch, _RecordingSearch(_SEARCH_RESULT))
        guide = _guide("Obscure SaaS Ltd")

        assert guide["verified"] is False
        assert guide["verified_on"] is None
        assert "not been verified" in guide["caveat"].lower()

    def test_search_content_is_carried_through(self, monkeypatch):
        _with_search(monkeypatch, _RecordingSearch(_SEARCH_RESULT))
        guide = _guide("Obscure SaaS Ltd")

        assert guide["url"] == _SEARCH_RESULT["url"]
        assert guide["steps"] == _SEARCH_RESULT["steps"]
        assert guide["gotchas"] == _SEARCH_RESULT["gotchas"]
        assert guide["results"] == _SEARCH_RESULT["results"]

    def test_a_search_that_finds_nothing_degrades_rather_than_erroring(
        self, monkeypatch
    ):
        _with_search(monkeypatch, _RecordingSearch(result=None))
        guide = _guide("Obscure SaaS Ltd")

        assert guide["found"] is False
        assert guide["searched"] is True
        assert "error" not in guide

    def test_a_search_that_raises_degrades_rather_than_propagating(self, monkeypatch):
        def exploding_search(merchant):
            raise RuntimeError("network unreachable")

        _with_search(monkeypatch, exploding_search)
        guide = _guide("Obscure SaaS Ltd")

        assert guide["found"] is False
        assert guide["searched"] is True
        assert guide["message"]


class TestCancellationGuideDegradedResponse:
    @pytest.fixture
    def degraded(self, monkeypatch):
        # No credentials, so no search: this is the shape a reviewer running the
        # code offline actually gets.
        _without_search(monkeypatch)
        return _guide("Obscure SaaS Ltd")

    def test_search_is_unavailable_without_credentials(self, monkeypatch):
        # The live backend needs ANTHROPIC_API_KEY. Without it there is nothing
        # to fall back to, and _search_web must say so rather than raising.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert _search_available() is False
        assert _search_web("Obscure SaaS Ltd") is None

    def test_a_miss_is_not_an_error_and_not_empty(self, degraded):
        assert degraded["found"] is False
        assert "error" not in degraded
        assert degraded["message"]

    def test_the_miss_names_the_merchant_that_was_asked_for(self, degraded):
        assert degraded["merchant"] == "Obscure SaaS Ltd"
        assert "Obscure SaaS Ltd" in degraded["message"]

    def test_the_miss_records_that_no_search_was_available(self, degraded):
        assert degraded["searched"] is False
        assert degraded["source"] == "unavailable"

    def test_the_miss_says_what_is_not_known(self, degraded):
        assert "do not have" in degraded["message"].lower()

    def test_the_miss_explains_how_to_find_the_answer(self, degraded):
        how = " ".join(degraded["how_to_find_it"]).lower()
        assert len(degraded["how_to_find_it"]) >= 2
        assert "confirmation email" in how or "website" in how

    def test_the_miss_states_the_customers_rights(self, degraded):
        rights = " ".join(degraded["your_rights"]).lower()
        assert len(degraded["your_rights"]) >= 2
        assert "direct debit" in rights
        assert "card" in rights

    def test_the_miss_carries_no_verification_date(self, degraded):
        assert degraded["verified"] is False
        assert degraded["verified_on"] is None

    def test_an_empty_merchant_degrades_rather_than_matching_something(self):
        guide = _guide("   ")
        assert guide["found"] is False

    def test_the_degraded_response_is_deterministic(self):
        assert _guide("Obscure SaaS Ltd") == _guide("Obscure SaaS Ltd")
