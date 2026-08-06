"""Tests for merchant → rail resolution.

This module is the answer to "how do I make sure cancelling Google One cancels
Google One". The mapping from a merchant to its rail and reference is a fact in
the fixtures; it used to be re-derived by the model from the JSON report when it
chose a tool and an identifier, and that second derivation is the one that
failed. These tests pin the first.
"""

import pytest

from src.tools.rails import RAIL_TOOL, find_rails, resolve_rail


class TestFindRails:
    @pytest.mark.parametrize(
        "merchant, rail, reference",
        [
            ("Google One", "card_on_file", "CARD-8834"),
            ("Spotify", "card_on_file", "CARD-8834"),
            ("FitLife Gym", "card_on_file", "CARD-5521"),
            ("Netflix", "direct_debit", "DD-4471"),
            ("Namecheap Domain Renewal", "direct_debit", "DD-4472"),
            ("James Wilson - Shared Netflix", "standing_order", "SO-102"),
            ("Landlord - Premier Properties", "standing_order", "SO-101"),
        ],
    )
    def test_every_subscription_resolves_to_its_own_reference(
        self, merchant, rail, reference
    ):
        assert [(r["rail"], r["reference"]) for r in find_rails(merchant)] == [
            (rail, reference)
        ]

    def test_an_exact_name_wins_over_a_substring_match(self):
        """ "Netflix" is inside "James Wilson - Shared Netflix" too.

        Returning both would make the customer's plainest possible request
        ambiguous, so an exact match short-circuits the fuzzy one.
        """
        assert [r["reference"] for r in find_rails("Netflix")] == ["DD-4471"]

    def test_a_merchant_not_on_the_account_matches_nothing(self):
        assert find_rails("Nonsense Ltd") == []

    def test_an_empty_name_matches_nothing(self):
        # An empty string is a substring of everything, so without a guard this
        # would resolve to whichever subscription happened to be first.
        assert find_rails("") == []


class TestResolveRail:
    def test_the_right_rail_resolves(self):
        match, error = resolve_rail("Netflix", "direct_debit")
        assert error == ""
        assert match["reference"] == "DD-4471"

    def test_a_card_merchant_is_refused_by_the_direct_debit_rail(self):
        """The original failure, at the layer that now prevents it."""
        match, error = resolve_rail("Google One", "direct_debit")
        assert match is None
        assert "block_merchant_on_card" in error
        assert "DD-4472" not in error

    def test_a_standing_order_is_refused_by_the_direct_debit_rail(self):
        match, error = resolve_rail("James Wilson - Shared Netflix", "direct_debit")
        assert match is None
        assert "cancel_standing_order" in error

    def test_a_direct_debit_is_refused_by_the_card_rail(self):
        match, error = resolve_rail("Netflix", "card_on_file")
        assert match is None
        assert "cancel_direct_debit" in error

    def test_an_unknown_merchant_is_refused_by_name(self):
        match, error = resolve_rail("Nonsense Ltd", "direct_debit")
        assert match is None
        assert "Nonsense Ltd" in error

    def test_the_refusal_names_the_tool_that_would_work(self):
        for merchant, rail in [
            ("Google One", "card_on_file"),
            ("Netflix", "direct_debit"),
            ("Landlord - Premier Properties", "standing_order"),
        ]:
            match, _ = resolve_rail(merchant, rail)
            assert RAIL_TOOL[match["rail"]]

    def test_every_rail_has_a_tool(self):
        assert set(RAIL_TOOL) == {"direct_debit", "standing_order", "card_on_file"}
