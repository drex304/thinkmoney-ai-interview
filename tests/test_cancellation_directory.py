"""Tests for the prefetched cancellation directory.

The directory is the cancellation research agent's primary source, so its value is entirely in
being accurate and complete. These tests lock the schema, the researched facts that
were easy to get wrong from memory, and the honest labelling of the two invented
entries.
"""

from datetime import date

import pytest

from src.agents.cancellation_research.directory import (
    CANCELLATION_DIRECTORY,
    FICTIONAL_MERCHANTS,
    REAL_MERCHANTS,
)
from src.tools.transactions import MOCK_TRANSACTIONS

REQUIRED_FIELDS = (
    "merchant",
    "url",
    "steps",
    "notice_period",
    "gotchas",
    "verified_on",
)
CORPUS_MERCHANTS = {t["description"] for t in MOCK_TRANSACTIONS}


def _entry(name):
    return CANCELLATION_DIRECTORY[name]


class TestSchema:
    @pytest.mark.parametrize("name", sorted(CANCELLATION_DIRECTORY))
    def test_entry_carries_every_required_field(self, name):
        entry = _entry(name)
        for field in REQUIRED_FIELDS:
            assert field in entry, f"{name} is missing {field}"

    @pytest.mark.parametrize("name", sorted(CANCELLATION_DIRECTORY))
    def test_key_matches_the_entry_merchant(self, name):
        assert _entry(name)["merchant"] == name

    @pytest.mark.parametrize("name", sorted(CANCELLATION_DIRECTORY))
    def test_steps_are_an_ordered_non_empty_list_of_strings(self, name):
        steps = _entry(name)["steps"]
        assert isinstance(steps, list)
        assert len(steps) >= 2
        assert all(isinstance(step, str) and step.strip() for step in steps)

    @pytest.mark.parametrize("name", sorted(CANCELLATION_DIRECTORY))
    def test_url_is_https(self, name):
        assert _entry(name)["url"].startswith("https://")

    @pytest.mark.parametrize("name", sorted(CANCELLATION_DIRECTORY))
    def test_notice_period_and_gotchas_are_substantive_prose(self, name):
        entry = _entry(name)
        assert len(entry["notice_period"]) > 3
        assert len(entry["gotchas"]) > 30

    @pytest.mark.parametrize("name", sorted(CANCELLATION_DIRECTORY))
    def test_aliases_include_the_merchant_name_lowercased(self, name):
        aliases = _entry(name)["aliases"]
        assert isinstance(aliases, list)
        assert name.lower() in aliases
        assert all(alias == alias.lower() for alias in aliases)


class TestCoverage:
    def test_real_and_fictional_partition_the_directory(self):
        assert set(REAL_MERCHANTS) | set(FICTIONAL_MERCHANTS) == set(
            CANCELLATION_DIRECTORY
        )
        assert not set(REAL_MERCHANTS) & set(FICTIONAL_MERCHANTS)

    def test_seven_researched_services_are_present(self):
        assert set(REAL_MERCHANTS) == {
            "Netflix",
            "Spotify",
            "Adobe",
            "Amazon Prime",
            "Disney+",
            "Apple iCloud+",
            "Google One",
        }

    def test_two_fictional_entries_are_present(self):
        assert set(FICTIONAL_MERCHANTS) == {"FitLife Gym", "Namecheap"}

    @pytest.mark.parametrize("name", sorted(REAL_MERCHANTS))
    def test_real_entries_are_verified_on_the_research_date(self, name):
        assert _entry(name)["verified_on"] == "2026-08-06"
        assert not _entry(name)["illustrative"]

    @pytest.mark.parametrize("name", sorted(FICTIONAL_MERCHANTS))
    def test_fictional_entries_are_explicitly_labelled_illustrative(self, name):
        entry = _entry(name)
        assert entry["illustrative"] is True
        assert "illustrative" in entry["note"].lower()
        assert entry["verified_on"] is None

    def test_verified_dates_are_not_in_the_future(self):
        for name in REAL_MERCHANTS:
            assert date.fromisoformat(_entry(name)["verified_on"]) <= date.today()


class TestProvesResearchBeyondTheCorpus:
    def test_amazon_prime_and_disney_plus_are_not_held_by_the_customer(self):
        """The agent can answer about a service the customer does not subscribe to."""
        for name in ("Amazon Prime", "Disney+"):
            assert name in CANCELLATION_DIRECTORY
            assert not any(name.lower() in m.lower() for m in CORPUS_MERCHANTS)

    def test_every_recurring_corpus_merchant_has_a_directory_route(self):
        """Every subscription the customer actually holds is answerable."""
        held = {
            "Netflix": "Netflix",
            "Spotify": "Spotify",
            "Adobe Creative Cloud": "Adobe",
            "Apple iCloud+": "Apple iCloud+",
            "Google One": "Google One",
            "FitLife Gym": "FitLife Gym",
            "Namecheap Domain Renewal": "Namecheap",
        }
        for description, entry_name in held.items():
            assert description in CORPUS_MERCHANTS
            assert description.lower() in _entry(entry_name)["aliases"]


class TestResearchedContent:
    """The specific facts the research pass corrected, and the ones most load-bearing.

    These assertions exist because the first draft of this data was written from
    memory and two entries were wrong. If someone rewrites the prose, these facts
    must survive the rewrite.
    """

    def test_netflix_cancels_at_period_end_and_warns_about_payment_partners(self):
        entry = _entry("Netflix")
        assert entry["url"] == "https://www.netflix.com/cancelplan"
        assert "billing period" in entry["notice_period"].lower()
        gotchas = entry["gotchas"].lower()
        assert "apple" in gotchas and "google" in gotchas

    def test_spotify_warns_that_trial_cancellation_is_immediate(self):
        gotchas = _entry("Spotify")["gotchas"].lower()
        assert "trial" in gotchas
        assert "immediately" in gotchas

    def test_adobe_records_the_14_day_window_and_the_50_percent_fee(self):
        entry = _entry("Adobe")
        assert "14 days" in entry["gotchas"]
        assert "50%" in entry["gotchas"]
        assert "remaining" in entry["gotchas"].lower()

    def test_amazon_prime_refund_depends_on_benefits_used(self):
        gotchas = _entry("Amazon Prime")["gotchas"].lower()
        assert "full" in gotchas and "partial" in gotchas
        assert "refund" in gotchas

    def test_disney_plus_cancellation_does_not_delete_the_account(self):
        gotchas = _entry("Disney+")["gotchas"].lower()
        assert "does not delete" in gotchas

    def test_icloud_stops_syncing_rather_than_deleting_data(self):
        gotchas = _entry("Apple iCloud+")["gotchas"].lower()
        assert "not deleted" in gotchas
        assert "sync" in gotchas

    def test_google_one_records_the_two_year_deletion_threshold(self):
        gotchas = _entry("Google One")["gotchas"].lower()
        assert "2 years" in gotchas
        assert "deleted" in gotchas
        assert (
            "60" not in gotchas
        ), "the 60-day read-only window was the researched-away error"

    def test_fitlife_gym_records_the_written_notice_period(self):
        entry = _entry("FitLife Gym")
        assert "30 days" in entry["notice_period"]
        assert "notice" in entry["notice_period"].lower()

    def test_namecheap_warns_auto_renew_is_not_cancellation(self):
        gotchas = _entry("Namecheap")["gotchas"].lower()
        assert "auto-renew" in gotchas
        assert "not the same" in gotchas


class TestSources:
    def test_every_real_entry_cites_a_source_url(self):
        for name in REAL_MERCHANTS:
            source = _entry(name)["source"]
            assert source.startswith("https://")

    def test_fictional_entries_cite_no_source(self):
        for name in FICTIONAL_MERCHANTS:
            assert _entry(name)["source"] is None
