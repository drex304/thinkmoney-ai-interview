"""Tests for the relative-dated mock transaction corpus."""

import json
import re
from datetime import date, datetime, timedelta

from src.tools.transactions import MOCK_TRANSACTIONS, _days_ago
from src.tools.payments import get_standing_orders
from src.tools.transactions import get_transaction_details


def _as_date(txn):
    return datetime.fromisoformat(txn["date"].replace("Z", "+00:00")).date()


def _for_merchant(name):
    return [t for t in MOCK_TRANSACTIONS if t["description"] == name]


class TestRelativeDating:
    def test_days_ago_returns_iso_timestamp_relative_to_today(self):
        stamp = _days_ago(7)
        assert stamp.endswith("Z")
        assert _as_date({"date": stamp}) == date.today() - timedelta(days=7)

    def test_no_transaction_is_dated_in_the_future(self):
        today = date.today()
        assert all(_as_date(t) <= today for t in MOCK_TRANSACTIONS)

    def test_source_file_contains_no_hardcoded_date_strings(self):
        from pathlib import Path

        import src.tools.transactions as transactions_module

        source = Path(transactions_module.__file__).read_text()
        # A hardcoded corpus date would look like "2026-03-24T08:30:00Z".
        assert not re.search(r'"\d{4}-\d{2}-\d{2}T', source)

    def test_corpus_bulk_spans_about_ninety_days(self):
        today = date.today()
        recent = [t for t in MOCK_TRANSACTIONS if (today - _as_date(t)).days <= 95]
        assert len(recent) >= 50
        oldest_recent = max((today - _as_date(t)).days for t in recent)
        assert 80 <= oldest_recent <= 95

    def test_corpus_has_roughly_sixty_entries(self):
        assert 55 <= len(MOCK_TRANSACTIONS) <= 70

    def test_transaction_ids_are_unique(self):
        ids = [t["transaction_id"] for t in MOCK_TRANSACTIONS]
        assert len(ids) == len(set(ids))


class TestPlantedRecurringMerchants:
    def test_netflix_direct_debit_charged_three_times(self):
        netflix = _for_merchant("Netflix")
        assert len(netflix) == 3
        assert all(t["amount"] == "-£15.99" for t in netflix)
        assert all(t["rail"] == "direct_debit" for t in netflix)
        assert all(t["mandate_id"] == "DD-4471" for t in netflix)

    def test_spotify_shows_a_price_rise_on_card_8834(self):
        spotify = sorted(_for_merchant("Spotify"), key=_as_date)
        assert [t["amount"] for t in spotify] == [
            "-£11.99",
            "-£11.99",
            "-£12.99",
            "-£12.99",
        ]
        assert all(t["rail"] == "card_on_file" for t in spotify)
        assert all(t["card_used"] == "CARD-8834" for t in spotify)

    def test_adobe_shows_a_converted_free_trial(self):
        adobe = sorted(_for_merchant("Adobe Creative Cloud"), key=_as_date)
        assert [t["amount"] for t in adobe] == ["-£0.00", "-£9.99", "-£9.99"]
        assert all(t["card_used"] == "CARD-8834" for t in adobe)

    def test_fitlife_gym_charged_three_times_on_card_5521(self):
        gym = _for_merchant("FitLife Gym")
        assert len(gym) == 3
        assert all(t["amount"] == "-£44.00" for t in gym)
        assert all(t["card_used"] == "CARD-5521" for t in gym)

    def test_cloud_storage_pair_is_present_and_same_category(self):
        icloud = _for_merchant("Apple iCloud+")
        google = _for_merchant("Google One")
        assert len(icloud) == 3 and len(google) == 3
        assert all(t["amount"] == "-£2.99" for t in icloud)
        assert all(t["amount"] == "-£7.99" for t in google)
        assert {t["category"] for t in icloud} == {t["category"] for t in google}
        assert all(t["card_used"] == "CARD-5521" for t in icloud)
        assert all(t["card_used"] == "CARD-8834" for t in google)

    def test_domain_renewal_is_annual_and_due_in_about_eleven_days(self):
        domain = sorted(_for_merchant("Namecheap Domain Renewal"), key=_as_date)
        assert len(domain) >= 2
        assert all(t["amount"] == "-£79.99" for t in domain)
        assert all(t["mandate_id"] == "DD-4472" for t in domain)
        gap = (_as_date(domain[-1]) - _as_date(domain[-2])).days
        assert 360 <= gap <= 370
        days_until_next = 365 - (date.today() - _as_date(domain[-1])).days
        assert 8 <= days_until_next <= 14

    def test_shared_netflix_standing_order_is_planted(self):
        shared = _for_merchant("James Wilson - Shared Netflix")
        assert len(shared) == 3
        assert all(t["amount"] == "-£25.00" for t in shared)
        assert all(t["rail"] == "standing_order" for t in shared)
        assert all(t["order_id"] == "SO-102" for t in shared)

    def test_rent_standing_order_is_planted(self):
        rent = _for_merchant("Landlord - Premier Properties")
        assert len(rent) == 3
        assert all(t["amount"] == "-£1,200.00" for t in rent)
        assert all(t["order_id"] == "SO-101" for t in rent)

    def test_every_recurring_transaction_carries_a_rail_and_category(self):
        recurring_rails = {"direct_debit", "standing_order", "card_on_file"}
        planted = [t for t in MOCK_TRANSACTIONS if t["rail"] in recurring_rails]
        assert len(planted) >= 27
        assert all(t["category"] for t in planted)

    def test_all_transactions_carry_a_rail_and_category(self):
        assert all(t.get("rail") for t in MOCK_TRANSACTIONS)
        assert all(t.get("category") for t in MOCK_TRANSACTIONS)


class TestNegativeCasesRetained:
    def test_one_off_transactions_are_retained_as_single_occurrences(self):
        for description in (
            "Tesco Express",
            "TfL - Contactless",
            "Acme Corp - Salary",
            "ATM Withdrawal - Barclays",
            "Transfer to James Wilson",
        ):
            assert len(_for_merchant(description)) == 1, description

    def test_pret_is_present_and_never_billed_on_a_subscription_rail(self):
        pret = _for_merchant("Pret A Manger")
        assert pret
        assert all(t["rail"] == "card_payment" for t in pret)


class TestExistingIdsStillRetrievable:
    def test_known_transaction_ids_remain_retrievable(self):
        for transaction_id in ("TXN-90001", "TXN-89985"):
            result = json.loads(
                get_transaction_details.invoke({"transaction_id": transaction_id})
            )
            assert result["transaction_id"] == transaction_id


class TestStandingOrdersStayCoherent:
    def test_next_payment_dates_are_in_the_future(self):
        result = json.loads(get_standing_orders.invoke({"user_id": "USR-2847"}))
        today = date.today()
        for order in result["standing_orders"]:
            next_payment = date.fromisoformat(order["next_payment"])
            assert today < next_payment <= today + timedelta(days=31)
