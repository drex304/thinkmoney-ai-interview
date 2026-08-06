"""Comprehensive tests for mock tools — verifies JSON structure, field values, and edge cases."""

import json


from src.tools import (
    ACCOUNT_TOOLS,
    CARD_TOOLS,
    TRANSACTION_TOOLS,
    PAYMENT_TOOLS,
    KYC_TOOLS,
    ALL_TOOLS,
)


class TestAccountTools:
    def test_get_account_details(self):
        from src.tools.account import get_account_details

        result = json.loads(get_account_details.invoke({"user_id": "USR-2847"}))
        assert result["user_id"] == "USR-2847"
        assert result["name"] == "Sarah Johnson"
        assert result["status"] == "active"
        assert "GBP" in result["balance"]

    def test_get_account_limits(self):
        from src.tools.account import get_account_limits

        result = json.loads(get_account_limits.invoke({"user_id": "USR-2847"}))
        assert "limits" in result
        assert "daily_spending" in result["limits"]

    def test_update_contact_info_valid(self):
        from src.tools.account import update_contact_info

        result = json.loads(
            update_contact_info.invoke(
                {"user_id": "USR-2847", "field": "email", "new_value": "new@email.com"}
            )
        )
        assert result["success"] is True
        assert result["new_value"] == "new@email.com"

    def test_update_contact_info_invalid_field(self):
        from src.tools.account import update_contact_info

        result = json.loads(
            update_contact_info.invoke(
                {"user_id": "USR-2847", "field": "password", "new_value": "x"}
            )
        )
        assert result["success"] is False


class TestCardTools:
    def test_list_cards(self):
        from src.tools.cards import list_cards

        result = json.loads(list_cards.invoke({"user_id": "USR-2847"}))
        assert len(result["cards"]) == 2
        assert result["cards"][0]["card_id"] == "CARD-5521"

    def test_freeze_card(self):
        from src.tools.cards import freeze_card

        result = json.loads(freeze_card.invoke({"card_id": "CARD-5521"}))
        assert result["success"] is True
        assert result["status"] == "frozen"

    def test_get_card_status_known(self):
        from src.tools.cards import get_card_status

        result = json.loads(get_card_status.invoke({"card_id": "CARD-5521"}))
        assert result["card_id"] == "CARD-5521"
        assert "error" not in result

    def test_get_card_status_unknown(self):
        from src.tools.cards import get_card_status

        result = json.loads(get_card_status.invoke({"card_id": "CARD-9999"}))
        assert "error" in result


class TestTransactionTools:
    def test_get_transaction_history(self):
        from src.tools.transactions import get_transaction_history

        result = json.loads(
            get_transaction_history.invoke({"user_id": "USR-2847", "days": 30})
        )
        assert result["transaction_count"] > 0
        assert len(result["transactions"]) > 0

    def test_get_transaction_details_found(self):
        from src.tools.transactions import get_transaction_details

        result = json.loads(
            get_transaction_details.invoke({"transaction_id": "TXN-90001"})
        )
        assert result["transaction_id"] == "TXN-90001"
        assert "error" not in result

    def test_dispute_transaction(self):
        from src.tools.transactions import dispute_transaction

        result = json.loads(
            dispute_transaction.invoke(
                {"transaction_id": "TXN-89985", "reason": "goods_not_received"}
            )
        )
        assert result["success"] is True
        assert result["status"] == "under_review"


class TestPaymentTools:
    def test_get_payees(self):
        from src.tools.payments import get_payees

        result = json.loads(get_payees.invoke({"user_id": "USR-2847"}))
        assert len(result["payees"]) == 3

    def test_initiate_transfer(self):
        from src.tools.payments import initiate_transfer

        result = json.loads(
            initiate_transfer.invoke(
                {
                    "user_id": "USR-2847",
                    "payee_id": "PAY-001",
                    "amount": 50.0,
                    "currency": "GBP",
                    "reference": "Test",
                }
            )
        )
        assert result["success"] is True
        assert result["status"] == "processing"

    def test_initiate_transfer_negative_amount(self):
        from src.tools.payments import initiate_transfer

        result = json.loads(
            initiate_transfer.invoke(
                {"user_id": "USR-2847", "payee_id": "PAY-001", "amount": -10.0}
            )
        )
        assert result["success"] is False

    def test_get_exchange_rate(self):
        from src.tools.payments import get_exchange_rate

        result = json.loads(
            get_exchange_rate.invoke({"from_currency": "GBP", "to_currency": "EUR"})
        )
        assert result["rate"] > 0
        assert "error" not in result


class TestKycTools:
    def test_get_kyc_status(self):
        from src.tools.kyc import get_kyc_status

        result = json.loads(get_kyc_status.invoke({"user_id": "USR-2847"}))
        assert result["status"] == "verified"
        assert result["identity_verified"] is True

    def test_submit_document_valid(self):
        from src.tools.kyc import submit_document

        result = json.loads(
            submit_document.invoke({"user_id": "USR-2847", "document_type": "passport"})
        )
        assert result["success"] is True
        assert result["status"] == "pending_review"

    def test_submit_document_invalid_type(self):
        from src.tools.kyc import submit_document

        result = json.loads(
            submit_document.invoke(
                {"user_id": "USR-2847", "document_type": "birth_certificate"}
            )
        )
        assert result["success"] is False


# ============================================================
# Additional coverage for tools not tested above
# ============================================================


class TestToolGroups:
    def test_account_tools_count(self):
        assert len(ACCOUNT_TOOLS) == 4

    def test_card_tools_count(self):
        # +1 on the provided five: block_merchant_on_card is a card control, so
        # it lives beside freeze_card rather than with the agent that uses it.
        assert len(CARD_TOOLS) == 6

    def test_transaction_tools_count(self):
        assert len(TRANSACTION_TOOLS) == 4

    def test_payment_tools_count(self):
        # +1 on the provided five: cancel_direct_debit sits with its analogue
        # cancel_standing_order — same operation, different rail.
        assert len(PAYMENT_TOOLS) == 6

    def test_kyc_tools_count(self):
        assert len(KYC_TOOLS) == 4

    def test_agent_owned_tools_are_not_in_the_shared_groups(self):
        # Detection and cancellation guidance belong to one agent each, so they
        # live in that agent's package rather than in the bank's tool surface.
        names = {tool.name for tool in ALL_TOOLS}
        assert "find_recurring_payments" not in names
        assert "find_cancellation_guide" not in names

    def test_all_tools_count(self):
        # The 22 provided tools, plus the two write tools this submission added
        # to existing capability groups.
        assert len(ALL_TOOLS) == 24

    def test_all_tools_count_equals_sum_of_groups(self):
        assert len(ALL_TOOLS) == sum(
            len(group)
            for group in (
                ACCOUNT_TOOLS,
                CARD_TOOLS,
                TRANSACTION_TOOLS,
                PAYMENT_TOOLS,
                KYC_TOOLS,
            )
        )

    def test_all_tools_have_names(self):
        for tool in ALL_TOOLS:
            assert hasattr(tool, "name")
            assert isinstance(tool.name, str)
            assert len(tool.name) > 0

    def test_all_tools_have_descriptions(self):
        for tool in ALL_TOOLS:
            assert hasattr(tool, "description")
            assert isinstance(tool.description, str)
            assert len(tool.description) > 0

    def test_no_duplicate_tool_names(self):
        names = [t.name for t in ALL_TOOLS]
        assert len(names) == len(
            set(names)
        ), f"Duplicate tool names: {[n for n in names if names.count(n) > 1]}"


class TestAccountToolsAdditional:
    def test_get_account_status(self):
        from src.tools.account import get_account_status

        result = json.loads(get_account_status.invoke({"user_id": "USR-2847"}))
        assert result["status"] == "active"
        assert result["kyc_status"] == "verified"
        assert "risk_score" in result

    def test_update_contact_phone(self):
        from src.tools.account import update_contact_info

        result = json.loads(
            update_contact_info.invoke(
                {
                    "user_id": "USR-2847",
                    "field": "phone",
                    "new_value": "+44 7700 999999",
                }
            )
        )
        assert result["success"] is True
        assert result["field"] == "phone"

    def test_update_contact_address(self):
        from src.tools.account import update_contact_info

        result = json.loads(
            update_contact_info.invoke(
                {"user_id": "USR-2847", "field": "address", "new_value": "1 New Street"}
            )
        )
        assert result["success"] is True


class TestCardToolsAdditional:
    def test_unfreeze_card(self):
        from src.tools.cards import unfreeze_card

        result = json.loads(unfreeze_card.invoke({"card_id": "CARD-5521"}))
        assert result["success"] is True
        assert result["status"] == "active"
        assert "unfrozen_at" in result

    def test_order_replacement_card_lost(self):
        from src.tools.cards import order_replacement_card

        result = json.loads(
            order_replacement_card.invoke(
                {"user_id": "USR-2847", "card_id": "CARD-5521", "reason": "lost"}
            )
        )
        assert result["success"] is True
        assert result["old_card_status"] == "cancelled"
        assert "tracking_ref" in result

    def test_order_replacement_card_damaged_express(self):
        from src.tools.cards import order_replacement_card

        result = json.loads(
            order_replacement_card.invoke(
                {
                    "user_id": "USR-2847",
                    "card_id": "CARD-5521",
                    "reason": "damaged",
                    "delivery": "express",
                }
            )
        )
        assert result["success"] is True
        assert result["fee"] == "£10.00"
        assert "1-2" in result["estimated_delivery"]

    def test_order_replacement_card_standard_is_free(self):
        from src.tools.cards import order_replacement_card

        result = json.loads(
            order_replacement_card.invoke(
                {
                    "user_id": "USR-2847",
                    "card_id": "CARD-5521",
                    "reason": "damaged",
                    "delivery": "standard",
                }
            )
        )
        assert result["fee"] == "£0.00"

    def test_get_card_status_virtual_card(self):
        from src.tools.cards import get_card_status

        result = json.loads(get_card_status.invoke({"card_id": "CARD-8834"}))
        assert result["type"] == "virtual"
        assert result["card_id"] == "CARD-8834"


class TestTransactionToolsAdditional:
    def test_get_transaction_details_not_found(self):
        from src.tools.transactions import get_transaction_details

        result = json.loads(
            get_transaction_details.invoke({"transaction_id": "TXN-00000"})
        )
        assert "error" in result

    def test_get_dispute_status(self):
        from src.tools.transactions import get_dispute_status

        result = json.loads(get_dispute_status.invoke({"dispute_id": "DSP-44210"}))
        assert result["dispute_id"] == "DSP-44210"
        assert result["status"] == "under_review"
        assert "updates" in result
        assert len(result["updates"]) > 0

    def test_dispute_transaction_invalid_reason(self):
        from src.tools.transactions import dispute_transaction

        result = json.loads(
            dispute_transaction.invoke(
                {"transaction_id": "TXN-90001", "reason": "i_dont_like_it"}
            )
        )
        assert result["success"] is False

    def test_dispute_transaction_unauthorised_gives_provisional_credit(self):
        from src.tools.transactions import dispute_transaction

        result = json.loads(
            dispute_transaction.invoke(
                {"transaction_id": "TXN-90001", "reason": "unauthorised"}
            )
        )
        assert result["success"] is True
        assert result["provisional_credit"] is True

    def test_transaction_history_has_varied_types(self):
        from src.tools.transactions import get_transaction_history

        result = json.loads(
            get_transaction_history.invoke({"user_id": "USR-2847", "days": 30})
        )
        types = {t["type"] for t in result["transactions"]}
        assert len(types) > 1  # Should have mix of card_payment, bank_transfer, etc.


class TestPaymentToolsAdditional:
    def test_get_standing_orders(self):
        from src.tools.payments import get_standing_orders

        result = json.loads(get_standing_orders.invoke({"user_id": "USR-2847"}))
        assert len(result["standing_orders"]) == 2
        assert result["standing_orders"][0]["status"] == "active"

    def test_cancel_standing_order(self):
        from src.tools.payments import cancel_standing_order

        # The body, without the confirmation gate the tool now carries — that
        # the gate stops it is asserted in tests/test_confirmation_gate.py:1.
        result = json.loads(
            cancel_standing_order.func.__wrapped__(
                merchant="Landlord - Premier Properties"
            )
        )
        assert result["success"] is True
        assert result["status"] == "cancelled"
        assert "cancelled_at" in result

    def test_get_exchange_rate_invalid_pair(self):
        from src.tools.payments import get_exchange_rate

        result = json.loads(
            get_exchange_rate.invoke({"from_currency": "GBP", "to_currency": "JPY"})
        )
        assert "error" in result

    def test_initiate_transfer_over_limit(self):
        from src.tools.payments import initiate_transfer

        result = json.loads(
            initiate_transfer.invoke(
                {"user_id": "USR-2847", "payee_id": "PAY-001", "amount": 30000.0}
            )
        )
        assert result["success"] is False
        assert "25,000" in result["error"]

    def test_initiate_international_transfer(self):
        from src.tools.payments import initiate_transfer

        result = json.loads(
            initiate_transfer.invoke(
                {
                    "user_id": "USR-2847",
                    "payee_id": "PAY-003",
                    "amount": 100.0,
                    "currency": "EUR",
                    "reference": "Gift",
                }
            )
        )
        assert result["success"] is True
        assert result["type"] == "SWIFT"
        assert result["fee"] == "£3.00"

    def test_get_exchange_rate_reverse(self):
        from src.tools.payments import get_exchange_rate

        result = json.loads(
            get_exchange_rate.invoke({"from_currency": "EUR", "to_currency": "GBP"})
        )
        assert result["rate"] > 0
        assert result["rate"] < 1  # EUR -> GBP should be less than 1


class TestKycToolsAdditional:
    def test_get_required_documents(self):
        from src.tools.kyc import get_required_documents

        result = json.loads(get_required_documents.invoke({"user_id": "USR-2847"}))
        assert result["all_documents_provided"] is True
        assert len(result["documents"]) > 0

    def test_get_account_restrictions(self):
        from src.tools.kyc import get_account_restrictions

        result = json.loads(get_account_restrictions.invoke({"user_id": "USR-2847"}))
        assert result["has_restrictions"] is False
        assert len(result["active_restrictions"]) == 0
        assert len(result["historical_restrictions"]) > 0

    def test_submit_document_driving_licence(self):
        from src.tools.kyc import submit_document

        result = json.loads(
            submit_document.invoke(
                {"user_id": "USR-2847", "document_type": "driving_licence"}
            )
        )
        assert result["success"] is True
        assert result["document_type"] == "driving_licence"
