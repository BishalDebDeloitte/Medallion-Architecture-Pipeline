"""Offline tests for sensitive-column detection and redaction."""

import unittest

import pandas as pd

from core import sensitivity as s


class NameDetectionTests(unittest.TestCase):
    def test_sensitive_names_are_recognised(self):
        cases = {
            "credit_card_number": "financial", "Credit Card": "financial", "cardNumber": "financial", "cvv": "financial",
            "upi_id": "financial", "bank_account_number": "financial", "IFSC Code": "financial", "iban": "financial",
            "ssn": "identity", "Aadhaar": "identity", "passport_no": "identity", "tax_id": "identity",
            "password": "credential", "api_key": "credential", "access_token": "credential",
            "email": "contact", "Email Address": "contact", "phone_number": "contact", "Phone 1": "contact", "mobile": "contact",
        }
        for name, kind in cases.items():
            self.assertEqual(s.sensitivity_by_name(name), kind, name)

    def test_ordinary_names_are_not_flagged(self):
        for name in ("category", "net_amount", "order_id", "customer_id", "first_name", "city", "postal_code", "location_id",
                     "account_manager", "cardinality", "payment_method", "bank_holiday", "emailed_flag_count_x"):
            self.assertIsNone(s.sensitivity_by_name(name), name)

    def test_name_normalisation(self):
        self.assertEqual(s.normalise_name("Phone 1"), "phone_1")
        self.assertEqual(s.normalise_name("creditCardNumber"), "credit_card_number")


class ValueDetectionTests(unittest.TestCase):
    CARDS = ["4111-1111-1111-1111", "5500 0000 0000 0004", "340000000000009", "6011000000000004", "4012888888881881", "4222222222222"]

    def test_luhn_valid_card_numbers_are_flagged_whatever_the_name(self):
        self.assertEqual(s.classify_column("ref", pd.Series(self.CARDS)), "financial")

    def test_random_digit_strings_are_not_cards(self):
        values = pd.Series(["1234567890123456", "9999999999999999", "1111222233334445", "5555666677778888", "1010101010101010"])
        self.assertIsNone(s.classify_column("ref", values))

    def test_emails_are_flagged_by_shape(self):
        self.assertEqual(s.classify_column("contact_info", pd.Series([f"user{i}@example.com" for i in range(10)])), "contact")

    def test_too_few_values_or_mixed_values_are_not_flagged(self):
        self.assertIsNone(s.classify_column("x", pd.Series(["a@b.co", "c@d.co"])))
        mixed = pd.Series(["a@b.co"] * 3 + ["not an email"] * 7)
        self.assertIsNone(s.classify_column("x", mixed))

    def test_name_wins_and_none_values_are_safe(self):
        self.assertEqual(s.classify_column("email", None), "contact")
        self.assertIsNone(s.classify_column("category", None))
        self.assertIsNone(s.classify_column("category", pd.Series([None] * 10)))


class RedactionTests(unittest.TestCase):
    def test_samples_are_replaced_only_for_sensitive_columns(self):
        self.assertEqual(s.redact_samples(["a", "b"], "financial"), ["[redacted: financial]"])
        self.assertEqual(s.redact_samples(["a", "b"], None), ["a", "b"])
        self.assertEqual(s.redact_samples([], "financial"), [])

    def test_records_mask_sensitive_columns_and_keep_the_rest(self):
        frame = pd.DataFrame({
            "name": ["Ann", "Bob"], "credit_card_number": ["4111-1111-1111-1111", "5500-0000-0000-0004"],
            "total": [1.5, 2.5], "note": ["ok", "fine"],
        })
        records = s.redacted_records(frame, 2)
        self.assertEqual(records[0], {"name": "Ann", "credit_card_number": "[redacted: financial]", "total": 1.5, "note": "ok"})
        self.assertNotIn("4111", str(records))

    def test_sensitive_columns_report(self):
        frame = pd.DataFrame({"a": [1], "Email": ["x@y.com"], "ref": ["4111111111111111"]})
        self.assertEqual(s.sensitive_columns(frame), {"Email": "contact"})


if __name__ == "__main__":
    unittest.main()
