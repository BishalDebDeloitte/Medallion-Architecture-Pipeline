"""Offline tests for the date_part, clean_number, abs and round operations."""

import unittest

import pandas as pd

from core import transforms
from core.sttm_schema import Operation, parse_transformation, validate_rows


def text(*values):
    return pd.Series(values, dtype="string")


def row(**overrides):
    base = {"layer": "silver", "target_table": "t", "source_table": "t", "source_col": "v", "target_col": "v",
            "data_type": "string", "transformation_logic": "", "aggregation": "", "nullable": "true",
            "join_key": "", "description": ""}
    return {**base, **overrides}


class DatePartTests(unittest.TestCase):
    DATES = text("2025-04-02", "2026-01-15 10:30:00", None, "not a date")

    def part(self, name):
        return transforms.apply_operation(self.DATES, Operation("date_part", name))

    def test_integer_parts(self):
        self.assertEqual(self.part("year").tolist()[:2], [2025, 2026])
        self.assertEqual(self.part("month").tolist()[:2], [4, 1])
        self.assertEqual(self.part("quarter").tolist()[:2], [2, 1])
        self.assertEqual(self.part("day").tolist()[:2], [2, 15])
        self.assertEqual(self.part("weekday").tolist()[:2], [2, 3])           # Wed, Thu (Monday = 0)
        self.assertEqual(self.part("week").tolist()[:2], [14, 3])
        self.assertEqual(str(self.part("year").dtype), "Int64")

    def test_label_parts(self):
        self.assertEqual(self.part("year_month").tolist()[:2], ["2025-04", "2026-01"])
        self.assertEqual(self.part("year_quarter").tolist()[:2], ["2025-Q2", "2026-Q1"])
        self.assertEqual(self.part("month_name").tolist()[:2], ["April", "January"])
        self.assertEqual(self.part("day_name").tolist()[:2], ["Wednesday", "Thursday"])

    def test_missing_and_unparseable_values_become_null(self):
        parts = self.part("year_month")
        self.assertTrue(parts.iloc[2:].isna().all())

    def test_chaining_with_an_explicit_day_first_format(self):
        frame = pd.DataFrame({"d": text("05/01/2024", "31/12/2023")})
        out, report = transforms.transform_columns(
            frame, [row(source_col="d", target_col="ym", transformation_logic="to_date:%d/%m/%Y|date_part:year_month")], "silver",
        )
        self.assertEqual(out["ym"].tolist(), ["2024-01", "2023-12"])
        self.assertEqual(report["coerced_to_null"], {})

    def test_works_on_yyyymmdd_integers_stored_as_text(self):
        frame = pd.DataFrame({"d": text("20250402", "20260101")})
        out, _ = transforms.transform_columns(
            frame, [row(source_col="d", target_col="y", data_type="integer", transformation_logic="to_date:%Y%m%d|date_part:year")], "silver",
        )
        self.assertEqual(out["y"].tolist(), [2025, 2026])

    def test_unconvertible_dates_are_counted(self):
        frame = pd.DataFrame({"d": text("2025-04-02", "garbage", None)})
        _, report = transforms.transform_columns(
            frame, [row(source_col="d", target_col="ym", transformation_logic="date_part:year_month")], "silver")
        self.assertEqual(report["coerced_to_null"], {"ym": 1})


class NumberCleaningTests(unittest.TestCase):
    def test_currency_symbols_separators_and_accounting_negatives(self):
        cleaned = transforms.apply_operation(text("₹1,234.50", " $ 99 ", "(12)", "-7", "N/A", "1,000,000", None), Operation("clean_number"))
        self.assertEqual(cleaned.tolist()[:4], ["1234.50", "99", "-12", "-7"])
        self.assertTrue(cleaned.iloc[4:5].isna().all())                       # "N/A" has no digits
        self.assertEqual(cleaned.iloc[5], "1000000")

    def test_clean_then_cast_gives_numbers_and_counts_failures(self):
        frame = pd.DataFrame({"p": text("₹1,200", "$3.5", "(50)", "free", None)})
        out, report = transforms.transform_columns(
            frame, [row(source_col="p", target_col="price", data_type="float", transformation_logic="clean_number")], "silver")
        self.assertEqual(out["price"].tolist()[:3], [1200.0, 3.5, -50.0])
        self.assertEqual(report["coerced_to_null"], {"price": 1})              # "free"; the real null is not counted

    def test_abs_and_round(self):
        numbers = text("-215.79", "10.456", None, "x")
        absolute = transforms.apply_operation(numbers, Operation("abs"))
        self.assertEqual(absolute.tolist()[:2], [215.79, 10.456])
        self.assertEqual(transforms.apply_operation(numbers, Operation("round", "1")).tolist()[:2], [-215.8, 10.5])
        self.assertEqual(transforms.apply_operation(numbers, Operation("round", "0")).tolist()[:2], [-216.0, 10.0])

    def test_fixing_a_sign_convention_end_to_end(self):
        frame = pd.DataFrame({"refund": text("-10.5", "0", "-3")})
        out, _ = transforms.transform_columns(
            frame, [row(source_col="refund", target_col="refund_amount", data_type="float", transformation_logic="abs")], "silver")
        self.assertEqual(out["refund_amount"].tolist(), [10.5, 0.0, 3.0])


class ValidationTests(unittest.TestCase):
    def test_arguments_are_parsed_and_checked(self):
        self.assertEqual(parse_transformation("date_part:Year_Month|round")[0].arg, "year_month")
        self.assertEqual(parse_transformation("round")[0].arg, "0")
        self.assertEqual(parse_transformation("round:2", "gold")[0].arg, "2")
        for bad in ("date_part", "date_part:decade", "round:9", "round:x", "abs:1", "clean_number:x"):
            with self.assertRaises(ValueError, msg=bad):
                parse_transformation(bad, "silver")

    def test_layers(self):
        for operation in ("date_part:year", "clean_number", "abs", "round:1"):
            with self.assertRaises(ValueError, msg=operation):
                parse_transformation(operation, "bronze")
            parse_transformation(operation, "silver")
            parse_transformation(operation, "gold")

    def test_the_target_type_must_match_what_the_operation_produces(self):
        with self.assertRaisesRegex(ValueError, "data_type must be integer"):
            validate_rows([row(transformation_logic="date_part:year", data_type="string")], "silver")
        with self.assertRaisesRegex(ValueError, "data_type must be string"):
            validate_rows([row(transformation_logic="date_part:year_month", data_type="integer")], "silver")
        with self.assertRaisesRegex(ValueError, "data_type must be integer or float"):
            validate_rows([row(transformation_logic="abs", data_type="string")], "silver")
        validate_rows([row(transformation_logic="to_date:%d/%m/%Y|date_part:year_month", data_type="string")], "silver")
        validate_rows([row(transformation_logic="clean_number|round:2", data_type="float")], "silver")

    def test_a_gold_dimension_can_be_a_month(self):
        validate_rows([
            row(layer="gold", target_table="monthly", source_table="orders", source_col="order_date", target_col="month",
                transformation_logic="date_part:year_month"),
            row(layer="gold", target_table="monthly", source_table="orders", source_col="amount", target_col="revenue",
                data_type="float", aggregation="sum"),
        ], "gold")


if __name__ == "__main__":
    unittest.main()
