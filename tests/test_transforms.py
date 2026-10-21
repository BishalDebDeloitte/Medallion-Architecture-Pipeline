"""Offline tests for the deterministic STTM execution engine."""

import unittest

import pandas as pd

from core import transforms
from core.sttm_schema import Operation


def text(*values):
    return pd.Series(values, dtype="string")


def row(**overrides):
    base = {
        "source_col": "v", "target_col": "v", "data_type": "string", "transformation_logic": "",
        "nullable": "true",
    }
    return {**base, **overrides}


class CastTests(unittest.TestCase):
    def test_integer_rejects_fractions_text_and_counts_them(self):
        result, invalid = transforms.cast_series(text("1", " 2", "3.5", "x", None, "1e3"), "integer")
        self.assertEqual(result.dtype, "Int64")
        self.assertEqual(result.tolist(), [1, 2, pd.NA, pd.NA, pd.NA, 1000])
        self.assertEqual(invalid, 2)          # "3.5" and "x"; the real null is not counted

    def test_float_and_thousands_separator_is_invalid(self):
        result, invalid = transforms.cast_series(text("1.5", "1,234", None), "float")
        self.assertEqual(result.dtype, "Float64")
        self.assertEqual((result.iloc[0], invalid), (1.5, 1))

    def test_boolean_tokens(self):
        result, invalid = transforms.cast_series(text("Yes", "n", "1", "maybe", None), "boolean")
        self.assertEqual(result.tolist(), [True, False, True, pd.NA, pd.NA])
        self.assertEqual(invalid, 1)

    def test_dates_formats_timezones_and_normalisation(self):
        values = text("2024-01-05", "05/01/2024", "2024-01-05 23:30:00+02:00", "junk", None)
        date, invalid = transforms.cast_series(values, "date")
        self.assertEqual([str(v)[:10] for v in date.dropna()], ["2024-01-05", "2024-05-01", "2024-01-05"])
        self.assertEqual(invalid, 1)
        self.assertTrue((date.dropna().dt.hour == 0).all())
        stamp, _ = transforms.cast_series(values, "datetime")
        self.assertEqual(str(stamp.iloc[2]), "2024-01-05 21:30:00")     # converted to UTC, stored naive

    def test_string_cast_and_unknown_type(self):
        result, invalid = transforms.cast_series(pd.Series([1, None], dtype="Int64"), "string")
        self.assertEqual((result.tolist(), invalid), (["1", pd.NA], 0))
        with self.assertRaises(ValueError):
            transforms.cast_series(text("a"), "blob")


class OperationTests(unittest.TestCase):
    def test_text_operations(self):
        series = text("  Hello   WORLD ", None)
        apply = transforms.apply_operation
        self.assertEqual(apply(series, Operation("trim")).iloc[0], "Hello   WORLD")
        self.assertEqual(apply(series, Operation("lowercase")).iloc[0], "  hello   world ")
        self.assertEqual(apply(series, Operation("uppercase")).iloc[0], "  HELLO   WORLD ")
        self.assertEqual(apply(text("ann marie"), Operation("titlecase")).iloc[0], "Ann Marie")
        self.assertEqual(apply(series, Operation("collapse_whitespace")).iloc[0], " Hello WORLD ")

    def test_blank_results_become_missing(self):
        self.assertTrue(transforms.apply_operation(text("   ", "x"), Operation("trim")).isna().iloc[0])

    def test_to_date_with_explicit_day_first_format(self):
        parsed = transforms.apply_operation(text("05/01/2024", "31/12/2023"), Operation("to_date", "%d/%m/%Y"))
        self.assertEqual([str(v)[:10] for v in parsed], ["2024-01-05", "2023-12-31"])

    def test_row_level_operations_are_not_column_operations(self):
        with self.assertRaises(ValueError):
            transforms.apply_operation(text("a"), Operation("dropna"))


class TransformColumnsTests(unittest.TestCase):
    def test_renames_selects_only_mapped_columns_and_reports_coercions(self):
        source = pd.DataFrame({"a": text("1", "x", " 3 "), "b": text("p", "q", "r"), "unused": text("u", "u", "u")})
        rows = [
            row(source_col="a", target_col="a_num", data_type="integer"),
            row(source_col="b", target_col="b_up", transformation_logic="uppercase"),
        ]
        frame, report = transforms.transform_columns(source, rows, "silver")
        self.assertEqual(list(frame.columns), ["a_num", "b_up"])
        self.assertEqual(frame["a_num"].tolist(), [1, pd.NA, 3])
        self.assertEqual(report["coerced_to_null"], {"a_num": 1})

    def test_blank_nulled_by_a_text_rule_is_not_reported_as_a_failed_conversion(self):
        source = pd.DataFrame({"v": text("  ", "ok")})
        _, report = transforms.transform_columns(source, [row(transformation_logic="trim")], "bronze")
        self.assertEqual(report["coerced_to_null"], {})

    def test_failed_to_date_is_reported_once(self):
        source = pd.DataFrame({"v": text("2024-01-01", "nope")})
        frame, report = transforms.transform_columns(
            source, [row(data_type="date", transformation_logic="to_date")], "silver",
        )
        self.assertEqual(report["coerced_to_null"], {"v": 1})
        self.assertTrue(frame["v"].isna().iloc[1])

    def test_unsupported_operation_for_the_layer_is_rejected(self):
        with self.assertRaises(ValueError):
            transforms.transform_columns(pd.DataFrame({"v": text("a")}), [row(transformation_logic="fillna:mode")], "bronze")


class CleanseTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "id": pd.array([1, 2, 2, None, 5], dtype="Int64"),
            "amount": pd.array([10.0, None, None, 40.0, 100.0], dtype="Float64"),
            "qty": pd.array([1, None, 3, 4, None], dtype="Int64"),
            "tag": text("a", None, None, "b", "a"),
        })

    def test_fill_strategies(self):
        rows = [
            row(target_col="amount", data_type="float", transformation_logic="fillna:median"),
            row(target_col="qty", data_type="integer", transformation_logic="fillna:mean"),
            row(target_col="tag", transformation_logic="fillna:mode"),
        ]
        frame, report = transforms.cleanse(self.frame(), rows)
        self.assertEqual(frame["amount"].tolist(), [10.0, 40.0, 40.0, 40.0, 100.0])
        self.assertEqual(frame["qty"].tolist(), [1, 3, 3, 4, 3])          # mean 2.67 rounds to 3
        self.assertEqual(frame["tag"].tolist(), ["a", "a", "a", "b", "a"])
        self.assertEqual(report["filled"], {"amount": 2, "qty": 2, "tag": 2})

    def test_fill_with_literal_zero_and_value(self):
        rows = [
            row(target_col="amount", data_type="float", transformation_logic="fillna:zero"),
            row(target_col="qty", data_type="integer", transformation_logic="fillna:value=7"),
            row(target_col="tag", transformation_logic="fillna:value=none"),
        ]
        frame, _ = transforms.cleanse(self.frame(), rows)
        self.assertEqual(frame["amount"].tolist()[1], 0.0)
        self.assertEqual(frame["qty"].tolist()[1], 7)
        self.assertEqual(frame["tag"].tolist()[1], "none")

    def test_invalid_literal_fill_value_fails(self):
        with self.assertRaises(ValueError):
            transforms.cleanse(self.frame(), [row(target_col="qty", data_type="integer", transformation_logic="fillna:value=abc")])

    def test_dropna_then_dedupe_on_key_columns_only(self):
        rows = [
            row(target_col="id", data_type="integer", transformation_logic="dropna|dedupe"),
            row(target_col="amount", data_type="float"),
        ]
        frame, report = transforms.cleanse(self.frame(), rows)
        self.assertEqual(frame["id"].tolist(), [1, 2, 5])
        self.assertEqual((report["rows_dropped_dropna"], report["rows_dropped_dedupe"]), (1, 1))
        self.assertEqual(list(frame.index), [0, 1, 2])

    def test_nothing_is_deduplicated_without_a_dedupe_rule(self):
        frame, report = transforms.cleanse(self.frame(), [row(target_col="id", data_type="integer")])
        self.assertEqual(len(frame), 5)
        self.assertEqual(report["rows_dropped_dedupe"], 0)

    def test_nullability_enforcement_lists_every_violation(self):
        frame = pd.DataFrame({"a": pd.array([1, None], dtype="Int64"), "b": text("x", None)})
        rows = [row(target_col="a", nullable="false"), row(target_col="b", nullable="false"), row(target_col="a")]
        with self.assertRaises(ValueError) as caught:
            transforms.enforce_nullability(frame, rows, "t")
        self.assertIn("t.a", str(caught.exception))
        self.assertIn("t.b", str(caught.exception))
        transforms.enforce_nullability(frame, [row(target_col="a", nullable="true")], "t")


if __name__ == "__main__":
    unittest.main()
