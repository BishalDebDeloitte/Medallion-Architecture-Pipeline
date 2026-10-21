"""Offline tests for fixing data-quality problems during transformation.

Covers strict/decimal-comma number cleaning, ``map_values`` (category standardisation and placeholder
nulling), ``valid_range`` (impossible values become counted nulls) and the profiler hints that point to them.
"""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from agents import profiler
from core import profiling as p
from core import transforms
from core.sttm_schema import Operation, parse_transformation, parse_value_map, parse_value_range, validate_rows


def text(*values):
    return pd.Series(values, dtype="string")


def row(**overrides):
    base = {"layer": "silver", "target_table": "t", "source_table": "t", "source_col": "v", "target_col": "v",
            "data_type": "string", "transformation_logic": "", "aggregation": "", "nullable": "true",
            "join_key": "", "description": ""}
    return {**base, **overrides}


class StrictNumberCleaningTests(unittest.TestCase):
    def clean(self, *values, eu=False):
        return transforms.apply_operation(text(*values), Operation("clean_number", "eu" if eu else "")).tolist()

    def test_text_that_is_not_a_number_after_cleaning_becomes_missing(self):
        out = self.clean("12abc", "1.2.3", "N/A", "", "--5", "12-", "₹1,200")
        self.assertTrue(all(pd.isna(v) for v in out[:6]), out)
        self.assertEqual(out[6], "1200")

    def test_currency_codes_and_words_are_stripped(self):
        self.assertEqual(self.clean("USD 1,200.50", "Rs.1,200", "INR1200", "1200 EUR", "−12"), ["1200.50", "1200", "1200", "1200", "-12"])

    def test_accounting_negatives_and_percent(self):
        self.assertEqual(self.clean("(12)", "(-12)", "(1,234.5)", "12%"), ["-12", "-12", "-1234.5", "12"])

    def test_decimal_commas(self):
        out = self.clean("1.234,56", "12,50", "(12,5)", "1.234.567,8", "12.50", "7", eu=True)
        self.assertEqual(out, ["1234.56", "12.50", "-12.5", "1234567.8", "12.50", "7"])

    def test_unconvertible_values_are_counted_by_the_step(self):
        frame = pd.DataFrame({"p": text("₹1,200", "12abc", "free", None)})
        out, report = transforms.transform_columns(
            frame, [row(source_col="p", target_col="price", data_type="float", transformation_logic="clean_number")], "silver")
        self.assertEqual(out["price"].dropna().tolist(), [1200.0])
        self.assertEqual(report["coerced_to_null"], {"price": 2})

    def test_decimal_comma_column_end_to_end(self):
        frame = pd.DataFrame({"p": text("1.234,56", "99,5", "10")})
        out, report = transforms.transform_columns(
            frame, [row(source_col="p", target_col="price", data_type="float", transformation_logic="clean_number:eu")], "silver")
        self.assertEqual(out["price"].tolist(), [1234.56, 99.5, 10.0])
        self.assertEqual(report["coerced_to_null"], {})


class MapValuesTests(unittest.TestCase):
    def test_variants_are_standardised_and_the_rest_is_kept(self):
        series = text(" N.Y. ", "NY", "ny", "Boston", None, "new york")
        out = transforms.apply_operation(
            transforms.apply_operation(transforms.apply_operation(series, Operation("trim")), Operation("lowercase")),
            Operation("map_values", "n.y.=ny;new york=ny"),
        )
        self.assertEqual(out.tolist()[:4], ["ny", "ny", "ny", "boston"])
        self.assertTrue(pd.isna(out.iloc[4]))
        self.assertEqual(out.iloc[5], "ny")

    def test_a_blank_replacement_turns_a_placeholder_into_null(self):
        out = transforms.apply_operation(text("unknown", "-", "blue", None), Operation("map_values", "unknown=;-="))
        self.assertTrue(out.iloc[:2].isna().all())
        self.assertEqual(out.iloc[2], "blue")

    def test_values_are_matched_exactly(self):
        out = transforms.apply_operation(text("Ny", "ny"), Operation("map_values", "ny=new york"))
        self.assertEqual(out.tolist(), ["Ny", "new york"])

    def test_chained_with_a_cast_in_a_row(self):
        frame = pd.DataFrame({"flag": text("Y", "N", "maybe")})
        out, _ = transforms.transform_columns(
            frame, [row(source_col="flag", target_col="is_active", data_type="boolean", transformation_logic="map_values:maybe=")], "silver")
        self.assertEqual(out["is_active"].tolist()[:2], [True, False])
        self.assertTrue(pd.isna(out["is_active"].iloc[2]))

    def test_argument_validation(self):
        self.assertEqual(parse_value_map("a=b; c = d ;"), {"a": "b", "c": "d"})
        self.assertEqual(parse_value_map("x="), {"x": ""})
        for bad in ("", "a", "=b", "a=b;a=c", ";"):
            with self.assertRaises(ValueError, msg=bad):
                parse_value_map(bad)
        with self.assertRaises(ValueError):
            parse_value_map(";".join(f"v{i}=w" for i in range(201)))
        self.assertEqual(parse_transformation("trim|map_values:a=b;c=d", "silver")[1].arg, "a=b;c=d")
        with self.assertRaises(ValueError):
            parse_transformation("map_values", "silver")
        with self.assertRaises(ValueError):
            parse_transformation("map_values:a=b", "bronze")


class ValidRangeTests(unittest.TestCase):
    def test_values_outside_the_range_become_counted_nulls(self):
        frame = pd.DataFrame({"age": text("25", "-3", "250", "120", "x", None)})
        out, report = transforms.transform_columns(
            frame, [row(source_col="age", target_col="age", data_type="integer", transformation_logic="valid_range:0..120")], "silver")
        self.assertEqual(out["age"].tolist()[0], 25)
        self.assertEqual(out["age"].iloc[3], 120)
        self.assertEqual(int(out["age"].isna().sum()), 4)
        self.assertEqual(report["coerced_to_null"], {"age": 3})                     # -3, 250 and "x"; the real null is not counted

    def test_open_ended_ranges_and_decimals(self):
        numbers = text("-1", "0", "0.5", "7")
        self.assertEqual(transforms.apply_operation(numbers, Operation("valid_range", "0..")).dropna().tolist(), [0.0, 0.5, 7.0])
        self.assertEqual(transforms.apply_operation(numbers, Operation("valid_range", "..0.5")).dropna().tolist(), [-1.0, 0.0, 0.5])
        self.assertEqual(transforms.apply_operation(numbers, Operation("valid_range", "0.5..7")).dropna().tolist(), [0.5, 7.0])

    def test_then_fill_to_keep_the_row(self):
        frame = pd.DataFrame({"qty": text("2", "-5", "4")})
        rows = [row(source_col="qty", target_col="qty", data_type="integer", transformation_logic="valid_range:0..|fillna:median")]
        out, _ = transforms.transform_columns(frame, rows, "silver")
        cleaned, report = transforms.cleanse(out, rows)
        self.assertEqual(cleaned["qty"].tolist(), [2, 3, 4])
        self.assertEqual(report["filled"], {"qty": 1})

    def test_argument_and_type_validation(self):
        self.assertEqual(parse_value_range("0..120"), (0.0, 120.0))
        self.assertEqual(parse_value_range("..5"), (None, 5.0))
        for bad in ("", "5", "..", "5..1", "a..b", "1-2"):
            with self.assertRaises(ValueError, msg=bad):
                parse_value_range(bad)
        with self.assertRaisesRegex(ValueError, "data_type must be integer or float"):
            validate_rows([row(transformation_logic="valid_range:0..10", data_type="string")], "silver")
        validate_rows([row(transformation_logic="valid_range:0..10|fillna:zero", data_type="integer")], "silver")
        with self.assertRaises(ValueError):
            parse_transformation("valid_range:0..10", "bronze")

    def test_clean_number_argument(self):
        self.assertEqual(parse_transformation("clean_number:EU", "silver")[0].arg, "eu")
        with self.assertRaises(ValueError):
            parse_transformation("clean_number:us", "silver")


class QualityHintTests(unittest.TestCase):
    def hints(self, name, values, inferred=None):
        series = pd.Series(values)
        inferred = inferred or p.inferred_type(series, name)
        non_null = series.dropna()
        padded = int((non_null.astype(str) != non_null.astype(str).str.strip()).sum()) if inferred == "string" else 0
        return {h["op"]: h["reason"] for h in p.transformation_hints(name, series, inferred, padded, None, int(non_null.nunique()), len(non_null))}

    def test_punctuation_variants_get_a_map_values_suggestion(self):
        hints = self.hints("state", ["NY", "N.Y.", "ny", "CA", "ca", "C.A."] * 6 + ["NY"] * 10)
        chain = next(op for op in hints if "map_values" in op)
        self.assertTrue(chain.startswith("trim|lowercase|map_values:"))
        self.assertIn("n.y.=ny", chain)
        self.assertIn("c.a.=ca", chain)
        operations = parse_transformation(chain, "silver")                          # the suggestion is itself a valid rule
        self.assertEqual([o.name for o in operations], ["trim", "lowercase", "map_values"])

    def test_case_only_variants_still_suggest_lowercase_with_examples(self):
        hints = self.hints("status", ["Open", "OPEN", "open", "Closed", "closed"] * 4)
        self.assertIn("'Open'", hints["lowercase"])
        self.assertFalse(any("map_values" in op for op in hints))

    def test_distinct_categories_and_keys_get_no_map_suggestion(self):
        self.assertFalse(any("map_values" in op for op in self.hints("city", ["Paris", "London", "Berlin", "Madrid"] * 5)))
        self.assertFalse(any("map_values" in op for op in self.hints("product_code", ["A-1", "A1", "B-2", "B2"] * 5)))

    def test_decimal_comma_numbers_suggest_the_eu_mode_not_the_us_one(self):
        hints = self.hints("price", ["1.234,56", "99,5", "10", "7,25", "300", "1.000,00"] * 4)
        self.assertIn("clean_number:eu", hints)
        self.assertNotIn("clean_number", hints)
        us = self.hints("price", ["₹1,200", "₹3,450.50", "(50)", "$7"] * 5)
        self.assertIn("clean_number", us)
        self.assertNotIn("clean_number:eu", us)

    def test_mostly_numeric_text_suggests_a_cast_and_shows_the_bad_values(self):
        values = [str(i) for i in range(1, 40)] + ["unknown"]
        hints = self.hints("quantity", values, "string")
        self.assertIn("cast:integer", hints)
        self.assertIn("'unknown'", hints["cast:integer"])
        self.assertIn("cast:float", self.hints("weight", ["1.5", "2.25", "n/a", "3.1", "4.0", "5.5", "6.1", "7.2", "8.3", "9.4", "10.5"], "string"))

    def test_codes_and_clean_numbers_get_no_cast_suggestion(self):
        self.assertFalse(any(op.startswith("cast:") for op in self.hints("zip", ["12345", "K1A0B1"] * 10, "string")))
        self.assertFalse(any(op.startswith("cast:") for op in self.hints("product", [f"item{i}" for i in range(30)], "string")))
        self.assertFalse(any(op.startswith("cast:") for op in self.hints("label", ["a", "b", "1"] * 10, "string")))

    def test_impossible_values_suggest_valid_range(self):
        self.assertIn("valid_range:0..120", self.hints("customer_age", [25, 31, 250, 40, -2, 52]))
        self.assertNotIn("valid_range:0..120", self.hints("customer_age", [25, 31, 40, 52, 60]))
        self.assertIn("valid_range:0..", self.hints("quantity", [1, 2, -4, 3, 5, 6]))
        self.assertNotIn("valid_range:0..", self.hints("balance", [1, 2, -4, 3, 5, 6]))
        self.assertNotIn("valid_range:0..", self.hints("quantity", [1, 2, 4, 3, 5, 6]))

    def test_yes_no_text_suggests_a_boolean(self):
        self.assertIn("cast:boolean", self.hints("is_active", ["Y", "n", "yes", "No", "TRUE"] * 4))
        self.assertNotIn("cast:boolean", self.hints("is_active", [True, False, True, False]))


class ProfileIntegrationTests(unittest.TestCase):
    def test_notes_and_hints_reach_the_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = pd.DataFrame({
                "customer_age": [25, 31, 250, 40, -2, 52] * 5,
                "quantity": [("unknown" if i in (5, 17) else str(i % 6 + 1)) for i in range(30)],
                "price": ["1.234,56", "99,5", "10", "7,25", "300", "1.000,00"] * 5,
            })
            path = Path(directory) / "t.csv"
            frame.to_csv(path, index=False)
            profile, profile_path = profiler.build_profile([str(path)], "run_quality")
            dataset = profile["datasets"][0]
            ops = {c["name"]: [h["op"] for h in c.get("hints", [])] for c in dataset["columns"]}
            self.assertIn("valid_range:0..120", ops["customer_age"])
            self.assertIn("cast:integer", ops["quantity"])
            self.assertIn("clean_number:eu", ops["price"])
            joined = " ".join(dataset["quality_notes"])
            self.assertIn("customer_age: 10 age value(s) outside 0-120", joined)
            self.assertIn("quantity: 2 of 30", joined)
            self.assertEqual(json.loads(Path(profile_path).read_text(encoding="utf-8"))["datasets"][0]["table"], "t")


if __name__ == "__main__":
    unittest.main()
