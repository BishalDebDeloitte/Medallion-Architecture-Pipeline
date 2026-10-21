"""Offline tests for the improved profiling: types, date formats, hints, coverage, join discovery, redaction."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from agents import profiler
from core import profiling as p

DAYS = pd.date_range("2025-04-02", "2026-04-02", freq="D")


def write_csv(directory, name, frame):
    path = Path(directory) / name
    frame.to_csv(path, index=False)
    return str(path)


class TypeAndFormatTests(unittest.TestCase):
    def test_yyyymmdd_integers_are_dates_only_when_the_name_says_date(self):
        values = pd.Series([20250402, 20250403, 20250404, 20250405, 20250406])
        self.assertEqual(p.inferred_type(values, "date_id"), "date")
        self.assertEqual(p.inferred_type(values, "order_number"), "integer")
        self.assertEqual(p.inferred_type(pd.Series([20250402, 20250499, 1, 2, 3]), "date_id"), "integer")

    def test_date_format_detection(self):
        cases = {
            ("2025-04-02", "2026-01-15"): ("%Y-%m-%d", "iso"),
            ("2025-04-02 10:30:00", "2026-01-15 08:00:00"): ("%Y-%m-%d %H:%M:%S", "iso"),
            ("31/12/2024", "05/01/2024"): ("%d/%m/%Y", "unambiguous"),
            ("12/31/2024", "01/05/2024"): ("%m/%d/%Y", "unambiguous"),
            ("05/01/2024", "03/02/2024"): ("%m/%d/%Y", "ambiguous"),
            ("31-12-24", "05-01-24"): ("%d-%m-%y", "unambiguous"),
            ("2025/04/02", "2025/04/03"): ("%Y/%m/%d", "iso"),
            ("hello", "world"): (None, "unknown"),
        }
        for values, expected in cases.items():
            self.assertEqual(p.detect_date_format(pd.Series(list(values))), expected, values)
        self.assertEqual(p.detect_date_format(pd.Series([20250402, 20250403])), ("%Y%m%d", "unambiguous"))

    def test_row_numbers_are_recognised_but_real_ids_are_not(self):
        self.assertTrue(p.is_row_number("Index", pd.Series(range(1, 101)), "integer"))
        self.assertTrue(p.is_row_number("row_number", pd.Series(range(0, 50)), "integer"))
        self.assertFalse(p.is_row_number("customer_id", pd.Series(range(1, 101)), "integer"))   # a real key can be 1..n
        self.assertFalse(p.is_row_number("Index", pd.Series([1, 2, 4, 5]), "integer"))
        self.assertEqual(p.semantic_role("Index", "integer", 100, 100, row_number=True), "row_number")


class TimeCoverageTests(unittest.TestCase):
    def test_a_partial_last_month_is_flagged_and_a_nearly_full_first_month_is_not(self):
        coverage = p.time_coverage(pd.Series(DAYS))
        self.assertEqual(coverage["first"], "2025-04-02")
        self.assertEqual(coverage["partial_periods"], [{"period": "2026-04", "days_present": 2, "days_in_month": 30}])

    def test_a_mid_month_start_is_partial_too(self):
        days = pd.Series(pd.date_range("2025-04-16", "2025-09-30"))
        self.assertEqual([x["period"] for x in p.time_coverage(days)["partial_periods"]], ["2025-04"])

    def test_complete_months_and_sparse_columns_have_no_findings(self):
        self.assertEqual(p.time_coverage(pd.Series(pd.date_range("2025-01-01", "2025-12-31")))["partial_periods"], [])
        birthdays = pd.Series(pd.to_datetime(["1950-01-05", "1982-07-19", "1999-03-02"] * 30))
        self.assertIsNone(p.time_coverage(birthdays))
        self.assertIsNone(p.time_coverage(pd.Series(pd.date_range("2025-01-01", "2025-02-28"))))      # under three months


class HintTests(unittest.TestCase):
    def hints(self, name, values, inferred=None):
        series = pd.Series(values)
        inferred = inferred or p.inferred_type(series, name)
        fmt = p.detect_date_format(series) if inferred in {"date", "datetime"} else None
        non_null = series.dropna()
        padded = int((non_null.astype(str) != non_null.astype(str).str.strip()).sum()) if inferred == "string" else 0
        return {h["op"]: h["reason"] for h in p.transformation_hints(name, series, inferred, padded, fmt, int(non_null.nunique()), len(non_null))}

    def test_text_hints(self):
        self.assertIn("trim", self.hints("status", [" open", "closed ", "open"] * 5))
        self.assertIn("lowercase", self.hints("status", ["Open", "OPEN", "open", "Closed", "closed"] * 4))
        self.assertEqual(self.hints("status", ["open", "closed", "open"] * 5), {})

    def test_numbers_stored_as_text_but_not_ids_or_plain_numbers(self):
        self.assertIn("clean_number", self.hints("price", ["₹1,200", "₹3,450.50", "(50)", "$7"] * 5))
        self.assertNotIn("clean_number", self.hints("product_id", [f"PROD{i:06d}" for i in range(40)]))
        self.assertNotIn("clean_number", self.hints("phone", [str(6000000000 + i) for i in range(40)], "string"))
        self.assertNotIn("clean_number", self.hints("zip", ["12345", "23456", "34567"] * 5, "string"))

    def test_date_hints_carry_the_detected_format(self):
        self.assertIn("to_date", self.hints("d", ["2025-04-02", "2025-04-03", "2025-05-01"] * 3))
        self.assertIn("to_date:%d/%m/%Y", self.hints("d", ["31/12/2024", "05/01/2024"] * 5))
        ambiguous = self.hints("d", ["05/01/2024", "03/02/2024"] * 5)
        self.assertIn("to_date:%m/%d/%Y", ambiguous)
        self.assertIn("confirm", ambiguous["to_date:%m/%d/%Y"])
        self.assertIn("to_date:%Y%m%d", self.hints("date_id", [20250402, 20250403, 20250404, 20250405, 20250406]))

    def test_sign_convention_and_duplicate_key_hints(self):
        self.assertIn("abs", self.hints("refund_amount", [-10.5, -3.0, -215.79, -1.0, -7.0, 0.0]))
        self.assertNotIn("abs", self.hints("balance", [-10.5, 3.0, 215.79, -1.0, 7.0, 0.0]))
        self.assertIn("dedupe", self.hints("order_id", ["A", "A", "B", "B", "C", "D"]))
        self.assertNotIn("dedupe", self.hints("order_id", list("ABCDEF")))


class JoinDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def keys(self, **tables):
        paths = [write_csv(self.dir.name, f"{name}.csv", frame) for name, frame in tables.items()]
        profile, _ = profiler.build_profile(paths, f"join-{id(tables)}", "q")
        return profile["join_keys"], profile

    @staticmethod
    def link(keys, left, right):
        return next((k for k in keys if {k["left_table"], k["right_table"]} == {left, right}), None)

    def test_row_number_columns_are_never_join_keys(self):
        keys, profile = self.keys(
            orgs=pd.DataFrame({"Index": range(1, 101), "name": [f"org{i}" for i in range(100)]}),
            people=pd.DataFrame({"Index": range(1, 101), "name": [f"person{i}" for i in range(100)]}),
        )
        self.assertEqual(keys, [])
        index = next(c for c in profile["datasets"][0]["columns"] if c["name"] == "Index")
        self.assertEqual((index["row_number"], index["semantic_role"]), (True, "row_number"))
        self.assertTrue(any("row number" in note for note in profile["datasets"][0]["quality_notes"]))

    def test_unrelated_values_with_a_shared_column_name_are_not_joined(self):
        keys, _ = self.keys(
            a=pd.DataFrame({"website": [f"http://a{i}.com" for i in range(100)]}),
            b=pd.DataFrame({"website": [f"http://a{i}.com" for i in range(1)] + [f"http://b{i}.com" for i in range(99)]}),
        )
        self.assertEqual(keys, [])                                   # 1% overlap

    def test_a_foreign_key_to_a_differently_named_primary_key(self):
        keys, _ = self.keys(
            orders=pd.DataFrame({"order_id": range(50), "customer_ref": [f"C{i % 10}" for i in range(50)]}),
            customers=pd.DataFrame({"customer_id": [f"C{i}" for i in range(10)], "name": list("abcdefghij")}),
        )
        link = self.link(keys, "orders", "customers")
        self.assertEqual((link["left_column"], link["right_column"], link["relationship"]), ("customer_ref", "customer_id", "many_to_one"))
        self.assertEqual(link["overlap_ratio"], 1.0)
        self.assertTrue(any("names differ" in step for step in link["alignment"]))

    def test_dates_match_across_iso_text_and_yyyymmdd_integers_and_different_names(self):
        calendar = pd.DataFrame({
            "date_id": [int(d.strftime("%Y%m%d")) for d in DAYS], "full_date": [d.strftime("%Y-%m-%d") for d in DAYS],
            "year": DAYS.year,
        })
        fact = pd.DataFrame({
            "line_id": [f"L{i}" for i in range(400)],
            "date_id": [DAYS[i % 300].strftime("%Y-%m-%d") for i in range(400)],     # ISO text under the SAME name
            "order_date": [DAYS[i % 300].strftime("%Y-%m-%d") for i in range(400)],
        })
        keys, profile = self.keys(fact=fact, calendar=calendar)
        by_columns = {(k["left_column"], k["right_column"]): k for k in keys if k["left_table"] == "fact"}
        self.assertIn(("date_id", "date_id"), by_columns)                       # ISO text vs yyyymmdd ints: same dates
        order_links = [k for k in keys if (k["left_table"], k["left_column"], k["right_table"]) == ("fact", "order_date", "calendar")]
        self.assertEqual(len(order_links), 1)                                    # different names, same dates: ONE link, not one per calendar column
        self.assertEqual(order_links[0]["right_column"], "date_id")             # tie broken in favour of the key-like name
        alignment = " ".join(by_columns[("date_id", "date_id")]["alignment"])
        self.assertIn("to_date:%Y%m%d", alignment)
        self.assertIn("fact.date_id: to_date", alignment)
        calendar_id = next(c for d in profile["datasets"] if d["table"] == "calendar" for c in d["columns"] if c["name"] == "date_id")
        self.assertEqual((calendar_id["inferred_type"], calendar_id["date_format"]), ("date", "%Y%m%d"))

    def test_unrelated_names_need_a_large_exact_text_match_to_be_linked(self):
        small = self.keys(
            orders=pd.DataFrame({"order_id": range(50), "buyer_id": [f"C{i % 10}" for i in range(50)]}),
            customers=pd.DataFrame({"customer_id": [f"C{i}" for i in range(10)], "name": list("abcdefghij")}),
        )[0]
        self.assertEqual(small, [])                              # 10 distinct values and unrelated names: not enough evidence
        large = self.keys(
            sales=pd.DataFrame({"sale_id": range(200), "buyer_id": [f"C{i % 60:03d}" for i in range(200)]}),
            buyers=pd.DataFrame({"customer_id": [f"C{i:03d}" for i in range(60)], "name": [f"n{i}" for i in range(60)]}),
        )[0]
        self.assertEqual(len(large), 1)                          # 60 distinct text values match exactly: strong evidence
        self.assertEqual((large[0]["left_column"], large[0]["right_column"]), ("buyer_id", "customer_id"))

    def test_name_relation(self):
        for fk, pk, table in (("cust_ref", "customer_id", "customers"), ("customer_ref", "id", "customers"),
                              ("sellerId", "seller_id", "dim_seller"), ("product_code", "sku", "products")):
            self.assertTrue(p.names_related(fk, pk, table), (fk, pk))
        for fk, pk, table in (("buyer_id", "customer_id", "customers"), ("quantity", "payment_id", "payments"), ("a_id", "b_id", "b")):
            self.assertFalse(p.names_related(fk, pk, table), (fk, pk))

    def test_small_integer_columns_are_not_mistaken_for_ids(self):
        keys, _ = self.keys(
            lines=pd.DataFrame({"line_id": range(100), "quantity": [1, 2, 3, 4, 5] * 20}),
            payments=pd.DataFrame({"payment_id": range(1, 15), "method": list("abcdefghijklmn")}),
        )
        self.assertEqual(keys, [])

    def test_orphans_are_tolerated_for_same_named_keys(self):
        keys, _ = self.keys(
            orders=pd.DataFrame({"order_id": range(60), "customer_id": [i % 12 for i in range(60)]}),
            customers=pd.DataFrame({"customer_id": range(8), "name": list("abcdefgh")}),     # 4 of 12 customers missing
        )
        link = self.link(keys, "orders", "customers")
        self.assertAlmostEqual(link["overlap_ratio"], 8 / 12, places=3)

    def test_many_to_many_is_reported_with_a_warning(self):
        keys, _ = self.keys(
            a=pd.DataFrame({"a_id": range(20), "customer_id": [i % 5 for i in range(20)]}),
            b=pd.DataFrame({"b_id": range(30), "customer_id": [i % 5 for i in range(30)]}),
        )
        link = self.link(keys, "a", "b")
        self.assertEqual(link["relationship"], "many_to_many")
        self.assertIn("multiply", link["warning"])

    def test_each_column_links_to_one_column_of_another_table(self):
        calendar = pd.DataFrame({"date_id": [int(d.strftime("%Y%m%d")) for d in DAYS], "full_date": [d.strftime("%Y-%m-%d") for d in DAYS]})
        fact = pd.DataFrame({"line_id": range(500), "order_date": [DAYS[i % 300].strftime("%Y-%m-%d") for i in range(500)],
                             "return_date": [DAYS[(i % 300) + 1].strftime("%Y-%m-%d") for i in range(500)]})
        keys, _ = self.keys(fact=fact, calendar=calendar)
        self.assertEqual(sorted((k["left_column"], k["right_table"]) for k in keys if k["left_table"] == "fact"),
                         [("order_date", "calendar"), ("return_date", "calendar")])

    def test_the_archive_star_schema_pattern(self):
        keys, _ = self.keys(
            fact=pd.DataFrame({"line_id": range(300), "product_id": [f"P{i % 20:03d}" for i in range(300)],
                               "location_id": [f"L{i % 7}" for i in range(300)], "qty": [1, 2, 3] * 100}),
            product=pd.DataFrame({"product_id": [f"P{i:03d}" for i in range(20)], "brand": [f"b{i}" for i in range(20)]}),
            location=pd.DataFrame({"location_id": [f"L{i}" for i in range(7)], "region": list("NSEWCXY")}),
        )
        self.assertEqual({(k["left_table"], k["left_column"], k["right_table"]) for k in keys},
                         {("fact", "product_id", "product"), ("fact", "location_id", "location")})
        self.assertTrue(all(k["relationship"] == "many_to_one" and "alignment" not in k for k in keys))


class SensitiveProfileTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        cards = ["4111-1111-1111-1111", "5500-0000-0000-0004", "3400-0000-0000-009", "6011-0000-0000-0004", "4012-8888-8888-1881"] * 4
        self.path = write_csv(self.dir.name, "customers.csv", pd.DataFrame({
            "customer_id": [f"C{i}" for i in range(20)], "credit_card_number": cards,
            "email": [f"user{i}@example.com" for i in range(20)], "city": ["Pune", "Delhi"] * 10,
        }))

    def test_profile_hides_values_stats_and_hints_of_sensitive_columns(self):
        profile, path = profiler.build_profile([self.path], "sens-1", "q")
        columns = {c["name"]: c for c in profile["datasets"][0]["columns"]}
        self.assertEqual(columns["credit_card_number"]["sensitivity"], "financial")
        self.assertEqual(columns["credit_card_number"]["sample_values"], ["[redacted: financial]"])
        self.assertEqual(columns["email"]["sample_values"], ["[redacted: contact]"])
        self.assertEqual(columns["city"]["sample_values"], ["Pune", "Delhi"])
        for name in ("credit_card_number", "email"):
            self.assertNotIn("hints", columns[name])
        self.assertEqual({(s["column"], s["kind"]) for s in profile["sensitive_columns"]}, {("credit_card_number", "financial"), ("email", "contact")})
        self.assertTrue(any("hidden from the AI" in n for n in profile["datasets"][0]["quality_notes"]))
        saved = path.read_text(encoding="utf-8")
        self.assertNotIn("4111", saved)
        self.assertNotIn("user1@example.com", saved)

    def test_the_inspect_tool_preview_is_redacted_as_well(self):
        preview = profiler.inspect_csv(self.path)
        columns = {c["name"]: c for c in preview["columns"]}
        self.assertEqual(columns["credit_card_number"]["sample_values"], ["[redacted: financial]"])
        self.assertNotIn("4111", json.dumps(preview))

    def test_real_values_are_still_in_the_data_the_pipeline_processes(self):
        from core.csv_io import read_csv

        frame, _ = read_csv(self.path)
        self.assertEqual(frame["credit_card_number"].iloc[0], "4111-1111-1111-1111")


class NotesTests(unittest.TestCase):
    def test_birth_dates_never_get_partial_period_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            people = pd.DataFrame({"person_id": range(len(DAYS)), "date_of_birth": [d.strftime("%Y-%m-%d") for d in DAYS]})
            profile, _ = profiler.build_profile([write_csv(directory, "people.csv", people)], "notes-2", "q")
        self.assertNotIn("time_coverage", profile["datasets"][0])
        self.assertFalse(any("partial period" in n for n in profile["datasets"][0]["quality_notes"]))

    def test_partial_period_and_ambiguous_date_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            orders = pd.DataFrame({"order_id": range(len(DAYS)), "order_date": [d.strftime("%Y-%m-%d") for d in DAYS],
                                   "ship_date": [d.strftime("%d/%m/%Y") if d.day > 12 else d.strftime("%m/%d/%Y") for d in DAYS][:0] or ["05/01/2024"] * len(DAYS)})
            profile, _ = profiler.build_profile([write_csv(directory, "orders.csv", orders)], "notes-1", "q")
        notes = profile["datasets"][0]["quality_notes"]
        self.assertTrue(any("2026-04 has data for only 2 of 30 days" in n for n in notes), notes)
        self.assertTrue(any("ship_date: day/month order is ambiguous" in n for n in notes), notes)
        self.assertEqual(profile["datasets"][0]["time_coverage"]["order_date"]["partial_periods"][0]["period"], "2026-04")


if __name__ == "__main__":
    unittest.main()
