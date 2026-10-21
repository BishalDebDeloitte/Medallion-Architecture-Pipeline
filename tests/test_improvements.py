"""Offline tests for the data-quality improvements: what the agents see, redaction, partial periods, reports in the UI."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from agents import profiler, reporter, silver_agent, sttm_generator
from agents.orchestrator import Orchestrator
from agents.sttm_generator import SttmInputs
from app import ui_logic as ui
from core.layer_agent import write_layer_report
from core.paths import gold_file, report_file, silver_file
from core.state import PipelineState
from core.sttm_schema import validate_rows
from tests.test_phase_loop import FakeSupervisor, FakeToolkit
from tests.test_reporter_agent import final, gold_files, query, script

DAYS = pd.date_range("2025-04-02", "2026-04-02", freq="D")
CARDS = ["4111-1111-1111-1111", "5500-0000-0000-0004", "3400-0000-0000-009", "6011-0000-0000-0004", "4012-8888-8888-1881"]
APP_PATH = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


def texts(elements):
    return " ".join(str(getattr(e, "value", "")) for e in elements)


class SttmAgentSeesTheRightThingsTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        root = Path(self.dir.name)
        n = len(DAYS)
        fact = pd.DataFrame({
            "line_id": [f"L{i}" for i in range(n)], "order_date": [d.strftime("%Y-%m-%d") for d in DAYS],
            "status": [" open" if i % 2 else "OPEN" for i in range(n)], "refund_amount": [-1.0 * (i % 7) - 1 for i in range(n)],
            "credit_card_number": [CARDS[i % 5] for i in range(n)],
        })
        calendar = pd.DataFrame({"date_id": [int(d.strftime("%Y%m%d")) for d in DAYS], "full_date": [d.strftime("%Y-%m-%d") for d in DAYS]})
        paths = []
        for name, frame in (("fact", fact), ("calendar", calendar)):
            frame.to_csv(root / f"{name}.csv", index=False)
            paths.append(str(root / f"{name}.csv"))
        _, self.profile_path = profiler.build_profile(paths, "imp-1", "revenue by month")
        self.context = sttm_generator.load_context(SttmInputs("bronze", "imp-1", "revenue by month", profile_path=str(self.profile_path)))

    def test_context_carries_hints_coverage_and_sensitivity(self):
        fact = self.context["tables"]["fact"]
        columns = {c["name"]: c for c in fact["columns"]}
        self.assertIn("trim", [h["op"] for h in columns["status"]["hints"]])
        self.assertIn("abs", [h["op"] for h in columns["refund_amount"]["hints"]])
        self.assertEqual(fact["time_coverage"]["order_date"]["partial_periods"][0]["period"], "2026-04")
        self.assertEqual(columns["credit_card_number"]["sensitivity"], "financial")
        self.assertEqual(columns["credit_card_number"]["sample_values"], ["[redacted: financial]"])
        self.assertNotIn("4111", json.dumps(self.context))
        calendar_id = next(c for c in self.context["tables"]["calendar"]["columns"] if c["name"] == "date_id")
        self.assertEqual(calendar_id["date_format"], "%Y%m%d")

    def test_the_date_link_and_its_alignment_steps_reach_the_agent(self):
        links = {frozenset({k["left_column"], k["right_column"]}): k for k in self.context["join_keys"]}
        self.assertIn(frozenset({"order_date", "full_date"}), links)
        self.assertIn(frozenset({"order_date", "date_id"}), links)         # ISO text vs yyyymmdd integers: the same dates
        steps = " ".join(links[frozenset({"order_date", "date_id"})]["alignment"])
        self.assertIn("to_date:%Y%m%d", steps)
        self.assertIn("names differ", steps)

    def test_the_inspect_tool_shows_all_of_it(self):
        inspect, _ = sttm_generator.build_sttm_tools(
            SttmInputs("bronze", "imp-1", "q", profile_path=str(self.profile_path)), {},
        )
        payload = inspect()
        self.assertIn("partial_periods", payload)
        self.assertIn("alignment", payload)
        self.assertNotIn("4111", payload)

    def test_keeping_a_sensitive_column_produces_a_warning_not_an_error(self):
        rows = validate_rows([
            {"layer": "bronze", "target_table": "fact", "source_table": "fact", "source_col": "credit_card_number",
             "target_col": "card", "data_type": "string"},
            {"layer": "bronze", "target_table": "fact", "source_table": "fact", "source_col": "status",
             "target_col": "status", "data_type": "string"},
            {"layer": "bronze", "target_table": "calendar", "source_table": "calendar", "source_col": "full_date",
             "target_col": "full_date", "data_type": "string"},
        ], "bronze")
        warnings = sttm_generator.mapping_warnings(rows, self.context)
        self.assertEqual(len(warnings), 1)
        self.assertIn("fact.credit_card_number (financial)", warnings[0])
        self.assertNotIn("status", warnings[0])
        self.assertEqual(sttm_generator.mapping_warnings(rows[1:], self.context), [])

    def test_the_prompts_tell_the_agent_how_to_use_the_new_signals(self):
        silver = sttm_generator.sttm_system_prompt("silver")
        for phrase in ("alignment", "to_date:%Y%m%d", "clean_number", "abs", "row_number", "SAME name and type"):
            self.assertIn(phrase, silver)
        gold = sttm_generator.sttm_system_prompt("gold")
        for phrase in ("date_part:year_month", "partial period", "numerator and denominator"):
            self.assertIn(phrase, gold)
        self.assertIn("sensitivity", sttm_generator.sttm_system_prompt("bronze"))


class ParquetPreviewsAreRedactedTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"customer_id": ["C1", "C2", "C3", "C4", "C5", "C6"], "card_number": CARDS + [CARDS[0]], "city": list("abcdef")})

    def test_the_silver_inspect_preview_and_the_sttm_context_hide_cards(self):
        preview = silver_agent._preview("customers", "x.parquet", self.frame())
        by_name = {c["name"]: c for c in preview["columns"]}
        self.assertEqual(by_name["card_number"]["sample_values"], ["[redacted: financial]"])
        self.assertEqual(by_name["city"]["sample_values"], ["a", "b", "c"])
        path = silver_file("customers", "imp-2").parent
        path.mkdir(parents=True, exist_ok=True)
        from core.paths import bronze_file

        target = bronze_file("customers", "imp-2")
        target.parent.mkdir(parents=True, exist_ok=True)
        self.frame().to_parquet(target, index=False)
        context = sttm_generator.load_context(SttmInputs("silver", "imp-2", "q", parquet_paths=(str(target),)))
        self.assertNotIn("4111", json.dumps(context))
        warning = sttm_generator.mapping_warnings(
            validate_rows([{"layer": "silver", "target_table": "customers", "source_table": "customers",
                            "source_col": "card_number", "target_col": "card", "data_type": "string"}], "silver"), context)
        self.assertEqual(len(warning), 1)


class ReporterPartialPeriodTests(unittest.TestCase):
    def test_gold_date_columns_are_checked_and_the_agent_is_told(self):
        daily = pd.DataFrame({"order_date": DAYS, "total_revenue": 1.0})
        path = gold_file("daily_revenue", "imp-3")
        path.parent.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(path, index=False)
        session = reporter.ReporterSession([str(path)], "imp-3")
        result = session.load()
        coverage = result["tables"][0]["time_coverage"]["order_date"]
        self.assertEqual(coverage["partial_periods"], [{"period": "2026-04", "days_present": 2, "days_in_month": 30}])
        self.assertEqual(result["partial_periods"], {"daily_revenue": {"2026-04": "2 of 30 days"}})

    def test_the_raw_profile_supplies_partial_periods_even_when_gold_only_has_month_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile.json"
            profile.write_text(json.dumps({"datasets": [{"table": "fact", "time_coverage": {"order_date": {
                "partial_periods": [{"period": "2024-04", "days_present": 2, "days_in_month": 30}]}}}]}), encoding="utf-8")
            session = reporter.ReporterSession(gold_files("imp-4"), "imp-4", profile_path=str(profile))
            self.assertEqual(session.load()["partial_periods"], {"(source data)": {"2024-04": "2 of 30 days"}})   # no mapping to trace

    def test_a_chart_that_shows_a_partial_month_gets_a_visible_warning_and_a_clean_chart_does_not(self):
        scratchpad = {}
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile.json"
            profile.write_text(json.dumps({"datasets": [{"table": "fact", "time_coverage": {"order_date": {
                "partial_periods": [{"period": "2024-04", "days_present": 2, "days_in_month": 30}]}}}]}), encoding="utf-8")
            month_sql = "SELECT strftime(month, '%Y-%m') AS month, sum(revenue) AS revenue FROM monthly_sales {where} GROUP BY 1 ORDER BY 1"
            report = {
                "headline": "h", "summary": "s",
                "charts": [
                    {"title": "All months", "type": "line", "query_id": "q1", "x": "month", "y": "revenue"},
                    {"title": "Full months only", "type": "line", "query_id": "q2", "x": "month", "y": "revenue"},
                ],
            }
            model = script(query(month_sql.format(where=""), "c3"), query(month_sql.format(where="WHERE month < DATE '2024-04-01'"), "c4"), final(report))
            result = json.loads(reporter.run_reporter_agent(
                gold_files("imp-5"), "Monthly revenue?", "imp-5", "go", scratchpad, model=model, profile_path=str(profile),
            ))
        self.assertEqual(result["status"], "success", result)
        html = report_file("imp-5").read_text(encoding="utf-8")
        self.assertEqual(html.count("chart-note\">"), 1)
        self.assertIn("2024-04 covers only 2 of 30 days", html)
        data = json.loads(Path(scratchpad["report_json_path"]).read_text(encoding="utf-8"))
        self.assertEqual([c["notes"] for c in data["charts"]], [["2024-04 covers only 2 of 30 days, so it is not comparable with full periods."], []])
        self.assertEqual(data["partial_periods"]["q1"], {"2024-04": "2 of 30 days"})

    def test_a_partial_month_is_tied_to_the_date_column_a_table_was_built_from(self):
        profile = {"datasets": [
            {"table": "FACT", "time_coverage": {"order_date": {"partial_periods": [{"period": "2024-04", "days_present": 2, "days_in_month": 30}]},
                                                 "return_date": {"partial_periods": [{"period": "2024-02", "days_present": 20, "days_in_month": 29}]}}},
            {"table": "CUSTOMER", "time_coverage": {"registered": {"partial_periods": [{"period": "2024-04", "days_present": 1, "days_in_month": 30}]}}},
        ]}
        rows = [  # monthly_sales is built from FACT.order_date only
            {"layer": "gold", "target_table": "monthly_sales", "source_table": "FACT", "source_col": "order_date", "target_col": "month",
             "data_type": "string", "transformation_logic": "date_part:year_month", "description": "month"},
            {"layer": "gold", "target_table": "sales_by_customer", "source_table": "FACT", "source_col": "amount", "target_col": "total",
             "data_type": "float", "aggregation": "sum", "description": "total"},
        ]
        from core.sttm_schema import write_sttm

        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profile.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            sttm_path = write_sttm(Path(directory) / "sttm_gold.csv", rows, "gold")
            session = reporter.ReporterSession(gold_files("imp-9"), "imp-9", str(sttm_path), str(profile_path))
            load = session.load()
        self.assertEqual(load["partial_periods"]["monthly_sales"], {"2024-04": "2 of 30 days"})   # not the customer's 1 of 30, not return_date's Feb
        self.assertNotIn("sales_by_customer", load["partial_periods"])
        self.assertEqual(session.partial_periods_for("SELECT month FROM monthly_sales"), {"2024-04": "2 of 30 days"})
        self.assertEqual(session.partial_periods_for("SELECT total FROM sales_by_customer"), {})
        self.assertEqual(session.partial_periods_for("SELECT * FROM monthly_sales_archive"), {})        # whole-word table match only

    def test_sensitive_columns_are_masked_in_previews_and_query_results_sent_to_the_model(self):
        frame = pd.DataFrame({"customer": list("abcdef"), "credit_card_number": CARDS + [CARDS[0]], "total": [1.0] * 6})
        path = gold_file("customers_gold", "imp-6")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        session = reporter.ReporterSession([str(path)], "imp-6")
        self.assertNotIn("4111", json.dumps(session.preview()))
        session.load()
        result = session.execute("SELECT customer, credit_card_number, total FROM customers_gold")
        self.assertEqual(result["rows"][0]["credit_card_number"], "[redacted: financial]")
        self.assertEqual(result["rows"][0]["customer"], "a")
        self.assertIn("4111", session.queries["q1"].frame["credit_card_number"].iloc[0])      # the real data is intact


class LayerReportTests(unittest.TestCase):
    def test_the_report_file_is_written_and_summarised_in_plain_words(self):
        tables = [{"table": "orders", "source_rows": 6, "output_rows": 4, "rows_dropped_dropna": 1, "rows_dropped_dedupe": 1,
                   "filled": {"amount": 2}, "coerced_to_null": {"order_date": 1}}]
        path = write_layer_report("silver", "imp-7", tables)
        report = ui.load_layer_report(path)
        table = ui.layer_report_table(report)
        self.assertEqual(table.iloc[0][["Table", "Rows in", "Rows out", "Dropped (missing key)", "Dropped (duplicates)", "Values filled",
                                        "Values that could not be converted"]].tolist(), ["orders", 6, 4, 1, 1, 2, 1])
        self.assertIn("order_date: 1 unconvertible", table.iloc[0]["Details"])
        gold = ui.layer_report_table({"tables": [{"table": "t", "source_tables": ["a", "b"], "join_keys": ["k"], "merged_rows": 10,
                                                  "output_rows": 3, "group_by": ["x"], "measures": {"total": "sum(a.amount)"}}]})
        self.assertEqual(gold.iloc[0]["Joined on"], "k")
        self.assertIsNone(ui.load_layer_report(""))
        self.assertIsNone(ui.load_layer_report(str(Path(path).with_name("missing.json"))))

    def test_report_paths_flow_from_the_tools_into_the_run_state(self):
        # the Bronze/Silver/Gold tools set <layer>_report_path; the Orchestrator copies it into PipelineState (end-to-end test)
        from tests import test_end_to_end

        self.assertIn("bronze_report_path", PipelineState.model_fields)
        self.assertTrue(hasattr(test_end_to_end, "EndToEndTests"))

    def test_helpers_for_the_review_screen(self):
        profile = {"sensitive_columns": [{"table": "c", "column": "ref", "kind": "financial"}],
                   "datasets": [{"table": "c", "columns": [{"name": "status", "hints": [{"op": "trim", "reason": "padded"}]}]}],
                   "join_keys": [{"left_table": "o", "left_column": "customer_ref", "right_table": "c", "right_column": "customer_id",
                                  "overlap_ratio": 0.98, "relationship": "many_to_one", "alignment": ["names differ: ..."]}]}
        rows = [{"source_table": "c", "source_col": "ref"}, {"source_table": "c", "source_col": "email"}, {"source_table": "c", "source_col": "city"}]
        self.assertEqual(ui.sensitive_in_mapping(rows, profile), ["c.email (contact)", "c.ref (financial)"])
        self.assertEqual(ui.hint_table(profile).iloc[0].to_dict(), {"Table": "c", "Column": "status", "Suggested operation": "trim", "Why": "padded"})
        link = ui.join_table(profile).iloc[0]
        self.assertEqual((link["Many side"], link["One side"], link["Match"], link["Type"]), ("o.customer_ref", "c.customer_id", "98%", "many to one"))
        self.assertEqual(ui.previous_step_report_path(PipelineState(bronze_report_path="b", silver_report_path="s"), "silver"), "b")
        self.assertEqual(ui.previous_step_report_path(PipelineState(bronze_report_path="b", silver_report_path="s"), "gold"), "s")
        self.assertEqual(ui.previous_step_report_path(PipelineState(bronze_report_path="b"), "bronze"), "")


class ReviewScreenTests(unittest.TestCase):
    def open(self, state, orchestrator):
        app = AppTest.from_file(str(APP_PATH), default_timeout=30)
        app.session_state["orchestrator"], app.session_state["state"] = orchestrator, state
        return app.run()

    def test_a_mapping_that_keeps_a_sensitive_column_shows_a_warning(self):
        orch = Orchestrator(toolkit=FakeToolkit(), supervisor=FakeSupervisor())
        state = orch.start_run(["orders.csv"], "q")
        from core.sttm_schema import write_sttm

        rows = [{"layer": "bronze", "target_table": "orders", "source_table": "orders", "source_col": "credit_card_number",
                 "target_col": "card", "data_type": "string"}]
        write_sttm(state.pending_approval_path, rows, "bronze")
        app = self.open(state, orch)
        self.assertFalse(app.exception)
        self.assertIn("orders.credit_card_number (financial)", texts(app.warning))
        self.assertIn("Approval 1 of 3", texts(app.caption))

    def test_the_silver_review_shows_what_the_bronze_step_did(self):
        toolkit = FakeToolkit()
        orch = Orchestrator(toolkit=toolkit, supervisor=FakeSupervisor())
        state = orch.start_run(["orders.csv"], "q")
        orch.approve(state)
        orch.run_next_phase(state)
        state.bronze_report_path = write_layer_report("bronze", state.run_id, [
            {"table": "orders", "source_rows": 790732, "output_rows": 790732, "coerced_to_null": {}}])
        app = self.open(state, orch)
        self.assertFalse(app.exception)
        self.assertIn("What the last step did", [tab.label for tab in app.tabs])
        report = next(f.value for f in app.dataframe if "Rows out" in f.value.columns)
        self.assertEqual(report.iloc[0]["Rows out"], 790732)
        self.assertIn("Approval 2 of 3", texts(app.caption))

    def test_the_bronze_review_has_the_profile_tab_with_links_and_suggestions(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = pd.DataFrame({"order_id": range(10), "status": [" open", "closed"] * 5})
            frame.to_csv(Path(directory) / "orders.csv", index=False)
            _, profile_path = profiler.build_profile([str(Path(directory) / "orders.csv")], "imp-8", "q")
            orch = Orchestrator(toolkit=FakeToolkit(), supervisor=FakeSupervisor())
            state = orch.start_run(["orders.csv"], "q")
            state.profile_path = str(profile_path)
            app = self.open(state, orch)
            self.assertFalse(app.exception)
            self.assertEqual([tab.label for tab in app.tabs], ["Mapping", "Source data", "Data profile"])
            self.assertTrue(any("Suggested operation" in f.value.columns for f in app.dataframe))


if __name__ == "__main__":
    unittest.main()
