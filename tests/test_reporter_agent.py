"""Offline tests for the Reporter agent (Step 6)."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from agents import reporter
from agents.orchestrator import safe_tool
from agents.reporter import ReporterSession
from core import memory
from core.paths import gold_file, report_data_file, report_file, trace_file
from core.sttm_schema import write_sttm
from tests.fakes import ScriptedChatModel, ai, last_tool_result, tool_call

SALES = pd.DataFrame({
    "customer_name": pd.array(["Ann", "Bob", "Cy", "Di", "Eve"], dtype="string"),
    "total_amount": pd.array([300.0, 250.0, 120.0, 80.0, 50.0], dtype="Float64"),
    "order_count": pd.array([6, 5, 3, 2, 1], dtype="Int64"),
})
MONTHLY = pd.DataFrame({
    "month": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"] * 2),
    "region": ["North"] * 4 + ["South"] * 4,
    "revenue": [10.0, 12.0, 15.0, 20.0, 8.0, 9.0, 14.0, 13.0],
})

TOP_CUSTOMERS = "SELECT customer_name, total_amount FROM sales_by_customer ORDER BY total_amount DESC"
TOTAL = "SELECT sum(total_amount) AS total FROM sales_by_customer"
GOOD_REPORT = {
    "headline": "Ann is the top customer with 300 in sales.",
    "summary": "Five customers generated 800 in revenue; the top two account for most of it.",
    "kpis": [{"label": "Total revenue", "query_id": "q2", "column": "total", "prefix": "$"}],
    "insights": [{"title": "Concentration", "detail": "Ann and Bob together buy 550 of 800.", "query_id": "q1"}],
    "charts": [{"title": "Revenue by customer", "type": "bar", "query_id": "q1", "x": "customer_name",
                "y": "total_amount", "description": "Total amount per customer."}],
    "limitations": ["Only five customers are in the data."],
}


def gold_files(run_id):
    paths = []
    for table, frame in (("sales_by_customer", SALES), ("monthly_sales", MONTHLY)):
        frame = frame.copy()
        frame.insert(0, f"pk_{table}_gold_id", pd.array(range(1, len(frame) + 1), dtype="Int64"))
        path = gold_file(table, run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        paths.append(str(path))
    return paths


def script(*steps):
    """Standard exploration (inspect, load, then the given tool calls) followed by the given steps."""
    return ScriptedChatModel(script=[
        ai("Plan: preview, load, query, then report.", tool_call("inspect_gold_tables_tool", "c1")),
        ai("Loading.", tool_call("load_gold_data_tool", "c2")),
        *steps,
    ])


def query(sql, call_id):
    return ai("Querying.", tool_call("execute_query_tool", call_id, sql=sql))


def final(report):
    return ai(json.dumps(report))


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.session = ReporterSession(gold_files("rp-session"), "rp-session")

    def execute(self, sql):
        self.session.load()
        return self.session.execute(sql)

    def test_preview_is_cheap_and_load_returns_the_exact_catalog(self):
        preview = {t["table"]: t for t in self.session.preview()}
        self.assertEqual(preview["sales_by_customer"]["row_count"], 5)
        self.assertEqual(len(preview["sales_by_customer"]["sample_rows"]), 3)
        self.assertIsNone(self.session.connection)                         # nothing loaded yet
        catalog = {t["table"]: t for t in self.session.load()["tables"]}
        columns = {c["name"]: c["type"] for c in catalog["monthly_sales"]["columns"]}
        self.assertEqual(columns["revenue"], "DOUBLE")
        self.assertEqual(catalog["monthly_sales"]["row_count"], 8)

    def test_column_descriptions_come_from_the_gold_sttm(self):
        sttm = Path(tempfile.mkdtemp()) / "sttm_gold.csv"
        write_sttm(sttm, [{
            "layer": "gold", "target_table": "sales_by_customer", "source_table": "orders", "source_col": "amount",
            "target_col": "total_amount", "data_type": "float", "transformation_logic": "", "aggregation": "sum",
            "nullable": "true", "join_key": "", "description": "Order value in USD",
        }], "gold")
        session = ReporterSession(gold_files("rp-desc"), "rp-desc", str(sttm))
        columns = {c["name"]: c for c in session.load()["tables"][0]["columns"]}
        self.assertEqual(columns["total_amount"]["description"], "Order value in USD")
        self.assertNotIn("description", columns["order_count"])

    def test_select_and_with_run_and_results_are_stored_in_full(self):
        first = self.execute(TOP_CUSTOMERS)
        self.assertEqual((first["query_id"], first["row_count"]), ("q1", 5))
        self.assertEqual(first["rows"][0], {"customer_name": "Ann", "total_amount": 300.0})
        second = self.session.execute("-- totals\nWITH t AS (SELECT * FROM sales_by_customer) SELECT count(*) AS n FROM t;")
        self.assertEqual(second["query_id"], "q2")
        self.assertEqual(self.session.queries["q1"].frame["total_amount"].tolist(), [300.0, 250.0, 120.0, 80.0, 50.0])

    def test_only_a_single_read_only_select_is_accepted(self):
        self.session.load()
        for sql in (
            "DROP TABLE sales_by_customer", "DELETE FROM sales_by_customer", "COPY sales_by_customer TO 'x.csv'",
            "SELECT 1; SELECT 2", "PRAGMA database_list", "ATTACH 'other.db'", "INSTALL httpfs", "SET threads=1",
            "CREATE TABLE x AS SELECT 1", "", "   ",
        ):
            with self.assertRaises(ValueError, msg=sql):
                self.session.execute(sql)
            self.session._errors_in_a_row = 0
        self.assertEqual(self.execute("SELECT count(*) AS n FROM sales_by_customer")["rows"], [{"n": 5}])

    def test_the_sandbox_blocks_file_access_even_inside_a_select(self):
        secret = Path(tempfile.mkdtemp()) / "secret.csv"
        secret.write_text("a\n1\n", encoding="utf-8")
        for sql in (
            f"SELECT * FROM read_csv('{secret.as_posix()}')", f"SELECT * FROM read_text('{secret.as_posix()}')",
            f"SELECT * FROM read_parquet('{self.session.paths['sales_by_customer']}')", "SELECT * FROM glob('*')",
        ):
            with self.assertRaisesRegex(ValueError, "SQL error"):
                self.execute(sql)
            self.session._errors_in_a_row = 0

    def test_configuration_cannot_be_unlocked(self):
        self.session.load()
        with self.assertRaises(ValueError):
            self.session.execute("SET enable_external_access=true")
        self.assertEqual(
            self.session.connection.execute("SELECT current_setting('enable_external_access')").fetchone()[0], False,
        )

    def test_sql_errors_name_the_tables_and_stop_after_two_in_a_row(self):
        self.session.load()
        with self.assertRaisesRegex(ValueError, "Available tables.*sales_by_customer"):
            self.session.execute("SELECT nope FROM sales_by_customer")
        with self.assertRaises(ValueError):
            self.session.execute("SELECT nope FROM sales_by_customer")
        with self.assertRaisesRegex(ValueError, "failed twice in a row"):
            self.session.execute(TOP_CUSTOMERS)
        self.assertEqual(self.session.queries, {})

    def test_a_success_resets_the_error_streak(self):
        self.session.load()
        with self.assertRaises(ValueError):
            self.session.execute("SELECT nope FROM sales_by_customer")
        self.session.execute(TOP_CUSTOMERS)
        with self.assertRaises(ValueError):
            self.session.execute("SELECT nope FROM sales_by_customer")
        self.assertEqual(self.session.execute(TOTAL)["query_id"], "q2")

    def test_query_budget_and_result_size_are_bounded(self):
        self.session.load()
        with self.assertRaisesRegex(ValueError, "more than 20000 rows"):
            self.session.execute("SELECT * FROM range(25000)")
        self.session._errors_in_a_row = 0
        self.session._attempts = reporter.MAX_QUERIES
        with self.assertRaisesRegex(ValueError, "budget"):
            self.session.execute(TOTAL)

    def test_execute_requires_load_and_inputs_are_validated(self):
        with self.assertRaisesRegex(ValueError, "load_gold_data_tool"):
            self.session.execute(TOTAL)
        with self.assertRaises(ValueError):
            ReporterSession([], "rp-x")
        with self.assertRaises(ValueError):
            ReporterSession(gold_files("rp-a"), "rp-b")                   # artifacts of another run


class ToolTests(unittest.TestCase):
    def test_tool_order_is_enforced_and_results_are_json(self):
        session = ReporterSession(gold_files("rp-tools"), "rp-tools")
        inspect, load, execute = reporter.build_reporter_tools(session)
        self.assertEqual([t.__name__ for t in (inspect, load, execute)],
                         ["inspect_gold_tables_tool", "load_gold_data_tool", "execute_query_tool"])
        early = json.loads(load())
        self.assertEqual(early["status"], "error")
        self.assertIn("inspect_gold_tables_tool", early["message"])
        self.assertEqual(json.loads(inspect())["status"], "success")
        self.assertIn("load_gold_data_tool", json.loads(execute(sql=TOTAL))["message"])
        catalog = json.loads(load())
        self.assertEqual({t["table"] for t in catalog["tables"]}, {"sales_by_customer", "monthly_sales"})
        result = json.loads(execute(sql=TOTAL))
        self.assertEqual((result["status"], result["query_id"], result["rows"]), ("success", "q1", [{"total": 800.0}]))
        bad = json.loads(execute(sql="DROP TABLE sales_by_customer"))
        self.assertEqual(bad["status"], "error")
        for tool in (inspect, load, execute):
            self.assertGreater(len(tool.__doc__), 60)


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.scratchpad = {}

    def run_agent(self, run_id, model, intent="Which customers buy the most?", sttm=None):
        return json.loads(reporter.run_reporter_agent(
            gold_files(run_id), intent, run_id, "write the executive report", self.scratchpad,
            sttm_gold_path=sttm, model=model,
        ))

    def read_trace(self, run_id):
        return json.loads(trace_file("reporter", run_id).read_text(encoding="utf-8"))[-1]

    def test_full_flow_writes_report_json_memory_and_trace(self):
        model = script(query(TOP_CUSTOMERS, "c3"), query(TOTAL, "c4"), final(GOOD_REPORT))
        result = self.run_agent("rp-ok", model)
        self.assertEqual(result["status"], "success", result)
        self.assertEqual((result["chart_count"], result["query_count"]), (1, 2))

        html_path, json_path = report_file("rp-ok"), report_data_file("rp-ok")
        self.assertEqual((self.scratchpad["report_path"], self.scratchpad["report_json_path"]), (str(html_path), str(json_path)))
        html = html_path.read_text(encoding="utf-8")
        for expected in ("Ann is the top customer", "$800", "Revenue by customer", "View data table", "Only five customers"):
            self.assertIn(expected, html)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(data["kpis"][0]["value"], 800.0)
        self.assertEqual(data["queries"][0]["sql"], TOP_CUSTOMERS)
        self.assertEqual({t["name"] for t in data["tables"]}, {"sales_by_customer", "monthly_sales"})

        self.assertEqual(memory.get_document("intent_rp-ok")["document"], "Which customers buy the most?")
        stored = memory.get_document("report_rp-ok")
        self.assertIn("Ann is the top customer", stored["document"])
        self.assertEqual(stored["metadata"]["report_path"], str(html_path))

        trace = self.read_trace("rp-ok")
        self.assertEqual(trace["status"], "success")
        self.assertEqual([c["name"] for c in trace["tool_calls"]],
                         ["inspect_gold_tables_tool", "load_gold_data_tool", "execute_query_tool", "execute_query_tool"])
        self.assertIn("Plan", trace["plan"])
        self.assertEqual(trace["output"]["chart_count"], 1)

    def test_agent_corrects_a_failed_query_once(self):
        model = script(
            query("SELECT customer, total_amount FROM sales_by_customer", "c3"),
            lambda messages: (
                self.assertIn("Available tables", last_tool_result(messages)["message"])
                or query(TOP_CUSTOMERS, "c4")
            ),
            query(TOTAL, "c5"),
            final(GOOD_REPORT),
        )
        result = self.run_agent("rp-retry", model)
        self.assertEqual(result["status"], "success", result)
        data = json.loads(report_data_file("rp-retry").read_text(encoding="utf-8"))
        self.assertEqual([q["id"] for q in data["queries"]], ["q1", "q2"])      # the failed attempt is not a query

    def test_agent_that_keeps_failing_sql_is_stopped(self):
        bad = "SELECT nope FROM sales_by_customer"
        model = script(
            query(bad, "c3"), query(bad, "c4"), query(TOP_CUSTOMERS, "c5"),
            lambda messages: (
                self.assertIn("failed twice in a row", last_tool_result(messages)["message"])
                or ai("I could not run any query.")
            ),
            ai("Still no JSON."), ai("Nope."),
        )
        result = self.run_agent("rp-stop", model)
        self.assertEqual(result["status"], "error")

    def test_malformed_final_answer_is_sent_back_and_fixed(self):
        model = script(
            query(TOP_CUSTOMERS, "c3"), query(TOTAL, "c4"),
            ai("Here is my report: revenue was high."),
            final(GOOD_REPORT),
        )
        result = self.run_agent("rp-repair", model)
        self.assertEqual(result["status"], "success", result)
        correction = model.seen[-1][-1].content
        self.assertIn("could not be used", correction)
        self.assertIn("JSON object", correction)

    def test_chart_rule_violations_are_sent_back_with_the_specific_problem(self):
        one_bar = "SELECT customer_name, total_amount FROM sales_by_customer WHERE customer_name = 'Ann'"
        bad_report = {**GOOD_REPORT, "charts": [{**GOOD_REPORT["charts"][0], "query_id": "q3"}]}
        model = script(
            query(TOP_CUSTOMERS, "c3"), query(TOTAL, "c4"), query(one_bar, "c5"),
            final(bad_report),
            final(GOOD_REPORT),
        )
        result = self.run_agent("rp-rule", model)
        self.assertEqual(result["status"], "success", result)
        self.assertIn("at least 2 rows", model.seen[-1][-1].content)

    def test_report_citing_an_unexecuted_query_is_rejected_until_fixed(self):
        bad_report = {**GOOD_REPORT, "insights": [{"title": "x", "detail": "y", "query_id": "q9"}]}
        model = script(query(TOP_CUSTOMERS, "c3"), query(TOTAL, "c4"), final(bad_report), final(bad_report), final(bad_report))
        result = self.run_agent("rp-q9", model)
        self.assertEqual(result["status"], "error")
        self.assertIn("'q9'", result["message"])
        self.assertNotIn("report_path", self.scratchpad)
        self.assertFalse(report_file("rp-q9").exists())
        self.assertEqual(self.read_trace("rp-q9")["status"], "failed")

    def test_report_with_no_queries_cannot_be_produced(self):
        model = script(final(GOOD_REPORT), final(GOOD_REPORT), final(GOOD_REPORT))
        result = self.run_agent("rp-noquery", model)
        self.assertEqual(result["status"], "error")
        self.assertIn("not executed", result["message"])

    def test_charts_with_series_use_stable_colours(self):
        model = script(
            query("SELECT month, region, revenue FROM monthly_sales ORDER BY month", "c3"),
            final({
                "headline": "North leads.", "summary": "North grew faster.",
                "charts": [{"title": "Revenue by region", "type": "line", "query_id": "q1", "x": "month",
                            "y": "revenue", "color": "region"}],
            }),
        )
        result = self.run_agent("rp-series", model)
        self.assertEqual(result["status"], "success", result)
        data = json.loads(report_data_file("rp-series").read_text(encoding="utf-8"))
        self.assertEqual(data["color_assignments"], {"North": "#2a78d6", "South": "#eb6834"})

    def test_prompt_injection_in_the_question_stays_inert_in_the_report(self):
        evil = "Which customers? <script>alert('x')</script>"
        model = script(query(TOP_CUSTOMERS, "c3"), query(TOTAL, "c4"), final(GOOD_REPORT))
        result = self.run_agent("rp-evil", model, intent=evil)
        self.assertEqual(result["status"], "success", result)
        html = report_file("rp-evil").read_text(encoding="utf-8")
        self.assertNotIn("<script>alert('x')", html)
        self.assertIn("&lt;script&gt;", html)

    def test_missing_gold_files_return_an_error(self):
        result = json.loads(reporter.run_reporter_agent(
            [str(gold_file("sales_by_customer", "rp-gone"))], "q", "rp-gone", "go", self.scratchpad,
            model=ScriptedChatModel(script=[]),
        ))
        self.assertEqual(result["status"], "error")

    def test_supervisor_tool_and_exact_tool_set(self):
        seen = {}

        def factory(model, tools, system_prompt=None):
            from langchain.agents import create_agent

            seen["tools"] = [t.__name__ for t in tools]
            return create_agent(model, tools, system_prompt=system_prompt)

        model = script(query(TOP_CUSTOMERS, "c3"), query(TOTAL, "c4"), final(GOOD_REPORT))
        tool = reporter.make_reporter_agent_tool(
            gold_files("rp-sup"), "intent", "rp-sup", self.scratchpad, model=model, agent_factory=factory,
        )
        self.assertEqual(tool.__name__, "reporter_agent")
        self.assertIn("report_path", tool.__doc__)
        self.assertEqual(json.loads(safe_tool(tool)("write the report"))["status"], "success")
        self.assertEqual(seen["tools"], ["inspect_gold_tables_tool", "load_gold_data_tool", "execute_query_tool"])


if __name__ == "__main__":
    unittest.main()
