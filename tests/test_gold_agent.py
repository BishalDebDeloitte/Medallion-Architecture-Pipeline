"""Offline tests for the Gold agent (Step 5)."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from agents import gold_agent
from agents.orchestrator import safe_tool
from core.paths import gold_file, silver_file, trace_file
from core.sttm_schema import validate_rows, write_sttm
from tests.fakes import ScriptedChatModel, ai, last_tool_result, tool_call


def g(target, source, column, target_col, data_type, logic="", aggregation="", join_key="", nullable="true"):
    """One Gold STTM row."""
    return {
        "layer": "gold", "target_table": target, "source_table": source, "source_col": column,
        "target_col": target_col, "data_type": data_type, "transformation_logic": logic,
        "aggregation": aggregation, "nullable": nullable, "join_key": join_key, "description": f"{target_col}",
    }


ORDERS = pd.DataFrame({
    "order_id": pd.array([1, 2, 3, 4, 5, 6], dtype="Int64"),
    "customer_id": pd.array([1, 1, 2, 2, 3, 99], dtype="Int64"),
    "amount": pd.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype="Float64"),
    "status": pd.array(["open", "open", "closed", "open", None, "closed"], dtype="string"),
})
CUSTOMERS = pd.DataFrame({
    "customer_id": pd.array([1, 2, 3, 4], dtype="Int64"),
    "name": pd.array(["Ann", "Bob", "Cy", "Di"], dtype="string"),
    "region": pd.array(["N", "S", "N", "S"], dtype="string"),
})

SALES_BY_CUSTOMER = [
    g("sales_by_customer", "customers", "name", "customer_name", "string", join_key="customer_id"),
    g("sales_by_customer", "orders", "amount", "total_amount", "float", aggregation="sum"),
    g("sales_by_customer", "orders", "order_id", "order_count", "integer", aggregation="count"),
]
ORDERS_BY_STATUS = [
    g("orders_by_status", "orders", "status", "status_label", "string", logic="uppercase"),
    g("orders_by_status", "orders", "amount", "avg_amount", "float", aggregation="avg"),
    g("orders_by_status", "orders", "amount", "max_amount", "float", aggregation="max"),
    g("orders_by_status", "orders", "amount", "min_amount", "float", aggregation="min"),
    g("orders_by_status", "orders", "customer_id", "customers_served", "integer", aggregation="nunique"),
    g("orders_by_status", "orders", "order_id", "order_count", "integer", aggregation="count"),
]


class GoldTestCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.scratchpad = {}

    def silver(self, run_id, tables=None):
        """Write Silver Parquet files (with pk columns) and return their paths."""
        paths = []
        for table, frame in (tables or {"orders": ORDERS, "customers": CUSTOMERS}).items():
            frame = frame.copy()
            frame.insert(0, f"pk_{table}_silver_id", pd.array(range(1, len(frame) + 1), dtype="Int64"))
            path = silver_file(table, run_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            paths.append(str(path))
        return paths

    def sttm(self, rows, run_id):
        return str(write_sttm(self.root / f"sttm_gold_{run_id}.csv", rows, "gold"))

    def execute(self, rows, run_id, tables=None):
        paths = self.silver(run_id, tables)
        return gold_agent.execute_gold(paths, self.sttm(rows, run_id), run_id)

    def table(self, outputs, name):
        return pd.read_parquet(next(p for p in outputs if Path(p).name.startswith(f"{name}_")))

    def read_trace(self, run_id):
        return json.loads(trace_file("gold", run_id).read_text(encoding="utf-8"))[-1]


class MaterialisationTests(GoldTestCase):
    def test_outer_join_group_by_and_aggregation(self):
        outputs, reports = self.execute(SALES_BY_CUSTOMER, "gd-join")
        table = self.table(outputs, "sales_by_customer")
        self.assertEqual(list(table.columns), ["pk_sales_by_customer_gold_id", "customer_name", "total_amount", "order_count"])
        self.assertEqual(table["pk_sales_by_customer_gold_id"].tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(table["customer_name"].tolist()[:4], ["Ann", "Bob", "Cy", "Di"])
        self.assertTrue(pd.isna(table["customer_name"].iloc[4]))        # order 99 has no customer: kept by the outer join
        self.assertEqual(table["total_amount"].iloc[:3].tolist(), [30.0, 70.0, 50.0])
        self.assertTrue(pd.isna(table["total_amount"].iloc[3]))         # Di has no orders: null, not 0
        self.assertEqual(table["total_amount"].iloc[4], 60.0)
        self.assertEqual(table["order_count"].tolist(), [2, 2, 1, 0, 1])
        self.assertEqual([str(t) for t in table.dtypes], ["Int64", "string", "Float64", "Int64"])
        report = reports[0]
        self.assertEqual((report["merged_rows"], report["output_rows"]), (7, 5))
        self.assertEqual(report["source_tables"], ["customers", "orders"])
        self.assertEqual(report["group_by"], ["customer_name"])

    def test_every_aggregation_on_a_single_table(self):
        outputs, _ = self.execute(ORDERS_BY_STATUS, "gd-aggs")
        table = self.table(outputs, "orders_by_status")
        self.assertEqual(table["status_label"].tolist()[:2], ["CLOSED", "OPEN"])
        self.assertTrue(pd.isna(table["status_label"].iloc[2]))          # null status keeps its own group
        self.assertEqual(table["avg_amount"].iloc[:2].round(4).tolist(), [45.0, 23.3333])
        self.assertEqual(table["max_amount"].tolist(), [60.0, 40.0, 50.0])
        self.assertEqual(table["min_amount"].tolist(), [30.0, 10.0, 50.0])
        self.assertEqual(table["customers_served"].tolist(), [2, 2, 1])
        self.assertEqual(table["order_count"].tolist(), [2, 3, 1])

    def test_measures_only_table_is_a_single_row(self):
        rows = [
            g("totals", "orders", "amount", "total_amount", "float", aggregation="sum"),
            g("totals", "orders", "order_id", "order_count", "integer", aggregation="count"),
            g("totals", "orders", "amount", "biggest", "float", aggregation="max"),
        ]
        outputs, _ = self.execute(rows, "gd-totals")
        table = self.table(outputs, "totals")
        self.assertEqual(len(table), 1)
        self.assertEqual(table.iloc[0][["total_amount", "order_count", "biggest"]].tolist(), [210.0, 6, 60.0])

    def test_dimension_only_table_is_distinct_rows(self):
        rows = [
            g("regions", "customers", "region", "region", "string"),
        ]
        outputs, _ = self.execute(rows, "gd-dims")
        self.assertEqual(self.table(outputs, "regions")["region"].tolist(), ["N", "S"])

    def test_several_target_tables_one_file_each_with_run_scoped_names(self):
        outputs, _ = self.execute(SALES_BY_CUSTOMER + ORDERS_BY_STATUS, "gd-multi")
        self.assertEqual(outputs, [str(gold_file("sales_by_customer", "gd-multi")), str(gold_file("orders_by_status", "gd-multi"))])
        self.assertTrue(all(Path(p).is_file() for p in outputs))

    def test_rename_cast_and_text_operations(self):
        rows = [
            g("by_customer", "orders", "customer_id", "customer_ref", "string"),
            g("by_customer", "orders", "status", "state", "string", logic="trim|uppercase"),
            g("by_customer", "orders", "amount", "spend", "float", aggregation="sum"),
        ]
        outputs, _ = self.execute(rows, "gd-rename")
        table = self.table(outputs, "by_customer")
        self.assertEqual(str(table["customer_ref"].dtype), "string")
        self.assertIn("1", table["customer_ref"].tolist())
        self.assertNotIn("status", table.columns)
        self.assertEqual(set(table["state"].dropna()), {"OPEN", "CLOSED"})

    def test_only_approved_columns_survive(self):
        outputs, _ = self.execute(SALES_BY_CUSTOMER, "gd-cols")
        columns = set(self.table(outputs, "sales_by_customer").columns)
        self.assertFalse(columns & {"region", "customer_id", "status", "pk_orders_silver_id"})

    def test_runs_do_not_overwrite_each_other(self):
        first, _ = self.execute(SALES_BY_CUSTOMER, "gd-run-a")
        second, _ = self.execute(SALES_BY_CUSTOMER, "gd-run-b")
        self.assertTrue(set(first).isdisjoint(second))


class JoinSafetyTests(GoldTestCase):
    def test_rows_with_null_keys_never_match_each_other(self):
        a = pd.DataFrame({"k": pd.array([1, None], dtype="Int64"), "v": pd.array([10.0, 20.0], dtype="Float64")})
        b = pd.DataFrame({"k": pd.array([1, None], dtype="Int64"), "w": pd.array([5.0, 6.0], dtype="Float64")})
        rows = [
            g("t", "a", "k", "key", "integer", join_key="k"),
            g("t", "a", "v", "total_v", "float", aggregation="sum"),
            g("t", "b", "w", "total_w", "float", aggregation="sum"),
        ]
        outputs, reports = self.execute(rows, "gd-nullkey", {"a": a, "b": b})
        self.assertEqual(reports[0]["merged_rows"], 3)       # 1 matched + 2 unmatched null-key rows, not 2
        table = self.table(outputs, "t")
        self.assertEqual(table["total_v"].tolist(), [10.0, 20.0])
        self.assertEqual(table["total_w"].tolist(), [5.0, 6.0])

    def test_join_that_would_multiply_rows_is_refused(self):
        returns = pd.DataFrame({"customer_id": pd.array([1, 1, 2], dtype="Int64"), "refund": pd.array([1.0, 2.0, 3.0], dtype="Float64")})
        rows = [
            g("risky", "orders", "customer_id", "customer_id", "integer", join_key="customer_id"),
            g("risky", "orders", "amount", "total", "float", aggregation="sum"),
            g("risky", "returns", "refund", "refunds", "float", aggregation="sum"),
        ]
        with self.assertRaisesRegex(ValueError, "would multiply rows"):
            self.execute(rows, "gd-fanout", {"orders": ORDERS, "returns": returns})

    def test_incompatible_key_types_are_refused(self):
        a = pd.DataFrame({"k": pd.array([1, 2], dtype="Int64"), "v": pd.array([1.0, 2.0], dtype="Float64")})
        b = pd.DataFrame({"k": pd.array(["1", "2"], dtype="string"), "w": pd.array([1.0, 2.0], dtype="Float64")})
        rows = [
            g("t", "a", "v", "total_v", "float", aggregation="sum", join_key="k"),
            g("t", "b", "w", "total_w", "float", aggregation="sum"),
        ]
        with self.assertRaisesRegex(ValueError, "incompatible types"):
            self.execute(rows, "gd-types", {"a": a, "b": b})

    def test_composite_join_key(self):
        a = pd.DataFrame({"k1": pd.array([1, 1], dtype="Int64"), "k2": pd.array(["x", "y"], dtype="string"),
                          "v": pd.array([10.0, 20.0], dtype="Float64")})
        b = pd.DataFrame({"k1": pd.array([1, 2], dtype="Int64"), "k2": pd.array(["x", "x"], dtype="string"),
                          "w": pd.array([5.0, 7.0], dtype="Float64")})
        rows = [
            g("t", "a", "k1", "k1", "integer", join_key="k1,k2"),
            g("t", "a", "k2", "k2", "string"),
            g("t", "a", "v", "total_v", "float", aggregation="sum"),
            g("t", "b", "w", "total_w", "float", aggregation="sum"),
        ]
        outputs, reports = self.execute(rows, "gd-composite", {"a": a, "b": b})
        self.assertEqual(reports[0]["merged_rows"], 3)       # (1,x) matched, (1,y) and (2,x) unmatched
        table = self.table(outputs, "t").sort_values(["k1", "k2"]).reset_index(drop=True)
        self.assertEqual(table[["total_v", "total_w"]].iloc[0].tolist(), [10.0, 5.0])


class RuleEnforcementTests(GoldTestCase):
    def test_sum_over_a_text_column_is_refused(self):
        rows = [g("bad", "orders", "status", "status_total", "float", aggregation="sum")]
        with self.assertRaisesRegex(ValueError, "needs a numeric column"):
            self.execute(rows, "gd-sumtext")

    def test_non_nullable_violation_fails_and_writes_nothing(self):
        rows = [dict(r, nullable="false") if r["target_col"] == "customer_name" else r for r in SALES_BY_CUSTOMER]
        with self.assertRaisesRegex(ValueError, "sales_by_customer.customer_name"):
            self.execute(rows + ORDERS_BY_STATUS, "gd-null")
        self.assertFalse(gold_file("orders_by_status", "gd-null").exists())      # no partial layer

    def test_sttm_edited_to_reference_a_missing_column_fails_clearly(self):
        rows = [dict(r, source_col="ghost") if r["target_col"] == "total_amount" else r for r in SALES_BY_CUSTOMER]
        with self.assertRaisesRegex(ValueError, "ghost"):
            self.execute(rows, "gd-ghost")

    def test_target_table_must_be_a_sql_friendly_name(self):
        rows = [g("bad name", "orders", "amount", "total", "float", aggregation="sum")]
        with self.assertRaisesRegex(ValueError, "SQL table name"):
            validate_rows(rows, "gold")
        validate_rows([g("ok_name2", "orders", "amount", "total", "float", aggregation="sum")], "gold")

    def test_silver_inputs_must_be_silver_artifacts_of_this_run(self):
        paths = self.silver("gd-src")
        with self.assertRaises(ValueError):
            gold_agent.execute_gold(paths, self.sttm(SALES_BY_CUSTOMER, "gd-src"), "other-run")


class ToolTests(GoldTestCase):
    def test_tools_require_approval_and_inspect_first(self):
        paths, sttm_path = self.silver("tl-gd"), self.sttm(SALES_BY_CUSTOMER + ORDERS_BY_STATUS, "tl-gd")
        with self.assertRaises(PermissionError):
            gold_agent.build_gold_tools(paths, sttm_path, "tl-gd", self.scratchpad, approved=False)
        inspect, ingest = gold_agent.build_gold_tools(paths, sttm_path, "tl-gd", self.scratchpad, approved=True)
        early = json.loads(ingest())
        self.assertEqual(early["status"], "error")
        self.assertIn("inspect_task_tool", early["message"])
        preview = json.loads(inspect())
        self.assertTrue(preview["sttm_check"]["valid"])
        self.assertEqual({t["table"] for t in preview["silver_tables"]}, {"orders", "customers"})
        plan = preview["gold_plan"]
        self.assertEqual(plan["sales_by_customer"]["group_by"], ["customer_name"])
        self.assertEqual(plan["sales_by_customer"]["measures"]["total_amount"], "sum(orders.amount)")
        self.assertEqual(plan["sales_by_customer"]["join_check"], {"ok": True, "merged_rows": 7})
        self.assertEqual(len(preview["sttm_rules"]["orders_by_status"]), 6)
        result = json.loads(ingest(plan="build both tables"))
        self.assertEqual(result["status"], "success", result)
        self.assertEqual(self.scratchpad["gold_output_paths"], result["output_paths"])

    def test_inspect_flags_an_unsafe_join_and_an_invalid_sttm(self):
        returns = pd.DataFrame({"customer_id": pd.array([1, 1], dtype="Int64"), "refund": pd.array([1.0, 2.0], dtype="Float64")})
        risky = [
            g("risky", "orders", "customer_id", "customer_id", "integer", join_key="customer_id"),
            g("risky", "returns", "refund", "refunds", "float", aggregation="sum"),
        ]
        paths = self.silver("tl-risky", {"orders": ORDERS, "returns": returns})
        inspect, ingest = gold_agent.build_gold_tools(paths, self.sttm(risky, "tl-risky"), "tl-risky", self.scratchpad, True)
        plan = json.loads(inspect())["gold_plan"]["risky"]
        self.assertFalse(plan["join_check"]["ok"])
        self.assertIn("multiply", plan["join_check"]["problem"])
        self.assertEqual(json.loads(ingest())["status"], "error")

        bad = [dict(r, source_col="ghost") if r["target_col"] == "order_count" else r for r in SALES_BY_CUSTOMER]
        inspect2, _ = gold_agent.build_gold_tools(
            self.silver("tl-bad"), self.sttm(bad, "tl-bad"), "tl-bad", self.scratchpad, True,
        )
        check = json.loads(inspect2())["sttm_check"]
        self.assertFalse(check["valid"])
        self.assertIn("ghost", check["problems"])


class AgentLoopTests(GoldTestCase):
    def run_agent(self, run_id, model, rows=SALES_BY_CUSTOMER + ORDERS_BY_STATUS, approved=True, tables=None):
        paths = self.silver(run_id, tables)
        return json.loads(gold_agent.run_gold_agent(
            paths, self.sttm(rows, run_id), "Which customers buy the most?", run_id,
            "materialise the gold tables", self.scratchpad, approved, model=model,
        ))

    def test_inspect_then_materialise_succeeds_and_is_traced(self):
        model = ScriptedChatModel(script=[
            ai("Plan: check the joins, then build both tables.", tool_call("inspect_task_tool", "call_1")),
            lambda messages: (
                self.assertTrue(last_tool_result(messages)["gold_plan"]["sales_by_customer"]["join_check"]["ok"])
                or ai("Joins are safe.", tool_call("gold_ingestion_tool", "call_2", plan="build both tables"))
            ),
            ai("Built sales_by_customer (5 rows) and orders_by_status (3 rows); they answer the question."),
        ])
        result = self.run_agent("ag-gd", model)
        self.assertEqual(result["status"], "success", result)
        self.assertEqual(result["table_count"], 2)
        self.assertEqual(self.scratchpad["gold_output_paths"], result["gold_output_paths"])
        trace = self.read_trace("ag-gd")
        self.assertEqual([c["name"] for c in trace["tool_calls"]], ["inspect_task_tool", "gold_ingestion_tool"])
        self.assertEqual(trace["status"], "success")
        self.assertIn("Plan", trace["plan"])
        self.assertIn("Which customers buy the most?", json.dumps(model.seen[0][-1].content))   # intent reaches the agent

    def test_agent_stops_when_the_join_is_unsafe(self):
        returns = pd.DataFrame({"customer_id": pd.array([1, 1], dtype="Int64"), "refund": pd.array([1.0, 2.0], dtype="Float64")})
        risky = [
            g("risky", "orders", "customer_id", "customer_id", "integer", join_key="customer_id"),
            g("risky", "returns", "refund", "refunds", "float", aggregation="sum"),
        ]
        model = ScriptedChatModel(script=[
            ai("Plan.", tool_call("inspect_task_tool", "call_1")),
            ai("I did not run it: the join would multiply rows."),
        ])
        result = self.run_agent("ag-gd-risky", model, risky, tables={"orders": ORDERS, "returns": returns})
        self.assertEqual(result["status"], "error")
        self.assertIn("multiply", result["message"])
        self.assertNotIn("gold_output_paths", self.scratchpad)

    def test_execution_error_is_reported_not_retried(self):
        rows = [dict(r, nullable="false") if r["target_col"] == "customer_name" else r for r in SALES_BY_CUSTOMER]
        model = ScriptedChatModel(script=[
            ai("Plan.", tool_call("inspect_task_tool", "call_1")),
            ai("Go.", tool_call("gold_ingestion_tool", "call_2", plan="p")),
            lambda messages: ai(f"Failed: {last_tool_result(messages)['message']}"),
        ])
        result = self.run_agent("ag-gd-err", model, rows)
        self.assertEqual(result["status"], "error")
        self.assertIn("customer_name", result["message"])
        self.assertEqual(len(self.read_trace("ag-gd-err")["tool_calls"]), 2)
        self.assertFalse(gold_file("sales_by_customer", "ag-gd-err").exists())

    def test_unapproved_sttm_never_runs(self):
        result = self.run_agent("ag-gd-unapproved", ScriptedChatModel(script=[ai("Nothing.")]), approved=False)
        self.assertEqual(result["status"], "error")
        self.assertIn("human approval", result["message"])
        self.assertFalse(gold_file("sales_by_customer", "ag-gd-unapproved").exists())

    def test_tampered_output_is_caught_by_verification(self):
        model = ScriptedChatModel(script=[
            ai("Plan.", tool_call("inspect_task_tool", "call_1")),
            ai("Go.", tool_call("gold_ingestion_tool", "call_2", plan="p")),
            ai("Done."),
        ])
        original = gold_agent.execute_gold

        def tampering(*args, **kwargs):
            outputs, reports = original(*args, **kwargs)
            pd.read_parquet(outputs[0]).assign(extra=1).to_parquet(outputs[0], index=False)
            return outputs, reports

        gold_agent.execute_gold = tampering
        self.addCleanup(setattr, gold_agent, "execute_gold", original)
        result = self.run_agent("ag-gd-tamper", model)
        self.assertEqual(result["status"], "error")
        self.assertIn("expected", result["message"])

    def test_supervisor_tool_and_exact_tool_set(self):
        seen = {}

        def factory(model, tools, system_prompt=None):
            from langchain.agents import create_agent

            seen["tools"] = [t.__name__ for t in tools]
            return create_agent(model, tools, system_prompt=system_prompt)

        model = ScriptedChatModel(script=[
            ai("Plan.", tool_call("inspect_task_tool", "call_1")),
            ai("Go.", tool_call("gold_ingestion_tool", "call_2", plan="p")),
            ai("Done."),
        ])
        tool = gold_agent.make_gold_agent_tool(
            self.silver("ag-gd-sup"), self.sttm(SALES_BY_CUSTOMER, "ag-gd-sup"), "intent", "ag-gd-sup",
            self.scratchpad, True, model=model, agent_factory=factory,
        )
        self.assertEqual(tool.__name__, "gold_agent")
        self.assertIn("gold_output_paths", tool.__doc__)
        self.assertEqual(json.loads(safe_tool(tool)("materialise"))["status"], "success")
        self.assertEqual(seen["tools"], ["inspect_task_tool", "gold_ingestion_tool"])


if __name__ == "__main__":
    unittest.main()
