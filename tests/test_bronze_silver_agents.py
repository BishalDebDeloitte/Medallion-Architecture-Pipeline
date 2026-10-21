"""Offline tests for the Bronze and Silver agents (Step 4)."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from agents import bronze_agent, silver_agent
from agents.orchestrator import safe_tool
from core.paths import bronze_file, silver_file, trace_file
from core.sttm_schema import write_sttm
from tests.fakes import ScriptedChatModel, ai, last_tool_result, tool_call

ORDERS_CSV = """order_id,customer_id,amount,status,order_date,discount,region,note
1,10,100.50, Open ,2024-01-05,5,north,keep
2,10,abc,CLOSED,2024-01-06,,south,
2,10,abc,CLOSED,2024-01-06,,south,
3,11,200,open,05/01/2024,10,   ,x
4,,50,Open,not a date,,north,y
5,12,75.25,closed,2024-02-01,2,SOUTH,z
"""
CUSTOMERS_CSV = "customer_id,name\n10,ann\n11,BOB\n12,cy lee\n"


def sttm(layer, table, *columns):
    """Build STTM rows: columns are (source_col, target_col, data_type[, logic[, nullable]])."""
    rows = []
    for column in columns:
        source, target, data_type, *rest = column
        logic = rest[0] if rest else ""
        nullable = rest[1] if len(rest) > 1 else "true"
        rows.append({
            "layer": layer, "target_table": table, "source_table": table, "source_col": source,
            "target_col": target, "data_type": data_type, "transformation_logic": logic,
            "aggregation": "", "nullable": nullable, "join_key": "", "description": f"{target} column",
        })
    return rows


BRONZE_STTM = sttm("bronze", "orders",
    ("order_id", "order_id", "integer"), ("customer_id", "customer_id", "integer"),
    ("amount", "amount", "float"), ("status", "status", "string", "trim"),
    ("order_date", "order_date", "string"), ("discount", "discount", "float"),
    ("region", "region", "string", "trim"),
) + sttm("bronze", "customers", ("customer_id", "customer_id", "integer"), ("name", "name", "string"))

SILVER_STTM = sttm("silver", "orders",
    ("order_id", "order_id", "integer", "dedupe"), ("customer_id", "customer_id", "integer", "dropna"),
    ("amount", "amount", "float", "fillna:median"), ("status", "status", "string", "trim|lowercase"),
    ("order_date", "order_date", "date", "to_date"), ("discount", "discount", "float", "fillna:zero"),
    ("region", "region", "string", "trim|uppercase|fillna:value=UNKNOWN"),
) + sttm("silver", "customers",
    ("customer_id", "customer_id", "integer", "dedupe"), ("name", "name", "string", "trim|titlecase"))


class LayerTestCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.orders = self.root / "orders.csv"
        self.customers = self.root / "customers.csv"
        self.orders.write_text(ORDERS_CSV, encoding="utf-8")
        self.customers.write_text(CUSTOMERS_CSV, encoding="utf-8")
        self.csvs = [str(self.orders), str(self.customers)]
        self.scratchpad = {}

    def write_sttm(self, layer, rows, run_id):
        return str(write_sttm(self.root / f"sttm_{layer}_{run_id}.csv", rows, layer))

    def read_trace(self, agent, run_id):
        return json.loads(trace_file(agent, run_id).read_text(encoding="utf-8"))[-1]

    def make_bronze(self, run_id):
        sttm_path = self.write_sttm("bronze", BRONZE_STTM, run_id)
        outputs, _ = bronze_agent.execute_bronze(self.csvs, sttm_path, run_id)
        return outputs


class BronzeExecutionTests(LayerTestCase):
    def test_writes_one_parquet_per_csv_with_only_approved_columns_and_types(self):
        sttm_path = self.write_sttm("bronze", BRONZE_STTM, "br-exec")
        outputs, reports = bronze_agent.execute_bronze(self.csvs, sttm_path, "br-exec")
        self.assertEqual(outputs, [str(bronze_file("orders", "br-exec")), str(bronze_file("customers", "br-exec"))])
        orders = pd.read_parquet(outputs[0])
        self.assertEqual(list(orders.columns), ["order_id", "customer_id", "amount", "status", "order_date", "discount", "region"])
        self.assertNotIn("note", orders.columns)
        self.assertEqual(
            [str(t) for t in orders.dtypes],
            ["Int64", "Int64", "Float64", "string", "string", "Float64", "string"],
        )
        self.assertEqual(len(orders), 6)                         # Bronze keeps duplicates
        self.assertEqual(orders["status"].tolist()[0], "Open")   # trimmed
        self.assertTrue(pd.isna(orders["region"].iloc[3]))       # blank after trim is missing
        self.assertEqual(orders["order_date"].iloc[3], "05/01/2024")   # dates stay raw text in Bronze
        self.assertEqual(pd.read_parquet(outputs[1])["name"].tolist(), ["ann", "BOB", "cy lee"])

    def test_unconvertible_values_become_null_and_are_reported(self):
        sttm_path = self.write_sttm("bronze", BRONZE_STTM, "br-warn")
        _, reports = bronze_agent.execute_bronze(self.csvs, sttm_path, "br-warn")
        orders = next(r for r in reports if r["table"] == "orders")
        self.assertEqual(orders["coerced_to_null"], {"amount": 2})
        self.assertEqual((orders["source_rows"], orders["output_rows"]), (6, 6))

    def test_non_nullable_violation_fails_and_writes_nothing(self):
        rows = [dict(r, nullable="false") if r["target_col"] == "amount" else r for r in BRONZE_STTM]
        sttm_path = self.write_sttm("bronze", rows, "br-null")
        with self.assertRaisesRegex(ValueError, "orders.amount has 2 missing"):
            bronze_agent.execute_bronze(self.csvs, sttm_path, "br-null")
        self.assertFalse(bronze_file("customers", "br-null").exists())   # no partial layer

    def test_sttm_edited_to_reference_a_missing_column_fails_clearly(self):
        rows = [dict(r, source_col="ghost") if r["target_col"] == "name" else r for r in BRONZE_STTM]
        sttm_path = self.write_sttm("bronze", rows, "br-ghost")
        with self.assertRaisesRegex(ValueError, "ghost"):
            bronze_agent.execute_bronze(self.csvs, sttm_path, "br-ghost")

    def test_every_csv_must_be_mapped(self):
        sttm_path = self.write_sttm("bronze", [r for r in BRONZE_STTM if r["source_table"] == "orders"], "br-miss")
        with self.assertRaisesRegex(ValueError, "customers"):
            bronze_agent.execute_bronze(self.csvs, sttm_path, "br-miss")

    def test_runs_do_not_overwrite_each_other(self):
        sttm_path = self.write_sttm("bronze", BRONZE_STTM, "br-run-a")
        first, _ = bronze_agent.execute_bronze(self.csvs, sttm_path, "br-run-a")
        second, _ = bronze_agent.execute_bronze(self.csvs, sttm_path, "br-run-b")
        self.assertTrue(set(first).isdisjoint(second))
        self.assertTrue(all(Path(p).is_file() for p in first + second))


class SilverExecutionTests(LayerTestCase):
    def run_silver(self, run_id, rows=SILVER_STTM):
        bronze = self.make_bronze(run_id)
        return silver_agent.execute_silver(bronze, self.write_sttm("silver", rows, run_id), run_id)

    def test_cleansing_follows_only_the_sttm_rules(self):
        outputs, reports = self.run_silver("sv-exec")
        orders = pd.read_parquet(outputs[0])
        self.assertEqual(outputs[0], str(silver_file("orders", "sv-exec")))
        self.assertEqual(orders.columns[0], "pk_orders_silver_id")
        self.assertEqual(orders["pk_orders_silver_id"].tolist(), [1, 2, 3, 4])
        self.assertEqual(orders["order_id"].tolist(), [1, 2, 3, 5])       # dropna on customer_id, dedupe on order_id
        self.assertEqual(orders["amount"].tolist(), [100.5, 87.875, 200.0, 75.25])   # median fill
        self.assertEqual(orders["status"].tolist(), ["open", "closed", "open", "closed"])
        self.assertEqual(orders["discount"].tolist(), [5.0, 0.0, 10.0, 2.0])
        self.assertEqual(orders["region"].tolist(), ["NORTH", "SOUTH", "UNKNOWN", "SOUTH"])
        report = next(r for r in reports if r["table"] == "orders")
        self.assertEqual((report["source_rows"], report["output_rows"]), (6, 4))
        self.assertEqual((report["rows_dropped_dropna"], report["rows_dropped_dedupe"]), (1, 1))
        self.assertEqual(report["filled"], {"amount": 2, "discount": 3, "region": 1})
        self.assertEqual(report["coerced_to_null"], {"order_date": 1})      # "not a date" (row later dropped)
        self.assertEqual(str(orders["order_date"].iloc[2])[:10], "2024-05-01")   # 05/01/2024, month-first
        customers = pd.read_parquet(outputs[1])
        self.assertEqual(customers["name"].tolist(), ["Ann", "Bob", "Cy Lee"])
        self.assertEqual(customers.columns[0], "pk_customers_silver_id")

    def test_no_dedupe_rule_means_duplicates_stay(self):
        rows = [dict(r, transformation_logic="") if r["target_col"] == "order_id" else r for r in SILVER_STTM]
        outputs, _ = self.run_silver("sv-nodedupe", rows)
        self.assertEqual(pd.read_parquet(outputs[0])["order_id"].tolist(), [1, 2, 2, 3, 5])

    def test_non_nullable_column_that_stays_null_fails(self):
        rows = [
            dict(r, nullable="false") if r["target_col"] == "order_date"
            else dict(r, transformation_logic="") if r["target_col"] == "customer_id" and r["target_table"] == "orders"
            else r
            for r in SILVER_STTM
        ]   # the unparseable date row is no longer dropped, so order_date stays null
        with self.assertRaisesRegex(ValueError, "order_date"):
            self.run_silver("sv-null", rows)
        self.assertFalse(silver_file("customers", "sv-null").exists())

    def test_pk_clash_is_rejected(self):
        bronze = self.make_bronze("sv-clash")
        frame = pd.read_parquet(bronze[1]).assign(pk_customers_silver_id=1)
        frame.to_parquet(bronze[1], index=False)
        rows = SILVER_STTM + sttm("silver", "customers", ("pk_customers_silver_id", "pk_customers_silver_id", "integer"))
        with self.assertRaisesRegex(ValueError, "primary key"):
            silver_agent.execute_silver(bronze, self.write_sttm("silver", rows, "sv-clash"), "sv-clash")

    def test_input_must_be_a_bronze_artifact_of_this_run(self):
        bronze = self.make_bronze("sv-src")
        with self.assertRaises(ValueError):
            silver_agent.execute_silver(bronze, self.write_sttm("silver", SILVER_STTM, "sv-src"), "other-run")


class ToolTests(LayerTestCase):
    def test_bronze_tools_require_approval_and_inspect_first(self):
        sttm_path = self.write_sttm("bronze", BRONZE_STTM, "tl-br")
        with self.assertRaises(PermissionError):
            bronze_agent.build_bronze_tools(self.csvs, sttm_path, "tl-br", self.scratchpad, approved=False)
        inspect, ingest = bronze_agent.build_bronze_tools(self.csvs, sttm_path, "tl-br", self.scratchpad, approved=True)
        early = json.loads(ingest())
        self.assertEqual(early["status"], "error")
        self.assertIn("inspect_task_tool", early["message"])
        self.assertNotIn("bronze_output_paths", self.scratchpad)
        preview = json.loads(inspect())
        self.assertTrue(preview["sttm_check"]["valid"])
        self.assertEqual([f["table"] for f in preview["files"]], ["orders", "customers"])
        self.assertEqual(len(preview["sttm_rules"]["orders"]), 7)       # 'note' has no rule
        self.assertEqual(preview["files"][0]["columns"][0]["sample_values"], ["1", "2", "3"])
        result = json.loads(ingest(plan="map and type everything"))
        self.assertEqual(result["status"], "success", result)
        self.assertEqual(self.scratchpad["bronze_output_paths"], result["output_paths"])
        self.assertTrue(any("amount: 2 value(s)" in w for w in result["warnings"]))

    def test_inspect_flags_an_invalid_sttm_instead_of_raising(self):
        rows = [dict(r, source_col="ghost") if r["target_col"] == "name" else r for r in BRONZE_STTM]
        inspect, ingest = bronze_agent.build_bronze_tools(
            self.csvs, self.write_sttm("bronze", rows, "tl-bad"), "tl-bad", self.scratchpad, approved=True,
        )
        check = json.loads(inspect())["sttm_check"]
        self.assertFalse(check["valid"])
        self.assertIn("ghost", check["problems"])
        self.assertEqual(json.loads(ingest())["status"], "error")

    def test_silver_tools_preview_bronze_and_rules(self):
        bronze = self.make_bronze("tl-sv")
        sttm_path = self.write_sttm("silver", SILVER_STTM, "tl-sv")
        with self.assertRaises(PermissionError):
            silver_agent.build_silver_tools(bronze, sttm_path, "tl-sv", self.scratchpad, approved=False)
        inspect, ingest = silver_agent.build_silver_tools(bronze, sttm_path, "tl-sv", self.scratchpad, approved=True)
        self.assertEqual(json.loads(ingest())["status"], "error")
        preview = json.loads(inspect())
        orders = next(t for t in preview["bronze_tables"] if t["table"] == "orders")
        amount = next(c for c in orders["columns"] if c["name"] == "amount")
        self.assertEqual((amount["dtype"], amount["null_count"]), ("Float64", 2))
        self.assertTrue(preview["sttm_check"]["valid"])
        self.assertEqual(preview["sttm_rules"]["orders"][2]["transformation_logic"], "fillna:median")
        result = json.loads(ingest(plan="clean"))
        self.assertEqual(result["status"], "success", result)
        self.assertEqual(self.scratchpad["silver_output_paths"], result["output_paths"])


class BronzeAgentLoopTests(LayerTestCase):
    def run_agent(self, run_id, model, sttm_rows=BRONZE_STTM, approved=True):
        sttm_path = self.write_sttm("bronze", sttm_rows, run_id)
        return json.loads(bronze_agent.run_bronze_agent(
            self.csvs, sttm_path, run_id, "ingest the raw files", self.scratchpad, approved, model=model,
        ))

    def test_inspect_then_ingest_succeeds_and_is_traced(self):
        model = ScriptedChatModel(script=[
            ai("Plan: preview the files and rules, then ingest.", tool_call("inspect_task_tool", "call_1")),
            lambda messages: (
                self.assertTrue(last_tool_result(messages)["sttm_check"]["valid"])
                or ai("Rules are valid; ingesting.", tool_call("bronze_ingestion_tool", "call_2", plan="ingest both files"))
            ),
            ai("Ingested orders (6 rows) and customers (3 rows)."),
        ])
        result = self.run_agent("ag-br", model)
        self.assertEqual(result["status"], "success", result)
        self.assertEqual(result["file_count"], 2)
        self.assertEqual(self.scratchpad["bronze_output_paths"], result["bronze_output_paths"])
        trace = self.read_trace("bronze", "ag-br")
        self.assertEqual([c["name"] for c in trace["tool_calls"]], ["inspect_task_tool", "bronze_ingestion_tool"])
        self.assertIn("Plan", trace["plan"])
        self.assertEqual(trace["status"], "success")

    def test_agent_stops_when_the_sttm_is_invalid_and_explains(self):
        rows = [dict(r, source_col="ghost") if r["target_col"] == "name" else r for r in BRONZE_STTM]
        model = ScriptedChatModel(script=[
            ai("Plan.", tool_call("inspect_task_tool", "call_1")),
            ai("I did not ingest: the STTM references column ghost, which does not exist."),
        ])
        result = self.run_agent("ag-br-bad", model, rows)
        self.assertEqual(result["status"], "error")
        self.assertIn("ghost", result["message"])
        self.assertNotIn("bronze_output_paths", self.scratchpad)
        self.assertEqual(self.read_trace("bronze", "ag-br-bad")["status"], "failed")

    def test_execution_error_is_reported_not_retried(self):
        rows = [dict(r, nullable="false") if r["target_col"] == "amount" else r for r in BRONZE_STTM]
        model = ScriptedChatModel(script=[
            ai("Plan.", tool_call("inspect_task_tool", "call_1")),
            ai("Ingesting.", tool_call("bronze_ingestion_tool", "call_2", plan="p")),
            lambda messages: ai(f"Failed: {last_tool_result(messages)['message']}"),
        ])
        result = self.run_agent("ag-br-err", model, rows)
        self.assertEqual(result["status"], "error")
        self.assertIn("orders.amount", result["message"])
        self.assertEqual(len(self.read_trace("bronze", "ag-br-err")["tool_calls"]), 2)
        self.assertFalse(bronze_file("orders", "ag-br-err").exists())

    def test_unapproved_sttm_never_runs(self):
        model = ScriptedChatModel(script=[ai("Plan.", tool_call("inspect_task_tool", "call_1"))])
        result = self.run_agent("ag-br-unapproved", model, approved=False)
        self.assertEqual(result["status"], "error")
        self.assertIn("human approval", result["message"])
        self.assertFalse(bronze_file("orders", "ag-br-unapproved").exists())

    def test_supervisor_tool_and_exact_tool_set(self):
        seen = {}

        def factory(model, tools, system_prompt=None):
            from langchain.agents import create_agent

            seen["tools"] = [t.__name__ for t in tools]
            return create_agent(model, tools, system_prompt=system_prompt)

        model = ScriptedChatModel(script=[
            ai("Plan.", tool_call("inspect_task_tool", "call_1")),
            ai("Go.", tool_call("bronze_ingestion_tool", "call_2", plan="p")),
            ai("Done."),
        ])
        tool = bronze_agent.make_bronze_agent_tool(
            self.csvs, self.write_sttm("bronze", BRONZE_STTM, "ag-br-sup"), "ag-br-sup",
            self.scratchpad, True, model=model, agent_factory=factory,
        )
        self.assertEqual(tool.__name__, "bronze_agent")
        self.assertIn("bronze_output_paths", tool.__doc__)
        self.assertEqual(json.loads(safe_tool(tool)("ingest"))["status"], "success")
        self.assertEqual(seen["tools"], ["inspect_task_tool", "bronze_ingestion_tool"])


class SilverAgentLoopTests(LayerTestCase):
    def run_agent(self, run_id, model, sttm_rows=SILVER_STTM, approved=True):
        bronze = self.make_bronze(run_id)
        sttm_path = self.write_sttm("silver", sttm_rows, run_id)
        return json.loads(silver_agent.run_silver_agent(
            bronze, sttm_path, run_id, "cleanse the bronze tables", self.scratchpad, approved, model=model,
        ))

    def test_inspect_then_cleanse_succeeds_and_verifies_outputs(self):
        model = ScriptedChatModel(script=[
            ai("Plan: apply the approved cleansing rules.", tool_call("inspect_task_tool", "call_1")),
            lambda messages: (
                self.assertTrue(last_tool_result(messages)["sttm_check"]["valid"])
                or ai("Valid.", tool_call("silver_ingestion_tool", "call_2", plan="clean both tables"))
            ),
            ai("Orders 6 -> 4 rows, customers 3 -> 3."),
        ])
        result = self.run_agent("ag-sv", model)
        self.assertEqual(result["status"], "success", result)
        outputs = self.scratchpad["silver_output_paths"]
        self.assertEqual(pd.read_parquet(outputs[0]).columns[0], "pk_orders_silver_id")
        trace = self.read_trace("silver", "ag-sv")
        self.assertEqual([c["name"] for c in trace["tool_calls"]], ["inspect_task_tool", "silver_ingestion_tool"])
        self.assertEqual(trace["output"]["file_count"], 2)

    def test_agent_adapts_after_running_before_inspecting(self):
        model = ScriptedChatModel(script=[
            ai("Cleansing right away.", tool_call("silver_ingestion_tool", "call_1")),
            lambda messages: (
                self.assertEqual(last_tool_result(messages)["status"], "error")
                or ai("Inspect first.", tool_call("inspect_task_tool", "call_2"))
            ),
            ai("Now cleansing.", tool_call("silver_ingestion_tool", "call_3", plan="p")),
            ai("Done."),
        ])
        result = self.run_agent("ag-sv-adapt", model)
        self.assertEqual(result["status"], "success", result)

    def test_tampered_output_is_caught_by_verification(self):
        model = ScriptedChatModel(script=[
            ai("Plan.", tool_call("inspect_task_tool", "call_1")),
            ai("Go.", tool_call("silver_ingestion_tool", "call_2", plan="p")),
            ai("Done."),
        ])
        original = silver_agent.execute_silver

        def tampering(*args, **kwargs):
            outputs, reports = original(*args, **kwargs)
            pd.read_parquet(outputs[1]).assign(extra=1).to_parquet(outputs[1], index=False)
            return outputs, reports

        silver_agent.execute_silver = tampering
        self.addCleanup(setattr, silver_agent, "execute_silver", original)
        result = self.run_agent("ag-sv-tamper", model)
        self.assertEqual(result["status"], "error")
        self.assertIn("expected", result["message"])

    def test_unapproved_sttm_never_runs(self):
        result = self.run_agent("ag-sv-unapproved", ScriptedChatModel(script=[ai("Nothing.")]), approved=False)
        self.assertEqual(result["status"], "error")
        self.assertFalse(silver_file("orders", "ag-sv-unapproved").exists())

    def test_supervisor_tool(self):
        bronze = self.make_bronze("ag-sv-sup")
        model = ScriptedChatModel(script=[
            ai("Plan.", tool_call("inspect_task_tool", "call_1")),
            ai("Go.", tool_call("silver_ingestion_tool", "call_2", plan="p")),
            ai("Done."),
        ])
        tool = silver_agent.make_silver_agent_tool(
            bronze, self.write_sttm("silver", SILVER_STTM, "ag-sv-sup"), "ag-sv-sup", self.scratchpad, True, model=model,
        )
        self.assertEqual(tool.__name__, "silver_agent")
        self.assertEqual(json.loads(safe_tool(tool)("cleanse"))["status"], "success")


if __name__ == "__main__":
    unittest.main()
