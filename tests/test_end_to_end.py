"""End-to-end: all four phases through the real Orchestrator, Supervisor and specialist agents.

Only the LLMs are scripted. The CSV files, profiling, STTM validation, Bronze/Silver/Gold execution,
DuckDB queries, report rendering, audit log, traces and persistence are all real.
"""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from agents.orchestrator import Orchestrator
from core import run_store
from core.state import AWAITING_APPROVAL, COMPLETE, FAILED
from tests.fakes import ScriptedChatModel, ai, tool_call
from tests.test_bronze_silver_agents import BRONZE_STTM, CUSTOMERS_CSV, ORDERS_CSV, SILVER_STTM

INTENT = "Which customers generate the most revenue?"
GOLD_STTM = [
    {"target_table": "sales_by_customer", "source_table": "customers", "source_col": "name", "target_col": "customer_name",
     "data_type": "string", "join_key": "customer_id", "description": "Customer name"},
    {"target_table": "sales_by_customer", "source_table": "orders", "source_col": "amount", "target_col": "total_amount",
     "data_type": "float", "aggregation": "sum", "description": "Revenue"},
    {"target_table": "sales_by_customer", "source_table": "orders", "source_col": "order_id", "target_col": "order_count",
     "data_type": "integer", "aggregation": "count", "description": "Orders"},
]
REGENERATED_BRONZE = [dict(row, description=f"Regenerated: {row['target_col']}") for row in BRONZE_STTM]
REPORT = {
    "headline": "Bob is the top customer with 200 in revenue.",
    "summary": "Three customers generated 463.63 in revenue across 4 orders.",
    "kpis": [{"label": "Total revenue", "query_id": "q2", "column": "total"}],
    "insights": [{"title": "Top customer", "detail": "Bob leads with 200.", "query_id": "q1"}],
    "charts": [{"title": "Revenue by customer", "type": "bar", "query_id": "q1", "x": "customer_name", "y": "total_amount"}],
    "limitations": ["Only three customers."],
}


def supervisor_run(*tool_names):
    calls = [
        ai(f"Calling {name}.", tool_call(name, f"s{i}", task_description=f"Please run {name} for the business question."))
        for i, name in enumerate(tool_names)
    ]
    return calls + [ai("Phase finished.")]


def sttm_run(layer, rows):
    return [
        ai(f"Plan: inspect the {layer} sources, then map.", tool_call("inspect_context_tool", "t1")),
        ai("Generating.", tool_call(f"generate_{layer}_sttm_tool", "t2", mappings=rows)),
        ai(f"Mapped the {layer} layer."),
    ]


def exec_run(inspect_name, execute_name):
    return [
        ai("Plan: preview then execute.", tool_call(inspect_name, "e1")),
        ai("Executing.", tool_call(execute_name, "e2", plan="apply the approved mapping")),
        ai("Done."),
    ]


def build_models():
    return {
        "supervisor": ScriptedChatModel(script=[
            *supervisor_run("profiler_agent", "sttm_bronze_agent"),          # phase 1
            *supervisor_run("sttm_bronze_agent"),                            # regenerate
            *supervisor_run("bronze_agent", "sttm_silver_agent"),            # phase 2
            *supervisor_run("silver_agent", "sttm_gold_agent"),              # phase 3
            *supervisor_run("gold_agent", "reporter_agent"),                 # phase 4
        ]),
        "profiler": ScriptedChatModel(script=[
            ai("Plan: inspect then profile.", tool_call("inspect_files_tool", "p1")),
            ai("Profiling.", tool_call("profiler_tool", "p2")),
            ai("Profiled both files."),
        ]),
        "sttm": ScriptedChatModel(script=[
            *sttm_run("bronze", BRONZE_STTM), *sttm_run("bronze", REGENERATED_BRONZE),
            *sttm_run("silver", SILVER_STTM), *sttm_run("gold", GOLD_STTM),
        ]),
        "bronze": ScriptedChatModel(script=exec_run("inspect_task_tool", "bronze_ingestion_tool")),
        "silver": ScriptedChatModel(script=exec_run("inspect_task_tool", "silver_ingestion_tool")),
        "gold": ScriptedChatModel(script=exec_run("inspect_task_tool", "gold_ingestion_tool")),
        "reporter": ScriptedChatModel(script=[
            ai("Plan.", tool_call("inspect_gold_tables_tool", "r1")),
            ai("Loading.", tool_call("load_gold_data_tool", "r2")),
            ai("Q1.", tool_call("execute_query_tool", "r3", sql="SELECT customer_name, total_amount FROM sales_by_customer ORDER BY total_amount DESC")),
            ai("Q2.", tool_call("execute_query_tool", "r4", sql="SELECT sum(total_amount) AS total FROM sales_by_customer")),
            ai(json.dumps(REPORT)),
        ]),
    }


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        root = Path(self.dir.name)
        (root / "orders.csv").write_text(ORDERS_CSV, encoding="utf-8")
        (root / "customers.csv").write_text(CUSTOMERS_CSV, encoding="utf-8")
        self.csvs = [str(root / "orders.csv"), str(root / "customers.csv")]
        self.models = build_models()
        self.orchestrator = Orchestrator(models=self.models)

    def test_full_run_with_regenerate_a_human_edit_and_three_approvals(self):
        orch = self.orchestrator

        # Phase 1: profile + Bronze mapping, then pause
        state = orch.start_run(self.csvs, INTENT)
        self.assertEqual((state.status, state.pending_approval_layer), (AWAITING_APPROVAL, "bronze"), state.error)
        self.assertTrue(Path(state.profile_path).is_file())
        self.assertEqual(state.sttm_bronze_path, "")

        # The reviewer is not happy: regenerate with feedback
        orch.regenerate(state, "Add clearer descriptions")
        self.assertEqual(state.status, AWAITING_APPROVAL, state.error)
        self.assertIn("Add clearer descriptions", state.task_description)
        self.assertTrue(orch.pending_rows(state)[0]["description"].startswith("Regenerated:"))

        # Approve Bronze -> Phase 2: Bronze files + Silver mapping
        orch.approve(state)
        orch.run_next_phase(state)
        self.assertEqual((state.current_phase, state.status, state.pending_approval_layer), ("phase_2", AWAITING_APPROVAL, "silver"), state.error)
        bronze = pd.read_parquet(state.bronze_output_paths[0])
        self.assertEqual(len(bronze), 6)                                      # Bronze keeps raw duplicates
        self.assertNotIn("note", bronze.columns)                              # unapproved column never lands

        # The reviewer edits the Silver mapping (UNKNOWN -> MISSING) before approving
        rows = orch.pending_rows(state)
        edited = [r for r in rows if r["target_col"] == "region"][0]
        self.assertIn("UNKNOWN", edited["transformation_logic"])
        edited["transformation_logic"] = edited["transformation_logic"].replace("UNKNOWN", "MISSING")
        orch.approve(state, rows)
        orch.run_next_phase(state)
        self.assertEqual((state.current_phase, state.pending_approval_layer), ("phase_3", "gold"), state.error)
        silver = pd.read_parquet([p for p in state.silver_output_paths if "orders" in Path(p).name][0])
        self.assertEqual(silver.columns[0], "pk_orders_silver_id")
        self.assertIn("MISSING", silver["region"].tolist())                   # the human edit reached execution
        self.assertNotIn("UNKNOWN", silver["region"].tolist())

        # Approve Gold -> Phase 4: Gold + report
        orch.approve(state)
        orch.run_next_phase(state)
        self.assertEqual(state.status, COMPLETE, state.error)
        gold = pd.read_parquet(state.gold_output_paths[0])
        self.assertEqual(gold["customer_name"].tolist(), ["Ann", "Bob", "Cy Lee"])
        self.assertEqual(gold["total_amount"].tolist(), [188.375, 200.0, 75.25])
        self.assertEqual(gold["order_count"].tolist(), [2, 1, 1])
        html = Path(state.report_path).read_text(encoding="utf-8")
        for expected in ("Bob is the top customer", "Revenue by customer", INTENT, "View data table"):
            self.assertIn(expected, html)
        data = json.loads(Path(state.report_json_path).read_text(encoding="utf-8"))
        self.assertEqual(data["kpis"][0]["value"], 463.625)                  # computed by DuckDB, not typed by the model

        # What each step did is saved and reachable from the state (the UI shows it before the next approval)
        from app import ui_logic

        for layer, path, tables in (("bronze", state.bronze_report_path, 2), ("silver", state.silver_report_path, 2), ("gold", state.gold_report_path, 1)):
            report = ui_logic.load_layer_report(path)
            self.assertEqual(report["layer"], layer)
            self.assertEqual(len(report["tables"]), tables)
        silver_orders = next(t for t in ui_logic.load_layer_report(state.silver_report_path)["tables"] if t["table"] == "orders")
        self.assertEqual((silver_orders["source_rows"], silver_orders["output_rows"]), (6, 4))
        self.assertEqual((silver_orders["rows_dropped_dropna"], silver_orders["rows_dropped_dedupe"]), (1, 1))
        self.assertEqual(ui_logic.layer_report_table(ui_logic.load_layer_report(state.gold_report_path)).iloc[0]["Joined on"], "customer_id")

        # Observability
        actions = [e["action"] for e in orch.audit_logs(state)]
        for expected in ("run_created", "phase_1_complete", "regenerate_requested", "sttm_approved", "phase_4_complete", "run_complete"):
            self.assertIn(expected, actions)
        self.assertEqual(actions.count("sttm_approved"), 3)
        approvals = [e for e in orch.audit_logs(state) if e["action"] == "sttm_approved"]
        self.assertEqual([e["edited"] for e in approvals], [False, True, False])
        traces = orch.trace_summaries(state)
        self.assertEqual({t["agent"] for t in traces}, {"supervisor", "profiler", "sttm", "bronze", "silver", "gold", "reporter"})
        self.assertEqual([t["agent"] for t in traces].count("sttm"), 4)
        self.assertTrue(all(t["status"] == "success" for t in traces), traces)
        self.assertIn("inspect_files_tool", next(t for t in traces if t["agent"] == "profiler")["tools"])

        # Persistence: a fresh Orchestrator (e.g. after a Streamlit restart) sees the finished run
        resumed = Orchestrator(models=self.models).resume(state.run_id)
        self.assertEqual((resumed.status, resumed.report_path), (COMPLETE, state.report_path))
        self.assertEqual(resumed.sttm_gold_path, state.sttm_gold_path)
        listed = {r["run_id"]: r for r in run_store.list_runs(limit=1000)}
        self.assertEqual(listed[state.run_id]["status"], COMPLETE)

    def test_only_approved_mappings_ever_reach_execution(self):
        state = self.orchestrator.start_run(self.csvs, INTENT)
        self.assertEqual(state.sttm_bronze_path, "")
        with self.assertRaises(RuntimeError):
            self.orchestrator.run_next_phase(state)                          # cannot skip the gate
        self.assertEqual(self.models["bronze"].seen, [])                     # the Bronze agent never ran

    def test_an_approved_mapping_that_conflicts_with_the_data_can_be_reopened_and_fixed(self):
        orch = self.orchestrator
        strict = [dict(r, nullable="false") if r["target_col"] == "amount" else r for r in BRONZE_STTM]
        self.models["sttm"].script[:3] = sttm_run("bronze", strict)             # phase 1 proposes a mapping that will fail
        self.models["bronze"].script[:] = [
            *exec_run("inspect_task_tool", "bronze_ingestion_tool")[:2],
            ai("Failed: amount has missing values but is nullable=false."),
            *exec_run("inspect_task_tool", "bronze_ingestion_tool"),         # second attempt, after the edit
        ]
        self.models["supervisor"].script[:] = [
            *supervisor_run("profiler_agent", "sttm_bronze_agent"),
            *supervisor_run("bronze_agent"), *supervisor_run("bronze_agent", "sttm_silver_agent"),
        ]
        state = orch.start_run(self.csvs, INTENT)
        orch.approve(state)
        orch.run_next_phase(state)
        self.assertEqual(state.status, FAILED)
        self.assertIn("nullable=false", state.error)

        orch.reopen_mapping(state)
        rows = orch.pending_rows(state)
        for row in rows:
            if row["target_col"] == "amount":
                row["nullable"] = "true"
        orch.approve(state, rows)
        self.models["sttm"].script[3:3] = sttm_run("silver", SILVER_STTM)
        orch.run_next_phase(state)
        self.assertEqual((state.status, state.pending_approval_layer), (AWAITING_APPROVAL, "silver"), state.error)
        self.assertEqual(len(state.errors), 1)


if __name__ == "__main__":
    unittest.main()
