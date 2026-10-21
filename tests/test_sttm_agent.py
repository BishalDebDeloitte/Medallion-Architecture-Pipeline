"""Offline tests for the STTM agent (Step 3)."""

import json
import tempfile
import unittest
from pathlib import Path

from agents import profiler, sttm_generator
from agents.orchestrator import safe_tool
from agents.sttm_generator import SttmInputs, SttmRow
from core.paths import sttm_file, trace_file
from core.sttm_schema import read_sttm
from tests.fakes import ScriptedChatModel, ai, last_tool_result, tool_call
from tests.fixtures import make_customers, make_orders, write_bronze, write_csv, write_silver

BRONZE_ROWS = [
    {"source_table": "orders", "source_col": "order_id", "target_col": "order_id", "data_type": "integer"},
    {"source_table": "orders", "source_col": "customer_id", "target_col": "customer_id", "data_type": "integer"},
    {"source_table": "orders", "source_col": "amount", "target_col": "amount", "data_type": "float"},
    {"source_table": "customers", "source_col": "customer_id", "target_col": "customer_id", "data_type": "integer"},
    {"source_table": "customers", "source_col": "name", "target_col": "customer_name", "data_type": "string",
     "transformation_logic": "trim"},
]
GOLD_ROWS = [
    {"target_table": "sales_by_customer", "source_table": "customers", "source_col": "name",
     "target_col": "customer_name", "data_type": "string", "join_key": "customer_id"},
    {"target_table": "sales_by_customer", "source_table": "orders", "source_col": "amount",
     "target_col": "total_amount", "data_type": "float", "aggregation": "sum"},
    {"target_table": "sales_by_customer", "source_table": "orders", "source_col": "order_id",
     "target_col": "order_count", "data_type": "integer", "aggregation": "count"},
]


def make_profile(run_id: str, root: Path) -> str:
    paths = [write_csv(root, "orders.csv", make_orders()), write_csv(root, "customers.csv", make_customers())]
    _, path = profiler.build_profile(
        paths, run_id, "Which customers buy most?",
        semantic_meanings={"orders": {"amount": "Order value in USD"}},
    )
    return str(path)


class InputsAndContextTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)

    def test_inputs_require_the_right_source_for_each_layer(self):
        with self.assertRaises(ValueError):
            SttmInputs("bronze", "r", "q")
        with self.assertRaises(ValueError):
            SttmInputs("silver", "r", "q", profile_path="p.json")
        with self.assertRaises(ValueError):
            SttmInputs("platinum", "r", "q", profile_path="p.json")
        SttmInputs("gold", "r", "q", parquet_paths=("a.parquet",))

    def test_bronze_context_comes_from_the_profile_with_meanings(self):
        profile_path = make_profile("ctx-bronze", self.root)
        context = sttm_generator.load_context(SttmInputs("bronze", "ctx-bronze", "q", profile_path=profile_path))
        self.assertEqual(set(context["tables"]), {"orders", "customers"})
        amount = next(c for c in context["tables"]["orders"]["columns"] if c["name"] == "amount")
        self.assertEqual(amount["meaning"], "Order value in USD")
        self.assertLessEqual(len(amount["sample_values"]), 3)
        self.assertEqual(len(context["join_keys"]), 1)
        self.assertIn("customer_id", sttm_generator.available_columns(context)["customers"])

    def test_profile_from_another_run_is_rejected(self):
        profile_path = make_profile("ctx-other", self.root)
        with self.assertRaises(ValueError):
            sttm_generator.load_context(SttmInputs("bronze", "ctx-different", "q", profile_path=profile_path))

    def test_parquet_context_uses_table_names_from_artifact_paths(self):
        paths = write_bronze("ctx-silver")
        context = sttm_generator.load_context(SttmInputs("silver", "ctx-silver", "q", parquet_paths=tuple(paths)))
        self.assertEqual(set(context["tables"]), {"orders", "customers"})
        self.assertEqual(context["tables"]["orders"]["row_count"], 31)
        self.assertEqual(context["tables"]["orders"]["columns"][0]["name"], "order_id")

    def test_wrong_layer_artifact_is_rejected(self):
        silver_paths = write_silver("ctx-wrong")
        with self.assertRaises(ValueError):
            sttm_generator.load_context(SttmInputs("silver", "ctx-wrong", "q", parquet_paths=tuple(silver_paths)))

    def test_previous_mappings_are_summarised(self):
        paths = write_bronze("ctx-prev")
        previous = sttm_file("bronze", "ctx-prev")
        from core.sttm_schema import write_sttm

        write_sttm(previous, [
            {**{k: "" for k in ("transformation_logic", "aggregation", "join_key")}, "layer": "bronze",
             "target_table": "orders", "source_table": "orders", "source_col": "amount", "target_col": "amount",
             "data_type": "float", "description": "Order value"},
        ], "bronze")
        context = sttm_generator.load_context(
            SttmInputs("silver", "ctx-prev", "q", parquet_paths=tuple(paths), previous_sttm_path=str(previous))
        )
        self.assertEqual(context["previous_mappings"][0]["description"], "Order value")

    def test_rules_are_layer_specific(self):
        self.assertNotIn("fillna", sttm_generator.layer_rules("bronze")["allowed_operations"])
        self.assertIn("fillna", sttm_generator.layer_rules("silver")["allowed_operations"])
        self.assertEqual(sttm_generator.layer_rules("silver")["aggregations"], [])
        self.assertIn("sum", sttm_generator.layer_rules("gold")["aggregations"])


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.scratchpad = {}

    def tools(self, layer, run_id):
        root = Path(self.dir.name)
        if layer == "bronze":
            inputs = SttmInputs(layer, run_id, "q", profile_path=make_profile(run_id, root))
        elif layer == "silver":
            inputs = SttmInputs(layer, run_id, "q", parquet_paths=tuple(write_bronze(run_id)))
        else:
            inputs = SttmInputs(layer, run_id, "q", parquet_paths=tuple(write_silver(run_id)))
        return sttm_generator.build_sttm_tools(inputs, self.scratchpad)

    def test_agent_gets_only_inspect_and_its_own_generate_tool(self):
        for layer in ("bronze", "silver", "gold"):
            inspect, generate = self.tools(layer, f"tools-{layer}")
            self.assertEqual(
                (inspect.__name__, generate.__name__), ("inspect_context_tool", f"generate_{layer}_sttm_tool"),
            )
            self.assertGreater(len(generate.__doc__), 60)

    def test_generate_requires_inspect_first(self):
        _, generate = self.tools("bronze", "tools-gate")
        result = json.loads(generate(mappings=BRONZE_ROWS))
        self.assertEqual(result["status"], "error")
        self.assertIn("inspect_context_tool", result["message"])
        self.assertNotIn("sttm_bronze_path", self.scratchpad)

    def test_valid_bronze_rows_are_saved_and_registered(self):
        inspect, generate = self.tools("bronze", "tools-bronze")
        self.assertEqual(json.loads(inspect())["status"], "success")
        result = json.loads(generate(mappings=BRONZE_ROWS))
        self.assertEqual(result["status"], "success", result)
        self.assertEqual(result["row_count"], 5)
        self.assertEqual(result["columns_per_target_table"], {"orders": 3, "customers": 2})
        path = Path(self.scratchpad["sttm_bronze_path"])
        self.assertEqual(path, sttm_file("bronze", "tools-bronze"))
        rows = read_sttm(path, "bronze")
        self.assertEqual(rows[0]["target_table"], "orders")      # defaulted from source_table
        self.assertEqual(rows[-1]["transformation_logic"], "trim")

    def test_sloppy_nullable_values_from_a_model_are_accepted(self):
        base = BRONZE_ROWS[0]
        for sent, expected in ((None, True), ("", True), ("True", True), ("false", False), ("NO", False), (False, False)):
            self.assertEqual(SttmRow(**base, nullable=sent).nullable, expected, sent)
        with self.assertRaises(ValueError):
            SttmRow(**base, nullable="maybe")

    def test_rows_may_be_pydantic_models_or_dicts(self):
        inspect, generate = self.tools("bronze", "tools-models")
        inspect()
        result = json.loads(generate(mappings=[SttmRow(**row) for row in BRONZE_ROWS]))
        self.assertEqual(result["status"], "success")

    def test_all_problems_are_reported_at_once_and_nothing_is_written(self):
        inspect, generate = self.tools("bronze", "tools-bad")
        inspect()
        bad = BRONZE_ROWS[:3] + [
            {"source_table": "orders", "source_col": "ghost", "target_col": "g", "data_type": "string"},
            {"source_table": "orders", "source_col": "status", "target_col": "s", "data_type": "string",
             "transformation_logic": "fillna:mode"},
            {"source_table": "orders", "source_col": "region", "target_col": "r", "data_type": "string",
             "aggregation": "sum"},
        ]
        result = json.loads(generate(mappings=bad))
        self.assertEqual(result["status"], "error")
        message = result["message"]
        for expected in ("ghost", "fillna", "aggregation is only allowed in the gold layer", "customers"):
            self.assertIn(expected, message)
        self.assertNotIn("sttm_bronze_path", self.scratchpad)
        self.assertFalse(sttm_file("bronze", "tools-bad").exists())

    def test_silver_must_map_every_bronze_table(self):
        inspect, generate = self.tools("silver", "tools-silver")
        inspect()
        orders_only = [{**r, "layer": "silver"} for r in BRONZE_ROWS if r["source_table"] == "orders"]
        result = json.loads(generate(mappings=orders_only))
        self.assertEqual(result["status"], "error")
        self.assertIn("customers", result["message"])
        ok_rows = BRONZE_ROWS + [
            {"source_table": "orders", "source_col": "order_date", "target_col": "order_date",
             "data_type": "date", "transformation_logic": "to_date"},
            {"source_table": "orders", "source_col": "discount", "target_col": "discount",
             "data_type": "float", "transformation_logic": "fillna:zero"},
            {"source_table": "orders", "source_col": "order_id", "target_col": "order_key",
             "data_type": "integer", "transformation_logic": "dedupe"},
        ]
        result = json.loads(generate(mappings=ok_rows))
        self.assertEqual(result["status"], "success", result)
        self.assertEqual(Path(self.scratchpad["sttm_silver_path"]).name, "sttm_silver_tools-silver.csv")

    def test_gold_requires_valid_join_keys_and_aggregations(self):
        inspect, generate = self.tools("gold", "tools-gold")
        preview = json.loads(inspect())
        self.assertIn("pk_orders_silver_id", [c["name"] for c in preview["source_tables"]["orders"]["columns"]])
        no_key = [{**r, "join_key": ""} for r in GOLD_ROWS]
        self.assertIn("join_key", json.loads(generate(mappings=no_key))["message"])
        bad_key = [{**r, "join_key": "order_id" if r.get("join_key") else ""} for r in GOLD_ROWS]
        self.assertIn("not a column of 'customers'", json.loads(generate(mappings=bad_key))["message"])
        bad_agg = [dict(GOLD_ROWS[0], aggregation="sum"), *GOLD_ROWS[1:]]
        self.assertIn("data_type must be integer or float", json.loads(generate(mappings=bad_agg))["message"])
        result = json.loads(generate(mappings=GOLD_ROWS))
        self.assertEqual(result["status"], "success", result)
        self.assertEqual(result["columns_per_target_table"], {"sales_by_customer": 3})

    def test_unknown_table_in_inspect_is_an_error(self):
        inspect, _ = self.tools("bronze", "tools-unknown")
        self.assertEqual(json.loads(inspect(tables=["nope"]))["status"], "error")


class AgentLoopTests(unittest.TestCase):
    """Run the real create_agent ReAct loop with a scripted model."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.scratchpad = {}

    def inputs(self, layer, run_id):
        root = Path(self.dir.name)
        if layer == "bronze":
            return SttmInputs(layer, run_id, "Which customers buy most?", profile_path=make_profile(run_id, root))
        paths = write_bronze(run_id) if layer == "silver" else write_silver(run_id)
        return SttmInputs(layer, run_id, "Which customers buy most?", parquet_paths=tuple(paths))

    def run_agent(self, inputs, model):
        return json.loads(sttm_generator.run_sttm_agent(inputs, "design the mapping", self.scratchpad, model=model))

    def trace(self, run_id):
        return json.loads(trace_file("sttm", run_id).read_text(encoding="utf-8"))[-1]

    @staticmethod
    def generate_call(tool_name, rows, call_id="call_2"):
        return ai("Generating the mapping.", tool_call(tool_name, call_id, mappings=rows))

    def test_bronze_flow_succeeds_and_is_traced(self):
        model = ScriptedChatModel(script=[
            ai("Plan: inspect the profile, then map the needed columns.", tool_call("inspect_context_tool", "call_1")),
            lambda messages: (
                self.assertIn("orders", last_tool_result(messages)["source_tables"])
                or self.generate_call("generate_bronze_sttm_tool", BRONZE_ROWS)
            ),
            ai("Mapped orders and customers."),
        ])
        result = self.run_agent(self.inputs("bronze", "agent-bronze"), model)
        self.assertEqual(result["status"], "success", result)
        self.assertEqual(result["row_count"], 5)
        self.assertEqual(self.scratchpad["sttm_bronze_path"], result["sttm_path"])
        trace = self.trace("agent-bronze")
        self.assertEqual(trace["status"], "success")
        self.assertEqual([c["name"] for c in trace["tool_calls"]], ["inspect_context_tool", "generate_bronze_sttm_tool"])
        self.assertIn("Plan", trace["plan"])
        self.assertEqual(trace["input"]["layer"], "bronze")

    def test_agent_fixes_validation_errors_and_retries(self):
        broken = [dict(BRONZE_ROWS[0], source_col="missing_col")] + BRONZE_ROWS[1:]
        model = ScriptedChatModel(script=[
            ai("Plan.", tool_call("inspect_context_tool", "call_1")),
            self.generate_call("generate_bronze_sttm_tool", broken),
            lambda messages: (
                self.assertIn("missing_col", last_tool_result(messages)["message"])
                or self.generate_call("generate_bronze_sttm_tool", BRONZE_ROWS, "call_3")
            ),
            ai("Fixed the column name."),
        ])
        result = self.run_agent(self.inputs("bronze", "agent-retry"), model)
        self.assertEqual(result["status"], "success", result)
        self.assertEqual(len(self.trace("agent-retry")["tool_calls"]), 3)

    def test_silver_and_gold_flows(self):
        silver_rows = BRONZE_ROWS + [
            {"source_table": "orders", "source_col": "order_date", "target_col": "order_date",
             "data_type": "date", "transformation_logic": "to_date"},
        ]
        for layer, rows in (("silver", silver_rows), ("gold", GOLD_ROWS)):
            model = ScriptedChatModel(script=[
                ai("Plan.", tool_call("inspect_context_tool", "call_1")),
                self.generate_call(f"generate_{layer}_sttm_tool", rows),
                ai("Done."),
            ])
            result = self.run_agent(self.inputs(layer, f"agent-{layer}"), model)
            self.assertEqual(result["status"], "success", result)
            self.assertEqual(read_sttm(result["sttm_path"], layer)[0]["layer"], layer)
            self.assertEqual(self.scratchpad[f"sttm_{layer}_path"], result["sttm_path"])

    def test_agent_that_never_saves_fails_cleanly(self):
        model = ScriptedChatModel(script=[
            ai("Plan.", tool_call("inspect_context_tool", "call_1")),
            ai("I think the data is fine as is."),
        ])
        result = self.run_agent(self.inputs("bronze", "agent-lazy"), model)
        self.assertEqual(result["status"], "error")
        self.assertIn("without saving", result["message"])
        self.assertEqual(self.trace("agent-lazy")["status"], "failed")
        self.assertNotIn("sttm_bronze_path", self.scratchpad)

    def test_bad_inputs_return_error_not_exception(self):
        inputs = SttmInputs("bronze", "agent-missing", "q", profile_path=str(Path(self.dir.name) / "nope.json"))
        result = self.run_agent(inputs, ScriptedChatModel(script=[]))
        self.assertEqual(result["status"], "error")

    def test_regeneration_overwrites_the_same_run_scoped_file(self):
        inputs = self.inputs("bronze", "agent-regen")
        for rows in (BRONZE_ROWS, BRONZE_ROWS[:3] + BRONZE_ROWS[3:]):
            model = ScriptedChatModel(script=[
                ai("Plan.", tool_call("inspect_context_tool", "call_1")),
                self.generate_call("generate_bronze_sttm_tool", rows),
                ai("Done."),
            ])
            self.assertEqual(self.run_agent(inputs, model)["status"], "success")
        shrunk = ScriptedChatModel(script=[
            ai("Plan.", tool_call("inspect_context_tool", "call_1")),
            self.generate_call("generate_bronze_sttm_tool", BRONZE_ROWS[2:]),
            ai("Done."),
        ])
        self.assertEqual(self.run_agent(inputs, shrunk)["status"], "success")
        self.assertEqual(len(read_sttm(sttm_file("bronze", "agent-regen"), "bronze")), 3)

    def test_supervisor_tool_is_named_per_layer_and_wraps_the_agent(self):
        inputs = self.inputs("bronze", "agent-sup")
        model = ScriptedChatModel(script=[
            ai("Plan.", tool_call("inspect_context_tool", "call_1")),
            self.generate_call("generate_bronze_sttm_tool", BRONZE_ROWS),
            ai("Done."),
        ])
        tool = sttm_generator.make_sttm_agent_tool(inputs, self.scratchpad, model=model)
        self.assertEqual(tool.__name__, "sttm_bronze_agent")
        self.assertIn("BRONZE", tool.__doc__)
        self.assertIn("sttm_bronze_path", tool.__doc__)
        result = json.loads(safe_tool(tool)("design the bronze mapping"))
        self.assertEqual(result["status"], "success", result)

    def test_agent_receives_only_two_tools_and_a_layer_prompt(self):
        seen = {}

        def factory(model, tools, system_prompt=None):
            from langchain.agents import create_agent

            seen["tools"] = [t.__name__ for t in tools]
            seen["prompt"] = system_prompt
            return create_agent(model, tools, system_prompt=system_prompt)

        sttm_generator.run_sttm_agent(
            self.inputs("gold", "agent-tools"), "go", self.scratchpad,
            model=ScriptedChatModel(script=[ai("Nothing to do.")]), agent_factory=factory,
        )
        self.assertEqual(seen["tools"], ["inspect_context_tool", "generate_gold_sttm_tool"])
        self.assertIn("GOLD", seen["prompt"])
        self.assertIn("join_key", seen["prompt"])


if __name__ == "__main__":
    unittest.main()
