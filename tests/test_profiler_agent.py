"""Offline tests for the Profiler agent (Step 2)."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from agents import profiler
from agents.orchestrator import safe_tool
from core.paths import profile_file, trace_file
from tests.fakes import ScriptedChatModel, ai, last_tool_result, tool_call
from tests.fixtures import make_customers, make_orders, write_csv


class ProfileContentTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        root = Path(self.dir.name)
        self.orders = write_csv(root, "orders.csv", make_orders())
        self.customers = write_csv(root, "customers.csv", make_customers())

    def profile(self, run_id="prof-content", **kwargs):
        return profiler.build_profile([self.orders, self.customers], run_id, "sales", **kwargs)

    def column(self, profile, table, name):
        dataset = next(d for d in profile["datasets"] if d["table"] == table)
        return next(c for c in dataset["columns"] if c["name"] == name)

    def test_numeric_stats_nulls_and_types(self):
        profile, path = self.profile()
        self.assertEqual(path, profile_file("prof-content"))
        amount = self.column(profile, "orders", "amount")
        self.assertEqual((amount["inferred_type"], amount["min"], amount["max"]), ("float", 10.5, 315.0))
        order_id = self.column(profile, "orders", "order_id")
        self.assertEqual((order_id["inferred_type"], order_id["min"], order_id["max"]), ("integer", 1, 30))
        discount = self.column(profile, "orders", "discount")
        self.assertEqual((discount["null_count"], discount["null_pct"]), (21, 67.74))
        self.assertEqual(self.column(profile, "orders", "order_date")["inferred_type"], "date")
        self.assertEqual(self.column(profile, "orders", "customer_id")["semantic_role"], "identifier")

    def test_quality_notes_cover_nulls_cardinality_whitespace_and_duplicates(self):
        profile, _ = self.profile()
        notes = next(d for d in profile["datasets"] if d["table"] == "orders")["quality_notes"]
        joined = "\n".join(notes)
        self.assertIn("discount: high nulls", joined)
        self.assertIn("status: low cardinality (2 distinct values)", joined)
        self.assertIn("region: ", joined)
        self.assertIn("whitespace", joined)
        self.assertIn("1 fully duplicated row", joined)
        self.assertTrue(any(n.startswith("orders: ") for n in profile["quality_notes"]))

    def test_join_key_discovered_with_relationship(self):
        profile, _ = self.profile()
        self.assertEqual(len(profile["join_keys"]), 1)
        key = profile["join_keys"][0]
        self.assertEqual((key["left_table"], key["right_table"]), ("orders", "customers"))
        self.assertEqual((key["left_column"], key["right_column"]), ("customer_id", "customer_id"))
        self.assertEqual((key["overlap_ratio"], key["relationship"]), (1.0, "many_to_one"))

    def test_shared_non_key_column_without_overlap_is_not_a_join_key(self):
        root = Path(self.dir.name)
        a = write_csv(root, "a.csv", pd.DataFrame({"x": [1, 1, 2], "label": ["p", "p", "q"]}))
        b = write_csv(root, "b.csv", pd.DataFrame({"y": [1, 2, 2], "label": ["r", "s", "s"]}))
        profile, _ = profiler.build_profile([a, b], "prof-nokey", "")
        self.assertEqual(profile["join_keys"], [])

    def test_semantic_meanings_are_merged_with_fallback_and_unknowns_reported(self):
        profile, _ = self.profile(
            semantic_meanings={
                "orders": {"amount": "Order value in USD", "ghost": "x"},
                "nope": {"a": "b"},
            }
        )
        meanings = profile["semantic_meanings"]
        self.assertEqual(meanings["orders"]["amount"], "Order value in USD")
        self.assertEqual(meanings["orders"]["status"], "status (dimension)")
        self.assertEqual(set(meanings), {"orders", "customers"})
        self.assertEqual(set(profile["ignored_semantic_meanings"]), {"orders.ghost", "nope.*"})

    def test_profile_is_valid_json_on_disk(self):
        _, path = self.profile("prof-disk")
        stored = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(stored["run_id"], "prof-disk")
        self.assertEqual({d["table"] for d in stored["datasets"]}, {"orders", "customers"})

    def test_runs_do_not_overwrite_each_other(self):
        _, first = self.profile("prof-run-a")
        _, second = self.profile("prof-run-b")
        self.assertNotEqual(first, second)
        self.assertTrue(first.exists() and second.exists())


class ReadingTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)

    def test_rejects_bad_headers_and_missing_files(self):
        cases = {"dup.csv": "a,A\n1,2\n", "blank.csv": "a,\n1,2\n", "empty.csv": ""}
        for name, text in cases.items():
            (self.root / name).write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError, msg=name):
                profiler.build_profile([str(self.root / name)], "prof-bad", "")
        with self.assertRaises(FileNotFoundError):
            profiler.inspect_csv(self.root / "missing.csv")

    def test_header_only_file_is_noted_not_crashed(self):
        (self.root / "headers.csv").write_text("a,b\n", encoding="utf-8")
        profile, _ = profiler.build_profile([str(self.root / "headers.csv")], "prof-header", "")
        self.assertEqual(profile["datasets"][0]["row_count"], 0)
        self.assertIn("no data rows", profile["datasets"][0]["quality_notes"][0])

    def test_duplicate_table_names_are_rejected(self):
        (self.root / "x").mkdir()
        for path in (self.root / "t.csv", self.root / "x" / "t.csv"):
            path.write_text("a\n1\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            profiler.build_profile([str(self.root / "t.csv"), str(self.root / "x" / "t.csv")], "prof-dup", "")

    def test_latin1_file_is_read(self):
        (self.root / "latin.csv").write_bytes("name\ncafé\n".encode("latin-1"))
        profile, _ = profiler.build_profile([str(self.root / "latin.csv")], "prof-latin", "")
        self.assertEqual(profile["datasets"][0]["encoding"], "latin-1")

    def test_inspect_is_lightweight_and_formats_integer_samples(self):
        path = write_csv(self.root, "n.csv", pd.DataFrame({"id": [1, 2, None, 4, 5], "v": ["a", "b", "c", "d", "e"]}))
        preview = profiler.inspect_csv(path)
        self.assertEqual(preview["shape"], [5, 2])
        ids = preview["columns"][0]
        self.assertEqual(ids["inferred_type"], "integer")
        self.assertEqual(ids["sample_values"], ["1", "2", "4"])
        self.assertEqual(len(preview["columns"][1]["sample_values"]), 3)
        self.assertNotIn("null_count", ids)


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.orders = write_csv(self.root, "orders.csv", make_orders())
        self.scratchpad = {}
        self.inspect, self.profile = profiler.build_profiler_tools(
            [self.orders], "tool-run", "sales", self.scratchpad,
        )

    def test_profiler_tool_refuses_to_run_before_inspect(self):
        result = json.loads(self.profile())
        self.assertEqual(result["status"], "error")
        self.assertIn("inspect_files_tool", result["message"])
        self.assertNotIn("profile_path", self.scratchpad)

    def test_inspect_then_profile_writes_scratchpad(self):
        preview = json.loads(self.inspect())
        self.assertEqual(preview["status"], "success")
        self.assertEqual(preview["files"][0]["table"], "orders")
        result = json.loads(self.profile(semantic_meanings={"orders": {"amount": "Order value"}}))
        self.assertEqual(result["status"], "success")
        self.assertEqual(self.scratchpad["profile_path"], result["profile_path"])
        self.assertTrue(Path(result["profile_path"]).is_file())

    def test_unknown_paths_are_rejected_with_the_valid_ones(self):
        result = json.loads(self.inspect(csv_paths=["C:/somewhere/else.csv"]))
        self.assertEqual(result["status"], "error")
        self.assertIn("Use only these paths", result["message"])
        self.assertIn("orders.csv", result["message"])

    def test_tools_never_raise_and_have_descriptions(self):
        Path(self.orders).unlink()
        self.assertEqual(json.loads(self.inspect())["status"], "error")
        for tool in (self.inspect, self.profile):
            self.assertTrue(tool.__doc__ and len(tool.__doc__) > 40)


class AgentLoopTests(unittest.TestCase):
    """Run the real create_agent ReAct loop with a scripted model."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        root = Path(self.dir.name)
        self.paths = [
            write_csv(root, "orders.csv", make_orders()),
            write_csv(root, "customers.csv", make_customers()),
        ]
        self.scratchpad = {}

    @staticmethod
    def meanings_from_preview(messages):
        preview = last_tool_result(messages)
        meanings = {
            f["table"]: {c["name"]: f"{c['name']} meaning" for c in f["columns"]}
            for f in preview["files"]
        }
        return ai("Profiling with meanings.", tool_call("profiler_tool", "call_2", semantic_meanings=meanings))

    def run_agent(self, model, run_id):
        return json.loads(
            profiler.run_profiler_agent(
                self.paths, "Which customers buy most?", run_id, "profile everything", self.scratchpad, model=model,
            )
        )

    def read_trace(self, run_id):
        return json.loads(trace_file("profiler", run_id).read_text(encoding="utf-8"))[-1]

    def test_inspect_then_profile_succeeds_and_is_traced(self):
        model = ScriptedChatModel(script=[
            ai("Plan: preview the files, then profile them.", tool_call("inspect_files_tool", "call_1")),
            self.meanings_from_preview,
            ai("Profiled 2 tables; customer_id joins them."),
        ])
        result = self.run_agent(model, "agent-ok")
        self.assertEqual(result["status"], "success")
        self.assertIn("customer_id", result["message"])
        profile = json.loads(Path(self.scratchpad["profile_path"]).read_text(encoding="utf-8"))
        self.assertEqual(profile["semantic_meanings"]["orders"]["amount"], "amount meaning")
        trace = self.read_trace("agent-ok")
        self.assertEqual(trace["status"], "success")
        self.assertEqual([c["name"] for c in trace["tool_calls"]], ["inspect_files_tool", "profiler_tool"])
        self.assertIn("Plan", trace["plan"])
        self.assertEqual(trace["input"]["task_description"], "profile everything")
        self.assertGreater(trace["duration_seconds"], 0)
        self.assertEqual(trace["output"]["profile_path"], self.scratchpad["profile_path"])

    def test_agent_adapts_after_calling_profiler_before_inspect(self):
        model = ScriptedChatModel(script=[
            ai("Profiling right away.", tool_call("profiler_tool", "call_1")),
            lambda messages: (
                self.assertEqual(last_tool_result(messages)["status"], "error")
                or ai("I must inspect first.", tool_call("inspect_files_tool", "call_2"))
            ),
            lambda messages: ai("Now profiling.", tool_call("profiler_tool", "call_3")),
            ai("Done."),
        ])
        result = self.run_agent(model, "agent-adapt")
        self.assertEqual(result["status"], "success")
        names = [c["name"] for c in self.read_trace("agent-adapt")["tool_calls"]]
        self.assertEqual(names, ["profiler_tool", "inspect_files_tool", "profiler_tool"])

    def test_agent_that_never_profiles_fails_cleanly(self):
        model = ScriptedChatModel(script=[
            ai("Plan: inspect.", tool_call("inspect_files_tool", "call_1")),
            ai("I looked at the data and I'm done."),
        ])
        result = self.run_agent(model, "agent-lazy")
        self.assertEqual(result["status"], "error")
        self.assertIn("without producing a profile", result["message"])
        self.assertEqual(self.read_trace("agent-lazy")["status"], "failed")
        self.assertNotIn("profile_path", self.scratchpad)

    def test_model_failure_is_returned_as_error_not_raised(self):
        result = self.run_agent(ScriptedChatModel(script=[]), "agent-boom")
        self.assertEqual(result["status"], "error")
        self.assertEqual(self.read_trace("agent-boom")["status"], "failed")

    def test_supervisor_tool_wraps_the_agent(self):
        model = ScriptedChatModel(script=[
            ai("Plan.", tool_call("inspect_files_tool", "call_1")),
            self.meanings_from_preview,
            ai("Done."),
        ])
        tool = profiler.make_profiler_agent_tool(
            self.paths, "intent", "agent-sup", self.scratchpad, model=model,
        )
        self.assertEqual(tool.__name__, "profiler_agent")
        self.assertIn("profile", tool.__doc__.lower())
        result = json.loads(safe_tool(tool)("profile the raw files"))
        self.assertEqual(result["status"], "success")
        self.assertTrue(Path(self.scratchpad["profile_path"]).is_file())

    def test_agent_only_has_its_two_tools(self):
        model = ScriptedChatModel(script=[ai("Nothing to do.")])
        seen_tools = []

        def factory(model, tools, system_prompt=None):
            seen_tools.extend(t.__name__ for t in tools)
            from langchain.agents import create_agent

            return create_agent(model, tools, system_prompt=system_prompt)

        profiler.run_profiler_agent(
            self.paths, "intent", "agent-tools", "go", self.scratchpad, model=model, agent_factory=factory,
        )
        self.assertEqual(seen_tools, ["inspect_files_tool", "profiler_tool"])


if __name__ == "__main__":
    unittest.main()
