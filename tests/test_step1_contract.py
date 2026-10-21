"""Offline interface tests for the Step 1 control boundary."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.orchestrator import Orchestrator
from agents.supervisor import Supervisor
from core.state import PipelineState


class FakeAudit:
    def __init__(self):
        self.events = []

    def log(self, agent, action, **details):
        self.events.append((agent, action, details))


class StepOneContractTests(unittest.TestCase):
    def test_supervisor_invokes_only_phase_tools_and_writes_scratchpad(self):
        scratchpad = {}

        def profiler_tool(task_description: str) -> str:
            """Inspect and profile the supplied CSV files."""
            scratchpad["profile_path"] = "data/profiles/profile_run.json"
            return json.dumps({"status": "success", "profile_path": scratchpad["profile_path"]})

        def forbidden_tool(task_description: str) -> str:
            """This tool must not be visible in phase 1."""
            self.fail("Out-of-phase tool was called")

        class FakeAgent:
            def invoke(self, input_data, config=None):
                self.assert_tools(["profiler_tool"])
                result = profiler_tool("profile raw data")
                return {"messages": [type("Message", (), {"content": result})()]}

            def assert_tools(self, expected):
                self_outer.assertEqual(visible_tools, expected)

        self_outer = self
        visible_tools = []

        def factory(model, tools, system_prompt=None):
            visible_tools.extend(tool.__name__ for tool in tools)
            return FakeAgent()

        from core import observability

        with tempfile.TemporaryDirectory() as directory, patch.object(
            observability, "TRACES_DIR", Path(directory)
        ):
            supervisor = Supervisor(model=object(), agent_factory=factory)
            result = supervisor.invoke("profile files", [profiler_tool], scratchpad, "run-1")
        self.assertEqual(result["status"], "success")
        self.assertEqual(scratchpad["profile_path"], "data/profiles/profile_run.json")
        self.assertNotIn(forbidden_tool.__name__, visible_tools)

    def test_supervisor_accepts_successful_retry_after_tool_error(self):
        from core import observability

        def phase_tool() -> str:
            """Return a structured result for a phase."""
            return json.dumps({"status": "success", "profile_path": "profile.json"})

        class FakeAgent:
            def invoke(self, input_data, config=None):
                return {"messages": [
                    type("Message", (), {"content": '{"status":"error","message":"try again"}'})(),
                    type("Message", (), {"content": phase_tool()})(),
                ]}

        with tempfile.TemporaryDirectory() as directory, patch.object(
            observability, "TRACES_DIR", Path(directory)
        ):
            supervisor = Supervisor(model=object(), agent_factory=lambda model, tools, system_prompt=None: FakeAgent())
            self.assertEqual(
                supervisor.invoke("profile", [phase_tool], {}, "retry-run")["status"], "success"
            )

    def test_orchestrator_copies_only_declared_scratchpad_outputs(self):
        audit = FakeAudit()
        state = PipelineState(run_id="run-1", csv_paths=["raw.csv"], business_intent="sales")
        scratchpad = {"profile_path": "data/profiles/profile_run.json", "bronze_output_paths": ["data/bronze_layer/orders_run-1_bronze.parquet"], "unknown": "ignore"}
        orchestrator = Orchestrator(audit_logger=audit)

        result = orchestrator.run_step(
            state, phase="phase_1", goal="profile raw data", tools=[lambda: None],
            scratchpad=scratchpad,
            supervisor=type("FakeSupervisor", (), {"invoke": lambda self, *args: {"status": "success"}})(),
            output_keys=("profile_path",),
        )
        self.assertIs(result, state)
        self.assertEqual(state.profile_path, scratchpad["profile_path"])
        self.assertEqual(state.bronze_output_paths, [])
        self.assertEqual(state.status, "pending")
        self.assertEqual(audit.events[-1][1], "phase_1_complete")

    def test_error_response_sets_failed_state_and_audits_without_raising(self):
        audit = FakeAudit()
        state = PipelineState(run_id="run-2")
        orchestrator = Orchestrator(audit_logger=audit)
        def failing_tool() -> str:
            """Return a structured error instead of raising."""
            return json.dumps({"status": "error", "message": "source unavailable"})

        class FakeSupervisor:
            def invoke(self, goal, tools, scratchpad, run_id):
                return json.loads(tools[0]())

        supervisor = FakeSupervisor()
        result = orchestrator.run_step(state, "phase_1", "profile", [failing_tool], {}, supervisor, ("profile_path",))
        self.assertIs(result, state)
        self.assertEqual(state.status, "failed")
        self.assertEqual(state.error, "source unavailable")
        self.assertEqual(state.errors, ["source unavailable"])
        self.assertEqual(audit.events[-1][1], "phase_1_failed")

    def test_tool_exception_is_structured_and_partial_handoff_is_not_copied(self):
        from agents.orchestrator import safe_tool

        def broken() -> str:
            """Simulate a specialist tool failure."""
            raise RuntimeError("disk unavailable")

        self.assertEqual(json.loads(safe_tool(broken)())["status"], "error")
        state = PipelineState(run_id="run-3")
        scratchpad = {"profile_path": "profile.json"}
        supervisor = type("FakeSupervisor", (), {"invoke": lambda self, *args: {"status": "success"}})()
        orchestrator = Orchestrator(audit_logger=FakeAudit())
        orchestrator.run_step(
            state, "phase_1", "prepare", [safe_tool(broken)], scratchpad,
            supervisor, ("profile_path", "sttm_bronze_path"),
        )
        self.assertEqual(state.status, "failed")
        self.assertEqual(state.profile_path, "")

    def test_output_path_patterns_are_run_scoped(self):
        from core.paths import bronze_file, gold_file, report_file, silver_file

        self.assertEqual(bronze_file("orders", "run-1").name, "orders_run-1_bronze.parquet")
        self.assertEqual(silver_file("orders", "run-1").name, "orders_run-1_silver.parquet")
        self.assertEqual(gold_file("sales", "run-1").name, "sales_run-1.parquet")
        self.assertEqual(report_file("run-1").name, "report_run-1.html")

    def test_trace_retains_each_invocation_for_one_agent_run(self):
        from core import observability

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(observability, "TRACES_DIR", Path(directory)):
                observability.AgentTrace("profiler", "run-12345678").set_output(path="first").complete()
                observability.AgentTrace("profiler", "run-12345678").set_output(path="second").complete()
                path = Path(directory) / "trace_profiler_run-1234.json"
                records = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual([record["output"]["path"] for record in records], ["first", "second"])


if __name__ == "__main__":
    unittest.main()