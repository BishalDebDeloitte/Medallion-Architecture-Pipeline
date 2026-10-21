"""Offline tests for the Orchestrator phase loop and human-in-the-loop gates (Step 7)."""

import json
import os
import time
import unittest
from pathlib import Path

from agents.orchestrator import Orchestrator, PHASE_OUTPUTS, phase_goal
from core import run_store
from core.audit import AuditLogger
from core.paths import approved_sttm_file, sttm_file
from core.state import AWAITING_APPROVAL, COMPLETE, FAILED, PENDING, RUNNING, PipelineState
from core.sttm_schema import read_sttm, write_sttm
from core.tooling import err, ok


def sttm_rows(layer, table="orders", description="a column", extra=None):
    rows = [{
        "layer": layer, "target_table": table, "source_table": "orders", "source_col": "amount",
        "target_col": "amount", "data_type": "float", "transformation_logic": "", "aggregation": "",
        "nullable": "true", "join_key": "", "description": description,
    }]
    return rows + (extra or [])


class FakeToolkit:
    """Stands in for the real agent tools: writes the same scratchpad keys and real STTM files."""

    def __init__(self, available=None):
        self.failures: dict[str, str] = {}
        self.available = available
        self.regenerations = 0
        self.phases_run: list[str] = []

    def _tool(self, name, produce):
        def tool(task_description: str) -> str:
            """Fake specialist tool."""
            if name in self.failures:
                return err(self.failures[name])
            produce()
            return ok(message="done")

        tool.__name__ = name
        return tool

    def phase_tools(self, phase, state, scratchpad):
        self.phases_run.append(phase)
        rid = state.run_id

        def write(layer, description="generated"):
            path = write_sttm(sttm_file(layer, rid), sttm_rows(layer, description=description), layer)
            scratchpad[f"sttm_{layer}_path"] = str(path)

        def phase_1():
            scratchpad["profile_path"] = "profile.json"
            write("bronze")

        def phase_2():
            scratchpad["bronze_output_paths"] = ["bronze_orders.parquet"]
            write("silver")

        def phase_3():
            scratchpad["silver_output_paths"] = ["silver_orders.parquet"]
            write("gold")

        def phase_4():
            scratchpad["gold_output_paths"] = ["gold_t.parquet"]
            scratchpad["report_path"] = "report.html"
            scratchpad["report_json_path"] = "report.json"

        return [self._tool(phase, {"phase_1": phase_1, "phase_2": phase_2, "phase_3": phase_3, "phase_4": phase_4}[phase])]

    def regenerate_tools(self, layer, state, scratchpad):
        def produce():
            self.regenerations += 1
            path = write_sttm(
                sttm_file(layer, state.run_id), sttm_rows(layer, description=f"regenerated #{self.regenerations}"), layer,
            )
            scratchpad[f"sttm_{layer}_path"] = str(path)

        return [self._tool(f"regen_{layer}", produce)]

    def available_columns(self, layer, state):
        return self.available


class FakeSupervisor:
    """Calls every tool of the phase in order; stops at the first error (like a careful Supervisor)."""

    def __init__(self):
        self.goals: list[str] = []

    def invoke(self, goal, tools, scratchpad, run_id):
        self.goals.append(goal)
        for tool in tools:
            payload = json.loads(tool("task"))
            if payload["status"] == "error":
                return {"status": "error", "message": payload["message"]}
        return {"status": "success", "message": "ok"}


class LoopTestCase(unittest.TestCase):
    def setUp(self):
        self.toolkit = FakeToolkit()
        self.supervisor = FakeSupervisor()
        self.orchestrator = Orchestrator(toolkit=self.toolkit, supervisor=self.supervisor)
        self.state = self.orchestrator.start_run(["orders.csv"], "Which product sold most?")

    def actions(self, state=None):
        return [e["action"] for e in self.orchestrator.audit_logs(state or self.state)]

    def advance(self, rows=None):
        self.orchestrator.approve(self.state, rows)
        return self.orchestrator.run_next_phase(self.state)


class PhaseOneTests(LoopTestCase):
    def test_phase_one_pauses_for_approval_without_setting_the_approved_path(self):
        state = self.state
        self.assertEqual(state.status, AWAITING_APPROVAL)
        self.assertEqual(state.current_phase, "phase_1")
        self.assertEqual(state.profile_path, "profile.json")
        self.assertEqual((state.pending_approval_layer, state.pending_approval_path), ("bronze", str(sttm_file("bronze", state.run_id))))
        self.assertEqual(state.sttm_bronze_path, "")                       # only an approval may set it
        self.assertEqual(self.orchestrator.next_phase(state), None)

    def test_audit_trail_records_decisions_and_rationale(self):
        self.assertEqual(
            self.actions(),
            ["run_created", "phase_selected", "phase_1_started", "phase_1_complete", "awaiting_approval"],
        )
        selected = next(e for e in self.orchestrator.audit_logs(self.state) if e["action"] == "phase_selected")
        self.assertEqual(selected["selected"], "phase_1")
        self.assertIn("derived from existing artifacts", selected["rationale"])

    def test_goal_is_deterministic_and_names_the_tools_and_question(self):
        self.assertEqual(self.supervisor.goals[0], phase_goal("phase_1", self.state))
        self.assertIn("profiler_agent", self.supervisor.goals[0])
        self.assertIn("Which product sold most?", self.supervisor.goals[0])

    def test_state_is_persisted_after_every_transition(self):
        saved = run_store.load_state(self.state.run_id)
        self.assertEqual(saved.model_dump(), self.state.model_dump())

    def test_running_the_next_phase_while_awaiting_approval_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "Approve or reject"):
            self.orchestrator.run_next_phase(self.state)

    def test_new_run_validates_inputs(self):
        with self.assertRaises(ValueError):
            self.orchestrator.new_run([], "question")
        with self.assertRaises(ValueError):
            self.orchestrator.new_run(["a.csv"], "   ")


class ApprovalTests(LoopTestCase):
    def test_approval_without_edits_stores_a_separate_approved_file(self):
        generated = self.state.pending_approval_path
        self.orchestrator.approve(self.state)
        approved = approved_sttm_file("bronze", self.state.run_id)
        self.assertEqual(self.state.sttm_bronze_path, str(approved))
        self.assertTrue(approved.is_file() and Path(generated).is_file())
        self.assertEqual((self.state.pending_approval_path, self.state.pending_approval_layer), ("", ""))
        self.assertEqual(self.state.status, PENDING)
        event = next(e for e in self.orchestrator.audit_logs(self.state) if e["action"] == "sttm_approved")
        self.assertFalse(event["edited"])

    def test_edits_are_validated_saved_and_audited(self):
        rows = self.orchestrator.pending_rows(self.state)
        rows[0]["description"] = "Reviewed: order value in USD"
        self.orchestrator.approve(self.state, rows)
        self.assertEqual(read_sttm(self.state.sttm_bronze_path, "bronze")[0]["description"], "Reviewed: order value in USD")
        self.assertEqual(read_sttm(sttm_file("bronze", self.state.run_id), "bronze")[0]["description"], "generated")
        event = next(e for e in self.orchestrator.audit_logs(self.state) if e["action"] == "sttm_approved")
        self.assertTrue(event["edited"])

    def test_invalid_edits_are_rejected_and_nothing_changes(self):
        rows = self.orchestrator.pending_rows(self.state)
        rows[0]["transformation_logic"] = "eval:os.system('x')"
        with self.assertRaisesRegex(ValueError, "Unsupported transformation"):
            self.orchestrator.approve(self.state, rows)
        self.assertEqual(self.state.status, AWAITING_APPROVAL)
        self.assertEqual(self.state.sttm_bronze_path, "")
        self.assertFalse(approved_sttm_file("bronze", self.state.run_id).exists())
        self.assertIn("sttm_edit_rejected", self.actions())

    def test_edits_are_checked_against_the_real_source_columns(self):
        self.toolkit.available = {"orders": {"order_id"}}
        with self.assertRaisesRegex(ValueError, "does not exist in 'orders'"):
            self.orchestrator.approve(self.state)
        self.toolkit.available = {"orders": {"amount"}}
        self.orchestrator.approve(self.state)

    def test_approval_requires_a_pending_mapping(self):
        self.orchestrator.approve(self.state)
        with self.assertRaisesRegex(RuntimeError, "No mapping is awaiting approval"):
            self.orchestrator.approve(self.state)
        with self.assertRaises(RuntimeError):
            self.orchestrator.pending_rows(self.state)


class FullRunTests(LoopTestCase):
    def test_four_phases_with_a_pause_after_each_mapping(self):
        self.advance()
        self.assertEqual((self.state.current_phase, self.state.status), ("phase_2", AWAITING_APPROVAL))
        self.assertEqual(self.state.bronze_output_paths, ["bronze_orders.parquet"])
        self.assertEqual(self.state.pending_approval_layer, "silver")
        self.advance()
        self.assertEqual((self.state.current_phase, self.state.pending_approval_layer), ("phase_3", "gold"))
        self.assertEqual(self.state.silver_output_paths, ["silver_orders.parquet"])
        self.advance()
        self.assertEqual((self.state.current_phase, self.state.status), ("phase_4", COMPLETE))
        self.assertEqual(
            (self.state.gold_output_paths, self.state.report_path, self.state.report_json_path),
            (["gold_t.parquet"], "report.html", "report.json"),
        )
        self.assertEqual(self.toolkit.phases_run, ["phase_1", "phase_2", "phase_3", "phase_4"])
        self.assertIn("run_complete", self.actions())
        self.assertIsNone(self.orchestrator.next_phase(self.state))
        self.assertIs(self.orchestrator.run_next_phase(self.state), self.state)         # no-op once complete
        self.assertEqual(self.toolkit.phases_run.count("phase_4"), 1)

    def test_next_phase_is_derived_from_artifacts(self):
        s = PipelineState(run_id="x", status=PENDING)
        self.assertEqual(Orchestrator.next_phase(s), "phase_1")
        s.sttm_bronze_path = "b"
        self.assertEqual(Orchestrator.next_phase(s), "phase_2")
        s.sttm_silver_path = "s"
        self.assertEqual(Orchestrator.next_phase(s), "phase_3")
        s.sttm_gold_path = "g"
        self.assertEqual(Orchestrator.next_phase(s), "phase_4")
        s.report_path = "r"
        self.assertIsNone(Orchestrator.next_phase(s))
        self.assertEqual(Orchestrator.next_phase(PipelineState(run_id="y", status=FAILED)), "phase_1")


class FailureTests(LoopTestCase):
    def test_a_failed_phase_is_recorded_not_raised_and_copies_nothing(self):
        self.toolkit.failures["phase_2"] = "disk unavailable"
        self.advance()
        state = self.state
        self.assertEqual((state.status, state.error, state.errors), (FAILED, "disk unavailable", ["disk unavailable"]))
        self.assertEqual(state.bronze_output_paths, [])
        self.assertEqual((state.pending_approval_path, state.sttm_silver_path), ("", ""))
        self.assertIn("phase_2_failed", self.actions())

    def test_retry_reruns_only_the_failed_phase(self):
        self.toolkit.failures["phase_2"] = "transient"
        self.advance()
        del self.toolkit.failures["phase_2"]
        self.orchestrator.retry(self.state)
        self.assertEqual((self.state.status, self.state.current_phase, self.state.error), (AWAITING_APPROVAL, "phase_2", None))
        self.assertEqual(self.toolkit.phases_run, ["phase_1", "phase_2", "phase_2"])
        self.assertIn("retry_requested", self.actions())
        self.assertEqual(self.state.errors, ["transient"])                                # history is kept

    def test_retry_of_a_failed_first_phase_starts_it_over(self):
        orchestrator = Orchestrator(toolkit=self.toolkit, supervisor=self.supervisor)
        self.toolkit.failures["phase_1"] = "bad csv"
        state = orchestrator.start_run(["orders.csv"], "q")
        self.assertEqual(state.status, FAILED)
        del self.toolkit.failures["phase_1"]
        self.assertEqual(orchestrator.retry(state).status, AWAITING_APPROVAL)

    def test_missing_scratchpad_output_fails_the_phase(self):
        class Lazy(FakeToolkit):
            def phase_tools(self, phase, state, scratchpad):
                scratchpad["profile_path"] = "p.json"          # but never produces the STTM
                return [self._tool("lazy", lambda: None)]

        orchestrator = Orchestrator(toolkit=Lazy(), supervisor=FakeSupervisor())
        state = orchestrator.start_run(["a.csv"], "q")
        self.assertEqual(state.status, FAILED)
        self.assertIn("Missing or invalid scratchpad output: sttm_bronze_path", state.error)
        self.assertEqual(state.profile_path, "")

    def test_errors_building_tools_and_supervisor_crashes_never_escape(self):
        class Broken(FakeToolkit):
            def phase_tools(self, phase, state, scratchpad):
                raise RuntimeError("cannot build tools")

        self.assertEqual(Orchestrator(toolkit=Broken(), supervisor=FakeSupervisor()).start_run(["a.csv"], "q").status, FAILED)

        class Crashing:
            def invoke(self, *args):
                raise RuntimeError("model down")

        crashed = Orchestrator(toolkit=FakeToolkit(), supervisor=Crashing()).start_run(["a.csv"], "q")
        self.assertEqual((crashed.status, crashed.error), (FAILED, "model down"))

    def test_an_approved_mapping_that_fails_on_the_data_can_be_reopened_edited_and_rerun(self):
        self.toolkit.failures["phase_2"] = "orders.amount has 2 missing value(s) but is nullable=false"
        self.advance()
        self.assertEqual(self.state.status, FAILED)
        approved_before = self.state.sttm_bronze_path
        self.orchestrator.reopen_mapping(self.state)
        self.assertEqual((self.state.status, self.state.pending_approval_layer), (AWAITING_APPROVAL, "bronze"))
        self.assertEqual((self.state.pending_approval_path, self.state.sttm_bronze_path), (approved_before, ""))
        self.assertIsNone(self.state.error)
        rows = self.orchestrator.pending_rows(self.state)
        rows[0]["nullable"] = "true"
        self.orchestrator.approve(self.state, rows)
        del self.toolkit.failures["phase_2"]
        self.orchestrator.run_next_phase(self.state)
        self.assertEqual((self.state.status, self.state.current_phase), (AWAITING_APPROVAL, "phase_2"))
        self.assertIn("mapping_reopened", self.actions())

    def test_reopen_is_only_possible_for_a_failed_execution_with_an_approved_mapping(self):
        with self.assertRaises(RuntimeError):
            self.orchestrator.reopen_mapping(self.state)                       # awaiting approval, not failed
        failed_first = Orchestrator(toolkit=FakeToolkit(), supervisor=FakeSupervisor())
        failed_first.toolkit.failures["phase_1"] = "bad csv"
        state = failed_first.start_run(["a.csv"], "q")
        with self.assertRaisesRegex(RuntimeError, "no approved mapping"):
            failed_first.reopen_mapping(state)                                 # phase 1 has no approved mapping yet

    def test_only_failed_uncancelled_runs_can_be_retried(self):
        with self.assertRaises(RuntimeError):
            self.orchestrator.retry(self.state)                  # awaiting approval, not failed


class RegenerateAndCancelTests(LoopTestCase):
    def test_regenerate_replaces_the_pending_mapping_and_passes_feedback(self):
        self.orchestrator.regenerate(self.state, "Keep only the amount column")
        self.assertEqual(self.state.status, AWAITING_APPROVAL)
        self.assertEqual(self.state.pending_approval_layer, "bronze")
        self.assertEqual(read_sttm(self.state.pending_approval_path, "bronze")[0]["description"], "regenerated #1")
        self.assertIn("Keep only the amount column", self.supervisor.goals[-1])
        self.assertIn("regenerate_requested", self.actions())
        self.assertEqual(self.toolkit.phases_run, ["phase_1"])        # phase 1 itself was not re-run
        self.orchestrator.regenerate(self.state, "")
        self.assertEqual(read_sttm(self.state.pending_approval_path, "bronze")[0]["description"], "regenerated #2")
        self.assertIn("none given", self.supervisor.goals[-1])

    def test_a_failed_regeneration_keeps_the_previous_mapping_reviewable(self):
        previous = self.state.pending_approval_path
        self.toolkit.regenerate_tools = lambda layer, state, scratchpad: [self.toolkit._tool("regen", lambda: None)]
        self.toolkit.failures["regen"] = "model unavailable"
        self.orchestrator.regenerate(self.state, "try again")
        self.assertEqual((self.state.status, self.state.pending_approval_path), (AWAITING_APPROVAL, previous))
        self.assertEqual(self.state.error, "model unavailable")
        self.assertEqual(self.orchestrator.pending_rows(self.state)[0]["description"], "generated")
        self.assertIn("regenerate_failed", self.actions())
        self.orchestrator.approve(self.state)                         # the original can still be approved
        self.assertTrue(self.state.sttm_bronze_path)

    def test_regenerate_requires_a_pending_mapping(self):
        self.orchestrator.approve(self.state)
        with self.assertRaises(RuntimeError):
            self.orchestrator.regenerate(self.state, "x")

    def test_cancel_is_terminal(self):
        self.orchestrator.cancel(self.state, "Wrong columns")
        state = self.state
        self.assertEqual((state.status, state.cancelled), (FAILED, True))
        self.assertIn("Wrong columns", state.error)
        self.assertEqual((state.pending_approval_path, state.sttm_bronze_path), ("", ""))
        with self.assertRaises(RuntimeError):
            self.orchestrator.retry(state)
        self.assertIn("run_cancelled", self.actions())


class ResumeTests(LoopTestCase):
    def test_a_saved_run_resumes_in_a_new_orchestrator(self):
        fresh = Orchestrator(toolkit=self.toolkit, supervisor=self.supervisor)
        resumed = fresh.resume(self.state.run_id)
        self.assertEqual(resumed.model_dump(), self.state.model_dump())
        self.assertEqual(fresh.pending_rows(resumed)[0]["target_col"], "amount")
        self.assertIn("run_resumed", [e["action"] for e in fresh.audit_logs(resumed)])
        self.assertIsNone(fresh.resume("no-such-run"))

    def test_a_run_interrupted_mid_phase_resumes_as_failed_and_retryable(self):
        self.state.status = RUNNING
        run_store.save_state(self.state)
        fresh = Orchestrator(toolkit=self.toolkit, supervisor=self.supervisor)
        resumed = fresh.resume(self.state.run_id)
        self.assertEqual(resumed.status, FAILED)
        self.assertIn("Interrupted", resumed.error)
        self.assertEqual(fresh.retry(resumed).status, AWAITING_APPROVAL)

    def test_runs_are_listed_newest_first(self):
        from core.paths import state_file

        other = self.orchestrator.new_run(["b.csv"], "second question")
        now = time.time()
        os.utime(state_file(self.state.run_id), (now - 60, now - 60))     # file times can tie within one tick
        os.utime(state_file(other.run_id), (now, now))
        runs = run_store.list_runs(limit=1000)
        self.assertEqual(runs[0]["run_id"], other.run_id)
        self.assertIn(self.state.run_id, [r["run_id"] for r in runs])
        self.assertEqual(runs[0]["business_intent"], "second question")

    def test_corrupt_state_files_are_ignored(self):
        from core.paths import state_file

        state_file("corrupt-run").write_text("{not json", encoding="utf-8")
        self.assertIsNone(run_store.load_state("corrupt-run"))
        self.assertTrue(all(r["run_id"] != "corrupt-run" for r in run_store.list_runs()))


class ExternalAuditTests(unittest.TestCase):
    def test_an_injected_audit_logger_receives_every_event(self):
        events = []

        class Recorder:
            def log(self, agent, action, **details):
                events.append((agent, action))

        orchestrator = Orchestrator(audit_logger=Recorder(), toolkit=FakeToolkit(), supervisor=FakeSupervisor(), persist=False)
        orchestrator.start_run(["a.csv"], "q")
        self.assertEqual({agent for agent, _ in events}, {"orchestrator"})
        self.assertIn("awaiting_approval", [action for _, action in events])
        self.assertEqual(orchestrator.audit_logs(PipelineState(run_id="x")), [])      # recorder has no get_logs


class PhaseOutputContractTests(unittest.TestCase):
    def test_outputs_match_the_handoff_contract(self):
        self.assertEqual(PHASE_OUTPUTS["phase_2"], ("bronze_output_paths", "sttm_silver_path"))
        self.assertEqual(PHASE_OUTPUTS["phase_4"], ("gold_output_paths", "report_path"))
        self.assertTrue(AuditLogger("contract-run").log_path.name.endswith("contract-run.jsonl"))


if __name__ == "__main__":
    unittest.main()
