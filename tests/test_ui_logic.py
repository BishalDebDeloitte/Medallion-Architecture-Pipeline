"""Offline tests for the pure UI logic (Step 8)."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from agents.orchestrator import Orchestrator
from app import ui_logic as ui
from core.state import AWAITING_APPROVAL, COMPLETE, FAILED, PENDING, RUNNING, PipelineState
from core.sttm_schema import STTM_COLUMNS
from tests.test_phase_loop import FakeSupervisor, FakeToolkit, sttm_rows


def upload(name, content=b"a,b\n1,2\n"):
    return SimpleNamespace(name=name, getvalue=lambda: content)


class TrackerTests(unittest.TestCase):
    def statuses(self, state):
        return [card["status"] for card in ui.phase_view(state)]

    def test_a_new_run_is_ready_for_phase_one(self):
        self.assertEqual(self.statuses(PipelineState(run_id="r", status=PENDING)), ["ready", "waiting", "waiting", "waiting"])

    def test_every_stage_of_a_normal_run(self):
        orch = Orchestrator(toolkit=FakeToolkit(), supervisor=FakeSupervisor())
        state = orch.start_run(["a.csv"], "q")
        self.assertEqual(self.statuses(state), ["awaiting", "waiting", "waiting", "waiting"])
        orch.approve(state)
        self.assertEqual(self.statuses(state), ["done", "ready", "waiting", "waiting"])
        orch.run_next_phase(state)
        self.assertEqual(self.statuses(state), ["done", "awaiting", "waiting", "waiting"])
        orch.approve(state); orch.run_next_phase(state); orch.approve(state); orch.run_next_phase(state)
        self.assertEqual(state.status, COMPLETE)
        self.assertEqual(self.statuses(state), ["done", "done", "done", "done"])

    def test_failure_cancel_and_running(self):
        failing = FakeToolkit()
        failing.failures["phase_1"] = "bad csv"
        state = Orchestrator(toolkit=failing, supervisor=FakeSupervisor()).start_run(["a.csv"], "q")
        cards = ui.phase_view(state)
        self.assertEqual((cards[0]["status"], cards[0]["icon"]), ("failed", "❌"))

        orch = Orchestrator(toolkit=FakeToolkit(), supervisor=FakeSupervisor())
        cancelled = orch.start_run(["a.csv"], "q")
        orch.cancel(cancelled, "no")
        self.assertEqual(self.statuses(cancelled), ["stopped", "waiting", "waiting", "waiting"])

        running = PipelineState(run_id="r", status=RUNNING, current_phase="phase_2", profile_path="p", sttm_bronze_path="b")
        self.assertEqual(self.statuses(running), ["done", "running", "waiting", "waiting"])

    def test_retry_and_reopen_availability(self):
        failing = FakeToolkit()
        orch = Orchestrator(toolkit=failing, supervisor=FakeSupervisor())
        state = orch.start_run(["a.csv"], "q")
        self.assertFalse(ui.can_retry(state) or ui.can_reopen(state))
        orch.approve(state)
        failing.failures["phase_2"] = "boom"
        orch.run_next_phase(state)
        self.assertTrue(ui.can_retry(state) and ui.can_reopen(state))
        orch.cancel(state, "")
        self.assertFalse(ui.can_retry(state) or ui.can_reopen(state))


class EditorTests(unittest.TestCase):
    rows = sttm_rows("silver", description="Order value")

    def test_round_trip_is_lossless_and_all_text(self):
        frame = ui.editor_frame(self.rows)
        self.assertEqual(list(frame.columns), list(STTM_COLUMNS))
        self.assertTrue(all(frame[c].dtype == object or str(frame[c].dtype) == "string" for c in frame.columns))
        back = ui.rows_from_editor(frame, "silver")
        self.assertEqual(ui.count_edits(self.rows, back), 0)

    def test_blank_rows_are_dropped_and_new_rows_get_the_layer(self):
        frame = ui.editor_frame(self.rows)
        added = pd.DataFrame([
            {"source_table": " orders ", "source_col": "qty", "target_col": "qty", "data_type": "integer"},
            {c: None for c in STTM_COLUMNS},                       # an empty row from the editor
            {c: "  " for c in STTM_COLUMNS},
        ])
        rows = ui.rows_from_editor(pd.concat([frame, added], ignore_index=True), "silver")
        self.assertEqual(len(rows), 2)
        self.assertEqual((rows[1]["layer"], rows[1]["source_table"], rows[1]["nullable"]), ("silver", "orders", ""))

    def test_edit_counting(self):
        edited = [dict(self.rows[0], description="Reviewed")]
        self.assertEqual(ui.count_edits(self.rows, edited), 2)                      # one removed + one added == one changed row
        self.assertEqual(ui.count_edits(self.rows, []), 1)
        self.assertEqual(ui.count_edits([], self.rows), 1)

    def test_dropdown_options(self):
        options = ui.column_options()
        self.assertIn("integer", options["data_type"])
        self.assertEqual(options["nullable"], ["true", "false"])
        self.assertIn("", options["aggregation"])
        self.assertIn("layer", ui.DISABLED_COLUMNS["gold"])
        self.assertIn("aggregation", ui.DISABLED_COLUMNS["silver"])


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)

    def test_valid_uploads_are_saved_under_their_own_names(self):
        paths = ui.save_uploads([upload("orders.csv"), upload("My Customers (2024).csv")], self.root)
        self.assertEqual([Path(p).name for p in paths], ["orders.csv", "My Customers (2024).csv"])
        self.assertEqual(len({Path(p).parent for p in paths}), 1)
        self.assertTrue(all(Path(p).read_bytes() == b"a,b\n1,2\n" for p in paths))
        again = ui.save_uploads([upload("orders.csv")], self.root)
        self.assertNotEqual(Path(again[0]).parent, Path(paths[0]).parent)            # batches never overwrite each other

    def test_problems_are_reported_before_anything_is_written(self):
        cases = {
            "no files": ([], "at least one"),
            "not csv": ([upload("data.xlsx")], "not a .csv"),
            "duplicate names": ([upload("a.csv"), upload("a.csv")], "share the name"),
            "same table name": ([upload("a b.csv"), upload("a_b.csv")], "same table name"),
            "empty file": ([upload("e.csv", b"  \n")], "is empty"),
        }
        for label, (uploads, expected) in cases.items():
            with self.assertRaisesRegex(ValueError, expected, msg=label):
                ui.save_uploads(uploads, self.root)
        self.assertEqual(list(self.root.iterdir()), [])                              # nothing left behind by rejected uploads
        with self.assertRaisesRegex(ValueError, "is empty"):
            ui.save_uploads([upload("ok.csv"), upload("e.csv", b"")], self.root)       # a good file before a bad one
        self.assertEqual(list(self.root.iterdir()), [])

    def test_path_components_in_names_are_ignored(self):
        paths = ui.save_uploads([upload("../../evil.csv")], self.root)
        self.assertEqual(Path(paths[0]).name, "evil.csv")
        self.assertTrue(Path(paths[0]).resolve().is_relative_to(self.root.resolve()))

    def test_start_new_run_requires_a_question_and_starts_phase_one(self):
        orch = Orchestrator(toolkit=FakeToolkit(), supervisor=FakeSupervisor())
        with self.assertRaisesRegex(ValueError, "business question"):
            ui.start_new_run(orch, [upload("a.csv")], "   ")
        state = ui.start_new_run(orch, [upload("orders.csv")], "Which product sold most?")
        self.assertEqual((state.status, state.current_phase), (AWAITING_APPROVAL, "phase_1"))
        self.assertTrue(state.csv_paths[0].endswith("orders.csv"))


class PreviewAndTableTests(unittest.TestCase):
    def test_source_previews_per_layer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "orders.csv").write_text("a,b\n1,x\n2,y\n", encoding="utf-8")
            pd.DataFrame({"a": [1, 2, 3]}).to_parquet(root / "bronze.parquet", index=False)
            pd.DataFrame({"k": [1]}).to_parquet(root / "silver.parquet", index=False)
            state = PipelineState(run_id="r", csv_paths=[str(root / "orders.csv")],
                                  bronze_output_paths=[str(root / "bronze.parquet")],
                                  silver_output_paths=[str(root / "silver.parquet")])
            self.assertEqual([(n, len(f)) for n, f in ui.source_previews(state, "bronze")], [("orders.csv", 2)])
            self.assertEqual([(n, len(f)) for n, f in ui.source_previews(state, "silver")], [("bronze.parquet", 3)])
            self.assertEqual([(n, len(f)) for n, f in ui.source_previews(state, "gold")], [("silver.parquet", 1)])
            self.assertIsNone(ui.load_profile(state))

    def test_previews_read_only_the_first_rows_never_whole_files(self):
        import pandas as real_pandas
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            big = real_pandas.DataFrame({"a": range(5000), "b": ["x"] * 5000})
            big.to_csv(root / "big.csv", index=False)
            big.to_parquet(root / "big.parquet", index=False)
            state = PipelineState(run_id="r", csv_paths=[str(root / "big.csv")], bronze_output_paths=[str(root / "big.parquet")])
            with patch.object(real_pandas, "read_parquet", side_effect=AssertionError("read the whole parquet file")):
                parquet_preview = ui.source_previews(state, "silver")[0][1]
            self.assertEqual(len(parquet_preview), ui.PREVIEW_ROWS)
            seen = {}
            original = real_pandas.read_csv
            with patch.object(real_pandas, "read_csv", side_effect=lambda *a, **k: seen.update(k) or original(*a, **k)):
                csv_preview = ui.source_previews(state, "bronze")[0][1]
            self.assertEqual((len(csv_preview), seen["nrows"]), (ui.PREVIEW_ROWS, ui.PREVIEW_ROWS))
            self.assertEqual(csv_preview["a"].iloc[0], "0")
            (root / "latin.csv").write_bytes("name\ncaf\u00e9\n".encode("latin-1"))
            self.assertEqual(ui.head_of_csv(root / "latin.csv")["name"].iloc[0], "caf\u00e9")

    def test_audit_and_trace_tables(self):
        events = [{"timestamp": "2026-10-05T10:11:12+00:00", "run_id": "r", "agent": "orchestrator", "action": "phase_1_started",
                   "phase": "phase_1", "goal": "x" * 300, "tools": ["a", "b"]}]
        table = ui.audit_table(events)
        self.assertEqual(list(table.columns), ["Time (UTC)", "Phase", "Agent", "Event", "Details"])
        self.assertEqual((table.iloc[0]["Time (UTC)"], table.iloc[0]["Event"]), ("10:11:12", "phase 1 started"))
        self.assertLess(len(table.iloc[0]["Details"]), 300)
        self.assertEqual(len(ui.audit_table([])), 0)
        traces = ui.trace_table([{"agent": "sttm", "status": "failed", "duration_seconds": 1.5, "tools": ["x", None], "error": "boom"}])
        self.assertEqual(traces.iloc[0].to_dict(), {"Agent": "sttm", "Status": "failed", "Seconds": 1.5, "Tools called": "x", "Error": "boom"})


if __name__ == "__main__":
    unittest.main()
