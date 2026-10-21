"""Streamlit app tests (Step 8): drive the real app script with Streamlit's AppTest harness.

The Orchestrator in these tests uses fake agent tools (see tests/test_phase_loop.py), so the UI flows run
offline in milliseconds. The agents themselves are covered by their own tests and tests/test_end_to_end.py.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from agents.orchestrator import Orchestrator
from core import llm
from core.state import AWAITING_APPROVAL, COMPLETE, FAILED
from tests.test_phase_loop import FakeSupervisor, FakeToolkit

APP_PATH = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


def make_orchestrator(toolkit=None):
    return Orchestrator(toolkit=toolkit or FakeToolkit(), supervisor=FakeSupervisor())


def open_app(orchestrator, state=None):
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.session_state["orchestrator"] = orchestrator
    if state is not None:
        app.session_state["state"] = state
    return app.run()


def has_button(app, key):
    return any(button.key == key for button in app.button)


def texts(elements):
    return " ".join(str(getattr(element, "value", "")) for element in elements)


class EmptyPageTests(unittest.TestCase):
    def test_welcome_page_with_upload_and_question(self):
        app = open_app(make_orchestrator())
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("file_uploader")), 1)
        self.assertEqual(len(app.text_area), 1)
        self.assertIn("Medallion pipeline", texts(app.title))
        self.assertIn("Nothing runs on a layer until you have approved its mapping", texts(app.caption))
        self.assertFalse(app.session_state["orchestrator"].preflight())

    def test_missing_api_key_is_explained_and_blocks_starting(self):
        with patch.object(llm.config, "LLM_PROVIDER", "anthropic"), patch.object(llm.config, "ANTHROPIC_API_KEY", None):
            app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertIn("ANTHROPIC_API_KEY is not set", texts(app.warning))
        start = [b for b in app.button if b.label == "Start pipeline"]
        self.assertTrue(start and start[0].disabled)


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.toolkit = FakeToolkit()
        self.orch = make_orchestrator(self.toolkit)
        self.state = self.orch.start_run(["orders.csv"], "Which product sold most?")

    def test_bronze_review_shows_tracker_mapping_and_decisions(self):
        app = open_app(self.orch, self.state)
        self.assertFalse(app.exception)
        self.assertIn("Bronze mapping review", texts(app.subheader))
        page = texts(app.markdown)
        self.assertIn("Phase 1", page)
        self.assertIn("Review the Bronze mapping", page)
        self.assertIn("Which product sold most?", page)
        for key in ("approve", "regenerate", "reject"):
            self.assertTrue(has_button(app, key), key)
        self.assertTrue(app.button(key="reject").disabled)                    # needs the confirmation checkbox
        mapping = next(frame.value for frame in app.dataframe if "target_col" in frame.value.columns)   # the editable table
        self.assertEqual(mapping.iloc[0]["target_col"], "amount")
        self.assertEqual(list(mapping.columns)[:3], ["layer", "target_table", "source_table"])

    def test_approving_moves_on_to_the_next_review(self):
        app = open_app(self.orch, self.state)
        app.button(key="approve").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(self.state.current_phase, "phase_2")
        self.assertTrue(self.state.sttm_bronze_path.endswith("_approved.csv"))
        self.assertIn("Silver mapping review", texts(app.subheader))
        self.assertIn("Bronze mapping approved", texts(app.markdown))

    def test_regenerating_passes_the_feedback_and_keeps_the_review_open(self):
        app = open_app(self.orch, self.state)
        app.text_area(key="feedback").input("Keep only the amount column").run()
        app.button(key="regenerate").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(self.toolkit.regenerations, 1)
        self.assertIn("Keep only the amount column", self.state.task_description)
        self.assertEqual(self.state.status, AWAITING_APPROVAL)
        self.assertIn("Bronze mapping review", texts(app.subheader))

    def test_rejecting_requires_confirmation_and_stops_the_run(self):
        app = open_app(self.orch, self.state)
        app.checkbox(key="confirm_reject").check().run()
        app.text_input(key="reject_reason").input("Wrong columns").run()
        self.assertFalse(app.button(key="reject").disabled)
        app.button(key="reject").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(self.state.cancelled)
        self.assertIn("Wrong columns", self.state.error)
        self.assertIn("Start a new run", texts(app.info))
        self.assertFalse(has_button(app, "retry"))

    def test_a_validation_error_is_shown_without_losing_the_review(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=30)
        app.session_state["orchestrator"], app.session_state["state"] = self.orch, self.state
        app.session_state["review_error"] = "STTM row 1: unsupported transformation 'explode'"
        app.run()
        self.assertIn("explode", texts(app.error))
        self.assertEqual(self.state.status, AWAITING_APPROVAL)
        self.assertTrue(has_button(app, "approve"))

    def test_a_rejected_edit_surfaces_the_validation_message(self):
        self.toolkit.available = {"orders": {"order_id"}}                       # the generated rows reference 'amount'
        app = open_app(self.orch, self.state)
        app.button(key="approve").click().run()
        self.assertFalse(app.exception)
        self.assertIn("does not exist in 'orders'", texts(app.error))
        self.assertEqual((self.state.status, self.state.sttm_bronze_path), (AWAITING_APPROVAL, ""))


class FailureTests(unittest.TestCase):
    def test_failure_shows_the_message_and_retry_recovers(self):
        toolkit = FakeToolkit()
        toolkit.failures["phase_1"] = "CSV file contains duplicate column names: orders.csv"
        orch = make_orchestrator(toolkit)
        state = orch.start_run(["orders.csv"], "q")
        self.assertEqual(state.status, FAILED)
        app = open_app(orch, state)
        self.assertFalse(app.exception)
        self.assertIn("duplicate column names", texts(app.error))
        self.assertTrue(has_button(app, "retry"))
        self.assertFalse(has_button(app, "reopen"))                          # no approved mapping to reopen yet
        del toolkit.failures["phase_1"]
        app.button(key="retry").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(state.status, AWAITING_APPROVAL)
        self.assertIn("Bronze mapping review", texts(app.subheader))

    def test_an_approved_mapping_that_fails_offers_to_reopen_it(self):
        toolkit = FakeToolkit()
        orch = make_orchestrator(toolkit)
        state = orch.start_run(["orders.csv"], "q")
        orch.approve(state)
        toolkit.failures["phase_2"] = "orders.amount has 2 missing value(s) but is nullable=false"
        orch.run_next_phase(state)
        app = open_app(orch, state)
        self.assertIn("nullable=false", texts(app.error))
        self.assertTrue(has_button(app, "reopen"))
        app.button(key="reopen").click().run()
        self.assertFalse(app.exception)
        self.assertEqual((state.status, state.pending_approval_layer), (AWAITING_APPROVAL, "bronze"))
        self.assertIn("Bronze mapping review", texts(app.subheader))


class CompleteAndResumeTests(unittest.TestCase):
    def complete_run(self, directory):
        orch = make_orchestrator()
        state = orch.start_run(["orders.csv"], "Which customers buy most?")
        for _ in range(3):
            orch.approve(state)
            orch.run_next_phase(state)
        self.assertEqual(state.status, COMPLETE)
        root = Path(directory)
        (root / "report.html").write_text("<html><body><h1>The report</h1></body></html>", encoding="utf-8")
        (root / "report.json").write_text("{}", encoding="utf-8")
        pd.DataFrame({"customer": ["a"], "total": [1.0]}).to_parquet(root / "gold_t.parquet", index=False)
        state.report_path, state.report_json_path = str(root / "report.html"), str(root / "report.json")
        state.gold_output_paths = [str(root / "gold_t.parquet")]
        return orch, state

    def test_finished_run_shows_the_report_tables_and_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            orch, state = self.complete_run(directory)
            app = open_app(orch, state)
            self.assertFalse(app.exception)
            self.assertIn("Pipeline complete", texts(app.success))
            self.assertEqual({b.key for b in app.get("download_button")}, {"download_html", "download_json"})
            self.assertIn("Phase 4", texts(app.markdown))
            self.assertEqual([tab.label for tab in app.tabs], ["Report", "Gold tables", "Downloads"])

    def test_the_url_always_points_at_the_current_run_and_survives_a_refresh(self):
        orch = make_orchestrator()
        state = orch.start_run(["orders.csv"], "Linkable question")
        app = open_app(orch, state)
        self.assertEqual(app.query_params["run"], state.run_id)                       # written for the current run
        refreshed = AppTest.from_file(str(APP_PATH), default_timeout=30)               # a brand-new session
        refreshed.session_state["orchestrator"] = make_orchestrator()
        refreshed.query_params["run"] = state.run_id
        refreshed.run()
        self.assertFalse(refreshed.exception)
        self.assertEqual(refreshed.session_state["state"].run_id, state.run_id)
        self.assertIn("Bronze mapping review", texts(refreshed.subheader))

    def test_an_unknown_or_malicious_run_link_is_reported_not_crashed_on(self):
        for bad in ("no-such-run", "../../etc/passwd", "x" * 300):
            app = AppTest.from_file(str(APP_PATH), default_timeout=30)
            app.session_state["orchestrator"] = make_orchestrator()
            app.query_params["run"] = bad
            app.run()
            self.assertFalse(app.exception, bad)
            self.assertIn("could not be opened", texts(app.error), bad)
            self.assertEqual(len(app.get("file_uploader")), 1)                         # still the welcome page

    def test_a_saved_run_can_be_resumed_from_the_sidebar(self):
        orch = make_orchestrator()
        state = orch.start_run(["orders.csv"], "Resumable question")
        app = open_app(make_orchestrator())                                          # a fresh session: nothing selected yet
        self.assertEqual(len(app.get("file_uploader")), 1)
        app.selectbox(key="resume_choice").select(state.run_id).run()
        app.button(key="open_run").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["state"].run_id, state.run_id)
        self.assertIn("Bronze mapping review", texts(app.subheader))


if __name__ == "__main__":
    unittest.main()
