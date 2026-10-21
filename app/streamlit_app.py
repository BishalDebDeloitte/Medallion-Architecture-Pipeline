"""Streamlit interface for the approval-gated medallion pipeline.

Upload CSVs and ask a business question; the pipeline profiles the data and proposes a mapping
(STTM) for each layer. After every mapping the run pauses: review and edit it, then approve,
regenerate with feedback, or reject. After the last approval the report appears here.
All state lives in the Orchestrator and is saved to disk, so a refresh does not lose a run.
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from agents.orchestrator import PHASE_LABELS, Orchestrator
from agents.sttm_generator import layer_rules
from app import ui_logic as ui
from core import run_store
from core.state import AWAITING_APPROVAL, COMPLETE, FAILED, PENDING, PipelineState

logger = logging.getLogger("medallion.app")

st.set_page_config(page_title="Medallion pipeline", layout="wide")


# --------------------------------------------------------------------------- plumbing
def get_orchestrator() -> Orchestrator:
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = Orchestrator()
    return st.session_state.orchestrator


@st.cache_data(show_spinner=False)
def _read_text(path: str, modified: float) -> str:
    return Path(path).read_text(encoding="utf-8")


def read_text(path: str) -> str:
    """Read a (possibly large) report file, cached until it changes."""
    return _read_text(path, Path(path).stat().st_mtime)


def act(label: str, action) -> None:
    """Run a pipeline action with a spinner. Failures are shown, never raised; then rerun the page."""
    try:
        with st.spinner(label):
            action()
    except ValueError as error:                 # invalid reviewer edits: keep the editor as it is
        st.session_state.review_error = str(error)
    except Exception as error:
        logger.exception("Action failed: %s", label)
        st.session_state.notice = f"{type(error).__name__}: {error}"
    st.rerun()


def set_state(state: PipelineState) -> None:
    st.session_state.state = state


# --------------------------------------------------------------------------- sidebar
def sidebar(orch: Orchestrator, state: PipelineState | None) -> None:
    with st.sidebar:
        st.header("New run")
        problem = orch.preflight()
        if problem:
            st.warning(problem)
        with st.form("new_run"):
            uploads = st.file_uploader("CSV files", type=["csv"], accept_multiple_files=True, key="uploads")
            intent = st.text_area(
                "Business question", key="intent", height=110,
                placeholder="e.g. Which customers generate the most revenue, and how is it trending?",
            )
            submitted = st.form_submit_button("Start pipeline", type="primary", width="stretch", disabled=bool(problem))
        st.caption("Each file becomes one table named after the file. Files are read in full, so keep them to a few hundred MB.")
        if submitted:
            try:
                with st.spinner("Phase 1: profiling the data and designing the Bronze mapping…"):
                    set_state(ui.start_new_run(orch, uploads or [], intent))
                st.session_state.pop("review_error", None)
            except ValueError as error:
                st.error(str(error))
            except Exception as error:
                logger.exception("Could not start the run")
                st.error(f"Could not start the run: {error}")
            else:
                st.rerun()

        runs = run_store.list_runs()
        if runs:
            st.divider()
            st.subheader("Resume a run")
            labels = {r["run_id"]: f"{r['status'].replace('_', ' ')} · {r['business_intent'][:38]}" for r in runs}
            choice = st.selectbox("Saved runs", list(labels), format_func=labels.get, index=None,
                                  placeholder="Choose a saved run", key="resume_choice")
            if st.button("Open run", key="open_run", disabled=choice is None, width="stretch"):
                loaded = orch.resume(choice)
                if loaded is None:
                    st.error("That run could not be loaded.")
                else:
                    set_state(loaded)
                    st.rerun()
        if state is not None:
            st.divider()
            st.caption("CURRENT RUN")
            st.code(state.run_id, language=None)


# --------------------------------------------------------------------------- tracker
def tracker(state: PipelineState) -> None:
    columns = st.columns(4)
    for column, card in zip(columns, ui.phase_view(state)):
        with column.container(border=True):
            st.markdown(f"**{card['icon']}  Phase {card['number']}**")
            st.caption(card["title"])
            st.markdown(f"{card['detail']}")


# --------------------------------------------------------------------------- review (human in the loop)
def mapping_editor(state: PipelineState, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    layer = state.pending_approval_layer
    revision = st.session_state.setdefault(f"revision-{state.run_id}-{layer}", 0)
    options = ui.column_options()
    config = {
        "data_type": st.column_config.SelectboxColumn("data_type", options=options["data_type"], required=True),
        "nullable": st.column_config.SelectboxColumn("nullable", options=options["nullable"], required=True),
        "aggregation": st.column_config.SelectboxColumn("aggregation", options=options["aggregation"]),
        "transformation_logic": st.column_config.TextColumn(
            "transformation_logic", help="Operations separated by |, for example trim|lowercase. See 'Allowed transformations'."),
        "description": st.column_config.TextColumn("description", width="medium"),
    }
    edited = st.data_editor(
        ui.editor_frame(rows), key=f"editor-{state.run_id}-{layer}-{revision}", num_rows="dynamic",
        hide_index=True, width="stretch", disabled=list(ui.DISABLED_COLUMNS[layer]), column_config=config,
    )
    return ui.rows_from_editor(edited, layer)


def review_panel(orch: Orchestrator, state: PipelineState) -> None:
    layer = state.pending_approval_layer
    title = ui.LAYER_TITLES[layer]
    st.subheader(f"{title} mapping review")
    st.caption(f"Approval {['bronze', 'silver', 'gold'].index(layer) + 1} of 3. Nothing runs on the next layer until you approve this mapping.")
    st.caption(ui.LAYER_HELP[layer] + " Edit any cell, add or delete rows, then approve.")
    rows = orch.pending_rows(state)
    tab_names = ["Mapping", "Source data", "Data profile" if layer == "bronze" else "Run summary"]
    if layer != "bronze":
        tab_names.append("What the last step did")
    tabs = st.tabs(tab_names)
    mapping_tab, data_tab, profile_tab = tabs[:3]

    with mapping_tab:
        edited_rows = mapping_editor(state, rows)
        metrics = st.columns(3)
        metrics[0].metric("Columns mapped", len(edited_rows))
        metrics[1].metric("Target tables", len({r["target_table"] for r in edited_rows if r["target_table"]}))
        metrics[2].metric("Edited rows", ui.count_edits(rows, edited_rows))
        with st.expander("Allowed transformations"):
            for name, hint in layer_rules(layer)["allowed_operations"].items():
                st.markdown(f"- `{name}`: {hint}")

    with data_tab:
        try:
            previews = ui.source_previews(state, layer)
        except Exception as error:
            st.warning(f"Source data could not be previewed: {error}")
            previews = []
        for name, frame in previews:
            with st.expander(f"{name}  ({len(frame)} rows shown)", expanded=len(previews) == 1):
                st.dataframe(frame, width="stretch", hide_index=True)

    with profile_tab:
        profile = ui.load_profile(state)
        if layer == "bronze" and profile:
            for dataset in profile["datasets"]:
                st.markdown(f"**{dataset['table']}**: {dataset['row_count']:,} rows, {dataset['column_count']} columns")
                notes = dataset["quality_notes"]
                if notes:
                    with st.expander(f"{len(notes)} data-quality note(s) for {dataset['table']}"):
                        for note in notes:
                            st.markdown(f"- {note}")
            links = ui.join_table(profile)
            if len(links):
                st.markdown("**Links found between tables**")
                st.dataframe(links, width="stretch", hide_index=True)
            hints = ui.hint_table(profile)
            if len(hints):
                st.markdown("**Transformations the data calls for** (the agent applies these where they help)")
                st.dataframe(hints, width="stretch", hide_index=True)
        else:
            st.write(f"Business question: {state.business_intent}")
            st.write(f"Bronze files: {len(state.bronze_output_paths)} · Silver files: {len(state.silver_output_paths)}")

    if layer != "bronze":
        with tabs[3]:
            report = ui.load_layer_report(ui.previous_step_report_path(state, layer))
            if report:
                st.caption(f"What the {ui.LAYER_TITLES[{'silver': 'bronze', 'gold': 'silver'}[layer]]} step did to your data:")
                st.dataframe(ui.layer_report_table(report), width="stretch", hide_index=True)
            else:
                st.caption("No results report is available for the previous step.")

    sensitive = ui.sensitive_in_mapping(edited_rows, ui.load_profile(state))
    if sensitive:
        st.warning(
            "This mapping keeps sensitive column(s): " + ", ".join(sensitive)
            + ". Their values are hidden from the AI, but they will flow into the next layers. "
            "Delete those rows unless the question needs them.", icon="🔒",
        )
    if message := st.session_state.pop("review_error", None):
        st.error(f"The mapping was not approved. Fix these and try again:\n\n{message}")

    st.divider()
    st.subheader("Decision")
    approve_col, regenerate_col, reject_col = st.columns(3)
    with approve_col:
        if st.button("Approve & continue", type="primary", key="approve", width="stretch"):
            def approve_and_continue() -> None:
                orch.approve(state, edited_rows)
                orch.run_next_phase(state)

            act(f"Approved. Running next: {PHASE_LABELS.get(orch.next_phase(state) or '', 'next phase')}…", approve_and_continue)
    with regenerate_col:
        with st.popover("Regenerate…", width="stretch"):
            feedback = st.text_area("What should change?", key="feedback", height=100,
                                    placeholder="e.g. Keep only the columns needed for revenue; treat dates day-first.")
            if st.button("Regenerate mapping", key="regenerate", width="stretch"):
                def regenerate() -> None:
                    orch.regenerate(state, feedback)
                    st.session_state[f"revision-{state.run_id}-{layer}"] += 1

                act("Asking the STTM agent for a new mapping…", regenerate)
    with reject_col:
        with st.popover("Reject…", width="stretch"):
            reason = st.text_input("Reason (optional)", key="reject_reason")
            confirmed = st.checkbox("Stop this run", key="confirm_reject")
            if st.button("Reject and stop run", key="reject", disabled=not confirmed, width="stretch"):
                act("Stopping the run…", lambda: orch.cancel(state, reason))
    if state.error:
        st.warning(f"The last regeneration failed, so the previous mapping is still shown: {state.error}")


# --------------------------------------------------------------------------- other states
def failure_panel(orch: Orchestrator, state: PipelineState) -> None:
    if state.cancelled:
        st.warning(state.error)
        st.info("This run was stopped. Start a new run from the sidebar.")
        return
    st.error(f"**{PHASE_LABELS.get(state.current_phase, 'The pipeline')} failed.**\n\n{state.error}")
    retry_col, reopen_col, _ = st.columns([1, 1, 2])
    with retry_col:
        if st.button("Retry phase", type="primary", key="retry", width="stretch"):
            act("Retrying…", lambda: orch.retry(state))
    if ui.can_reopen(state):
        with reopen_col:
            if st.button("Edit approved mapping", key="reopen", width="stretch",
                         help="Use this when the error says the data conflicts with the approved mapping."):
                act("Reopening the mapping…", lambda: orch.reopen_mapping(state))
        st.caption("A plain retry fails the same way if the data conflicts with the approved rules; edit the mapping instead.")


def ready_panel(orch: Orchestrator, state: PipelineState) -> None:
    phase = orch.next_phase(state)
    if phase is None:
        return
    st.info(f"Ready: {PHASE_LABELS[phase]}.")
    if st.button("Run next phase", type="primary", key="run_next"):
        act(f"{PHASE_LABELS[phase]}…", lambda: orch.run_next_phase(state))


def report_panel(state: PipelineState) -> None:
    st.success("Pipeline complete. The report below answers your question.")
    report_tab, gold_tab, files_tab = st.tabs(["Report", "Gold tables", "Downloads"])
    with report_tab:
        try:
            components.html(read_text(state.report_path), height=900, scrolling=True)
        except OSError as error:
            st.error(f"The report file could not be read: {error}")
    with gold_tab:
        gold_report = ui.load_layer_report(state.gold_report_path)
        if gold_report:
            st.caption("How each Gold table was built")
            st.dataframe(ui.layer_report_table(gold_report), width="stretch", hide_index=True)
        for path in state.gold_output_paths:
            st.markdown(f"**{Path(path).stem}**")
            st.dataframe(pd.read_parquet(path), width="stretch", hide_index=True)
    with files_tab:
        if state.report_path and Path(state.report_path).is_file():
            st.download_button("Download HTML report", data=read_text(state.report_path),
                               file_name=Path(state.report_path).name, mime="text/html", key="download_html")
        if state.report_json_path and Path(state.report_json_path).is_file():
            st.download_button("Download report data (JSON)", data=read_text(state.report_json_path),
                               file_name=Path(state.report_json_path).name, mime="application/json", key="download_json")
        st.caption("Approved mappings and layer files")
        for label, value in (
            ("Bronze mapping", state.sttm_bronze_path), ("Silver mapping", state.sttm_silver_path),
            ("Gold mapping", state.sttm_gold_path), *[(f"Gold table", p) for p in state.gold_output_paths],
        ):
            if value:
                st.code(f"{label}: {value}", language=None)


def diagnostics(orch: Orchestrator, state: PipelineState) -> None:
    st.divider()
    with st.expander("Agent traces"):
        summaries = orch.trace_summaries(state)
        if summaries:
            st.dataframe(ui.trace_table(summaries), width="stretch", hide_index=True)
            for item in summaries:
                if item["plan"] or item["error"]:
                    st.markdown(f"**{item['agent']}** ({item['status']})")
                    if item["plan"]:
                        st.caption(str(item["plan"])[:600])
                    if item["error"]:
                        st.error(str(item["error"])[:600])
        else:
            st.caption("No agent has run yet.")
    with st.expander("Audit log"):
        events = orch.audit_logs(state)
        if events:
            st.dataframe(ui.audit_table(events), width="stretch", hide_index=True)
        else:
            st.caption("No events yet.")


# --------------------------------------------------------------------------- page
orch = get_orchestrator()
state: PipelineState | None = st.session_state.get("state")
if state is None and (linked := st.query_params.get("run")):     # a refresh or a bookmarked run link
    if (state := orch.resume(linked)) is not None:
        set_state(state)
    else:
        st.session_state.notice = "That run link could not be opened: the run was not found."
if state is not None and st.query_params.get("run") != state.run_id:
    st.query_params["run"] = state.run_id                         # keep the URL pointing at the current run
sidebar(orch, state)

st.title("Medallion pipeline")
if notice := st.session_state.pop("notice", None):
    st.error(notice)
if state is None:
    st.write("Upload one or more CSV files and ask a business question. The pipeline will:")
    st.markdown(
        "1. **Profile** the data and propose a Bronze mapping, then wait for your review\n"
        "2. **Ingest** Bronze and propose a Silver mapping, then wait\n"
        "3. **Cleanse** Silver and propose a Gold mapping, then wait\n"
        "4. **Build** the Gold tables and write an executive report with charts"
    )
    st.caption("Nothing runs on a layer until you have approved its mapping.")
    st.stop()

st.markdown(f"#### {state.business_intent}")
tracker(state)

if state.status == AWAITING_APPROVAL:
    review_panel(orch, state)
elif state.status == FAILED:
    failure_panel(orch, state)
elif state.status == COMPLETE:
    report_panel(state)
elif state.status == PENDING:
    ready_panel(orch, state)
diagnostics(orch, state)
