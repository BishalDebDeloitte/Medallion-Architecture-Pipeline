"""Pure UI logic for the Streamlit app (no Streamlit calls, so it can be unit tested).

Covers: the four-phase tracker, converting STTM rows to and from the editable table,
validating and saving uploads, and shaping audit/trace data for display.
"""

import json
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from agents.orchestrator import PHASE_LABELS, PHASES, REOPEN_LAYER, Orchestrator
from core.csv_io import ENCODINGS, table_names
from core.paths import LANDING_DIR
from core.sensitivity import sensitivity_by_name
from core.state import AWAITING_APPROVAL, COMPLETE, FAILED, RUNNING, PipelineState
from core.sttm_schema import AGGREGATIONS, DATA_TYPES, STTM_COLUMNS

PHASE_TITLES = {
    "phase_1": "Profile & Bronze mapping",
    "phase_2": "Bronze & Silver mapping",
    "phase_3": "Silver & Gold mapping",
    "phase_4": "Gold & report",
}
LAYER_TITLES = {"bronze": "Bronze", "silver": "Silver", "gold": "Gold"}
LAYER_HELP = {
    "bronze": "Raw ingestion: which source columns are kept, and their types. Dates stay as text here.",
    "silver": "Cleansing: nulls (fillna/dropna), duplicates (dedupe), dates (to_date) and text normalisation.",
    "gold": "Business tables: joins on join_key, group-by columns (blank aggregation) and aggregated measures.",
}
PREVIEW_ROWS = 100
DISABLED_COLUMNS = {"bronze": ("layer", "aggregation", "join_key"), "silver": ("layer", "aggregation", "join_key"), "gold": ("layer",)}


# --------------------------------------------------------------------------- tracker
def phase_view(state: PipelineState) -> list[dict[str, str]]:
    """The four tracker cards: number, title, status, icon and a short detail line."""
    ran = {
        "phase_1": bool(state.profile_path),
        "phase_2": bool(state.bronze_output_paths),
        "phase_3": bool(state.silver_output_paths),
        "phase_4": bool(state.report_path),
    }
    proposes = {"phase_1": "bronze", "phase_2": "silver", "phase_3": "gold"}
    approved = {"phase_1": state.sttm_bronze_path, "phase_2": state.sttm_silver_path, "phase_3": state.sttm_gold_path}
    next_phase = Orchestrator.next_phase(state) if not state.cancelled else None
    cards = []
    for number, phase in enumerate(PHASES, start=1):
        layer = proposes.get(phase)
        if state.status == FAILED and state.current_phase == phase and not ran[phase]:
            status, icon, detail = "failed", "❌", "Failed - see the message below"
        elif layer and ran[phase] and state.status == AWAITING_APPROVAL and state.pending_approval_layer == layer:
            status, icon, detail = "awaiting", "⏸", f"Review the {LAYER_TITLES[layer]} mapping"
        elif ran[phase] and (phase == "phase_4" or approved.get(phase)):
            status, icon, detail = "done", "✅", "Done" if phase == "phase_4" else f"{LAYER_TITLES[layer]} mapping approved"
        elif ran[phase]:
            status, icon, detail = "stopped", "⏹", "Stopped before approval"
        elif state.status == RUNNING and state.current_phase == phase:
            status, icon, detail = "running", "⏳", "Running"
        elif next_phase == phase:
            status, icon, detail = "ready", "▶", "Ready to run"
        else:
            status, icon, detail = "waiting", "○", "Waiting"
        cards.append({"phase": phase, "number": str(number), "title": PHASE_TITLES[phase], "status": status,
                      "icon": icon, "detail": detail, "label": PHASE_LABELS[phase]})
    return cards


def can_reopen(state: PipelineState) -> bool:
    """True if a failed run has an approved mapping that can be put back in review for editing."""
    layer = REOPEN_LAYER.get(state.current_phase)
    return bool(
        state.status == FAILED and not state.cancelled and layer and getattr(state, f"sttm_{layer}_path", "")
    )


def can_retry(state: PipelineState) -> bool:
    """True if the failed phase can simply be run again."""
    return state.status == FAILED and not state.cancelled


# --------------------------------------------------------------------------- the editable mapping
def editor_frame(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """STTM rows as a text table (every cell a string, blanks as empty strings)."""
    frame = pd.DataFrame(list(rows), columns=list(STTM_COLUMNS))
    return frame.fillna("").astype(str)


def rows_from_editor(frame: pd.DataFrame, layer: str) -> list[dict[str, str]]:
    """Convert the edited table back to STTM rows: blank rows dropped, layer filled in, cells as text."""
    cleaned = frame.reindex(columns=list(STTM_COLUMNS)).fillna("").astype(str)
    rows = []
    for record in cleaned.to_dict("records"):
        record = {key: value.strip() for key, value in record.items()}
        if not any(value for key, value in record.items() if key != "layer"):
            continue
        record["layer"] = layer
        rows.append(record)
    return rows


def count_edits(original: Sequence[dict[str, Any]], edited: Sequence[dict[str, Any]]) -> int:
    """How many rows differ between the generated and the edited mapping (added, removed or changed)."""
    def key(row: dict[str, Any]) -> tuple:
        return tuple(str(row.get(column, "")).strip() for column in STTM_COLUMNS)

    before, after = [key(r) for r in original], [key(r) for r in edited]
    return sum(1 for row in after if row not in before) + sum(1 for row in before if row not in after)


def column_options() -> dict[str, list[str]]:
    """Allowed values for the dropdown columns of the mapping editor."""
    return {
        "data_type": sorted(DATA_TYPES),
        "nullable": ["true", "false"],
        "aggregation": sorted(AGGREGATIONS),
    }


# --------------------------------------------------------------------------- uploads
def upload_problems(names: Sequence[str]) -> list[str]:
    """Reasons a set of uploaded file names cannot be used (empty list when fine)."""
    if not names:
        return ["Upload at least one CSV file."]
    problems = [f"{name!r} is not a .csv file." for name in names if not str(name).lower().endswith(".csv")]
    plain = [Path(name).name for name in names]
    duplicates = sorted({n for n in plain if plain.count(n) > 1})
    if duplicates:
        problems.append(f"Two uploads share the name {duplicates[0]!r}; rename one.")
    if not problems:
        try:
            table_names(plain)
        except ValueError as error:
            problems.append(str(error))
    return problems


def save_uploads(uploads: Sequence[Any], directory: Path | None = None) -> list[str]:
    """Validate and save uploaded files to landing/<batch>/<name>.csv; return their paths."""
    names = [Path(upload.name).name for upload in uploads]
    problems = upload_problems(names)
    if problems:
        raise ValueError(" ".join(problems))
    contents = [upload.getvalue() for upload in uploads]
    empty = [name for name, content in zip(names, contents) if not content.strip()]
    if empty:
        raise ValueError(f"{empty[0]!r} is empty.")
    batch = (directory or LANDING_DIR) / uuid.uuid4().hex      # validated first: a rejected upload leaves nothing behind
    batch.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, content in zip(names, contents):
        destination = batch / name
        destination.write_bytes(content)
        paths.append(str(destination))
    return paths


def start_new_run(orchestrator: Orchestrator, uploads: Sequence[Any], business_intent: str) -> PipelineState:
    """Save the uploads and start Phase 1 (profile + Bronze mapping)."""
    if not business_intent.strip():
        raise ValueError("Enter the business question the report should answer.")
    return orchestrator.start_run(save_uploads(uploads), business_intent)


# --------------------------------------------------------------------------- data previews
def source_previews(state: PipelineState, layer: str) -> list[tuple[str, pd.DataFrame]]:
    """The data the mapping under review reads: raw CSVs, Bronze files or Silver files."""
    if layer == "bronze":
        return [(Path(p).name, head_of_csv(p)) for p in state.csv_paths]
    paths = state.bronze_output_paths if layer == "silver" else state.silver_output_paths
    return [(Path(p).name, head_of_parquet(p)) for p in paths]


def head_of_csv(path: str | Path, rows: int = PREVIEW_ROWS) -> pd.DataFrame:
    """The first rows of a CSV as text, reading only those rows (never the whole file)."""
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(path, nrows=rows, dtype="string", encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {Path(path).name}")


def head_of_parquet(path: str | Path, rows: int = PREVIEW_ROWS) -> pd.DataFrame:
    """The first rows of a Parquet file, reading one small batch only."""
    batch = next(pq.ParquetFile(path).iter_batches(batch_size=rows), None)
    return batch.to_pandas() if batch is not None else pd.DataFrame()


def load_profile(state: PipelineState) -> dict[str, Any] | None:
    """The raw-data profile JSON of the run, if it exists."""
    if not state.profile_path or not Path(state.profile_path).is_file():
        return None
    return json.loads(Path(state.profile_path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- review helpers
def sensitive_in_mapping(rows: Sequence[dict[str, Any]], profile: dict[str, Any] | None) -> list[str]:
    """Source columns kept by a mapping that look sensitive (cards, bank details, ids, credentials, contacts)."""
    flagged = {(s["table"], s["column"]): s["kind"] for s in (profile or {}).get("sensitive_columns", [])}
    found = set()
    for row in rows:
        table, column = str(row.get("source_table", "")), str(row.get("source_col", ""))
        kind = flagged.get((table, column)) or sensitivity_by_name(column)
        if kind:
            found.add(f"{table}.{column} ({kind})")
    return sorted(found)


def previous_step_report_path(state: PipelineState, layer: str) -> str:
    """The results report of the step whose output the mapping under review reads (none for Bronze)."""
    return {"silver": state.bronze_report_path, "gold": state.silver_report_path}.get(layer, "")


def load_layer_report(path: str) -> dict[str, Any] | None:
    """A layer step's JSON report, or None if it is missing or unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")) if path else None
    except (OSError, ValueError):
        return None


def layer_report_table(report: dict[str, Any]) -> pd.DataFrame:
    """What a layer step did, one row per table, in plain words."""
    rows = []
    for table in report.get("tables", []):
        if "merged_rows" in table:                                   # Gold
            rows.append({
                "Table": table["table"], "Built from": ", ".join(table.get("source_tables", [])),
                "Joined on": ", ".join(table.get("join_keys", [])) or "-", "Rows after joining": table["merged_rows"],
                "Rows out": table["output_rows"], "Grouped by": ", ".join(table.get("group_by", [])) or "-",
                "Measures": ", ".join(f"{k} = {v}" for k, v in table.get("measures", {}).items()) or "-",
            })
        else:
            converted = table.get("coerced_to_null", {})
            rows.append({
                "Table": table["table"], "Rows in": table["source_rows"], "Rows out": table["output_rows"],
                "Dropped (missing key)": table.get("rows_dropped_dropna", 0), "Dropped (duplicates)": table.get("rows_dropped_dedupe", 0),
                "Values filled": sum(table.get("filled", {}).values()),
                "Values that could not be converted": sum(converted.values()),
                "Details": "; ".join([f"{c}: {n} unconvertible" for c, n in converted.items()]
                                     + [f"{c}: {n} filled" for c, n in table.get("filled", {}).items()]),
            })
    return pd.DataFrame(rows)


def hint_table(profile: dict[str, Any] | None) -> pd.DataFrame:
    """Suggested transformations found in the data, one row per hint."""
    rows = [
        {"Table": d["table"], "Column": c["name"], "Suggested operation": h["op"], "Why": h["reason"]}
        for d in (profile or {}).get("datasets", []) for c in d["columns"] for h in c.get("hints", [])
    ]
    return pd.DataFrame(rows, columns=["Table", "Column", "Suggested operation", "Why"])


def join_table(profile: dict[str, Any] | None) -> pd.DataFrame:
    """Detected links between tables, with the steps needed to make them joinable."""
    rows = [
        {"Many side": f"{k['left_table']}.{k['left_column']}", "One side": f"{k['right_table']}.{k['right_column']}",
         "Match": f"{k['overlap_ratio']:.0%}", "Type": k["relationship"].replace("_", " "),
         "To make joinable": "; ".join(k.get("alignment", [])) or ("-" if "warning" not in k else k["warning"])}
        for k in (profile or {}).get("join_keys", [])
    ]
    return pd.DataFrame(rows, columns=["Many side", "One side", "Match", "Type", "To make joinable"])


# --------------------------------------------------------------------------- diagnostics
def audit_table(events: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Audit events as a table: time, who, what, and the remaining details as text."""
    rows = []
    for event in events:
        details = {k: v for k, v in event.items() if k not in {"timestamp", "run_id", "agent", "action", "phase"}}
        rows.append(
            {
                "Time (UTC)": str(event.get("timestamp", ""))[11:19],
                "Phase": event.get("phase", ""),
                "Agent": event.get("agent", ""),
                "Event": str(event.get("action", "")).replace("_", " "),
                "Details": "; ".join(f"{k}={_short(v)}" for k, v in details.items()),
            }
        )
    return pd.DataFrame(rows, columns=["Time (UTC)", "Phase", "Agent", "Event", "Details"])


def trace_table(summaries: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Agent trace summaries as a table."""
    rows = [
        {
            "Agent": s["agent"], "Status": s["status"], "Seconds": s["duration_seconds"],
            "Tools called": ", ".join(t for t in s["tools"] if t) or "-", "Error": s.get("error") or "",
        }
        for s in summaries
    ]
    return pd.DataFrame(rows, columns=["Agent", "Status", "Seconds", "Tools called", "Error"])


def _short(value: Any, limit: int = 120) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"
