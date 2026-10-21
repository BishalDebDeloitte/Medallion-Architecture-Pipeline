"""Canonical storage directories and run-scoped artifact filenames.

All runtime output (data, reports, audit logs, memory) lives under RUNTIME_DIR,
which defaults to the project root. Set MEDALLION_RUNTIME_DIR to redirect it,
e.g. to a temporary directory in tests.
"""

import os
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RUNTIME_DIR = Path(os.environ.get("MEDALLION_RUNTIME_DIR") or BASE_DIR).resolve()
DATA_DIR = RUNTIME_DIR / "data"
LANDING_DIR = DATA_DIR / "landing"
PROFILES_DIR = DATA_DIR / "profiles"
STTM_DIR = DATA_DIR / "sttm"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
TRACES_DIR = DATA_DIR / "traces"
REPORTS_DIR = DATA_DIR / "reports"
RUNS_DIR = DATA_DIR / "runs"
LAYER_REPORTS_DIR = DATA_DIR / "layer_reports"
AUDIT_DIR = RUNTIME_DIR / "audit_logs"
CHROMA_DIR = RUNTIME_DIR / ".chroma"

LAYERS = ("bronze", "silver", "gold")


def _component(value: str) -> str:
    """Reject unsafe filename components rather than accepting path traversal."""
    if not value or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Artifact name and run_id must contain only letters, digits, _ or -")
    return value


def safe_stem(name: str) -> str:
    """Turn an arbitrary file or table name into a safe artifact component."""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(name).stem).strip("_")
    if not stem:
        raise ValueError(f"Cannot derive a safe artifact name from {name!r}")
    return stem


def _layer(layer: str) -> str:
    if layer not in LAYERS:
        raise ValueError("layer must be bronze, silver or gold")
    return layer


def profile_file(run_id: str) -> Path:
    """Return the combined raw-data profile path for a run."""
    return PROFILES_DIR / f"profile_combined_{_component(run_id)}.json"


def parquet_profile_file(layer: str, run_id: str) -> Path:
    """Return the Parquet profile path for a materialized layer."""
    return PROFILES_DIR / f"profile_{_layer(layer)}_{_component(run_id)}.json"


def sttm_file(layer: str, run_id: str) -> Path:
    """Return the mapping CSV path for a layer and run."""
    return STTM_DIR / f"sttm_{_layer(layer)}_{_component(run_id)}.csv"


def approved_sttm_file(layer: str, run_id: str) -> Path:
    """Return the path of the human-approved (possibly edited) mapping CSV for a layer and run."""
    return STTM_DIR / f"sttm_{_layer(layer)}_{_component(run_id)}_approved.csv"


def layer_report_file(layer: str, run_id: str) -> Path:
    """Return the path of the JSON report of what a layer step did (rows in/out, dropped, filled)."""
    return LAYER_REPORTS_DIR / f"{_component(run_id)}_{_layer(layer)}_report.json"


def state_file(run_id: str) -> Path:
    """Return the persisted PipelineState path for a run."""
    return RUNS_DIR / f"{_component(run_id)}.json"


def bronze_file(source_stem: str, run_id: str) -> Path:
    """Return a run-scoped Bronze Parquet path."""
    return BRONZE_DIR / f"{_component(source_stem)}_{_component(run_id)}_bronze.parquet"


def silver_file(source_stem: str, run_id: str) -> Path:
    """Return a run-scoped Silver Parquet path."""
    return SILVER_DIR / f"{_component(source_stem)}_{_component(run_id)}_silver.parquet"


def gold_file(target_table: str, run_id: str) -> Path:
    """Return a run-scoped Gold Parquet path."""
    return GOLD_DIR / f"{_component(target_table)}_{_component(run_id)}.parquet"


def table_from_artifact(path: str | Path, run_id: str, layer: str) -> str:
    """Recover the table name from a Bronze, Silver or Gold artifact path of this run."""
    if layer not in LAYERS:
        raise ValueError("layer must be bronze, silver or gold")
    name = Path(path).name
    suffix = f"_{_component(run_id)}.parquet" if layer == "gold" else f"_{_component(run_id)}_{layer}.parquet"
    if not name.endswith(suffix) or len(name) == len(suffix):
        raise ValueError(f"{name!r} is not a {layer} artifact of run {run_id!r}")
    return name[: -len(suffix)]


def report_file(run_id: str) -> Path:
    """Return the executive HTML report path."""
    return REPORTS_DIR / f"report_{_component(run_id)}.html"


def report_data_file(run_id: str) -> Path:
    """Return the matching report metadata JSON path."""
    return REPORTS_DIR / f"report_{_component(run_id)}.json"


def audit_file(run_id: str, directory: Path | None = None) -> Path:
    """Return the audit JSONL path for a run."""
    return (directory or AUDIT_DIR) / f"{_component(run_id)}.jsonl"


def trace_file(agent: str, run_id: str, directory: Path | None = None) -> Path:
    """Return the per-agent trace path for a run."""
    return (directory or TRACES_DIR) / f"trace_{_component(agent)}_{_component(run_id)[:8]}.json"
