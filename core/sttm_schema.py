"""STTM (Source-to-Target Mapping) schema, validation and CSV I/O.

An STTM is one CSV per layer and run. Each row maps one source column to one
target column. `transformation_logic` is NOT free-form code: it is a
pipe-separated list of allowlisted operations (``trim|lowercase|fillna:mode``)
so a human can edit it safely and the layer agents can execute it without eval.
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from core.paths import LAYERS

STTM_COLUMNS = (
    "layer",
    "target_table",
    "source_table",
    "source_col",
    "target_col",
    "data_type",
    "transformation_logic",
    "aggregation",
    "nullable",
    "join_key",
    "description",
)
REQUIRED_COLUMNS = ("layer", "target_table", "source_table", "source_col", "target_col", "data_type")

DATA_TYPES = {
    "string", "integer", "float", "boolean", "date", "datetime",
}
DATA_TYPE_ALIASES = {
    "str": "string", "text": "string", "category": "string", "object": "string",
    "int": "integer", "int64": "integer", "number": "float", "float64": "float",
    "double": "float", "decimal": "float", "bool": "boolean", "timestamp": "datetime",
}
MAX_PROBLEMS = 15
TABLE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
AGGREGATIONS = {"", "sum", "avg", "mean", "count", "max", "min", "first", "last", "nunique"}

# Allowed transformation operations -> (allowed layers, allowed arguments or None).
# `None` arguments means the operation takes no argument; "*" means free text.
_FILLNA_ARGS = {"mean", "median", "mode", "zero", "empty", "value=*"}
# date_part:<part> extracts a component of a date; the data_type the target column must have is fixed per part.
DATE_PARTS: dict[str, str] = {
    "year": "integer", "quarter": "integer", "month": "integer", "week": "integer", "day": "integer",
    "weekday": "integer",                                    # 0 = Monday
    "year_month": "string", "year_quarter": "string",        # "2025-04", "2025-Q2"
    "month_name": "string", "day_name": "string",
}
ROUND_MAX_DIGITS = 6
MAX_MAP_PAIRS = 200
OPERATIONS: dict[str, tuple[set[str], set[str] | None]] = {
    "identity": ({"bronze", "silver", "gold"}, None),
    "trim": ({"bronze", "silver", "gold"}, None),
    "lowercase": ({"bronze", "silver", "gold"}, None),
    "uppercase": ({"bronze", "silver", "gold"}, None),
    "titlecase": ({"silver", "gold"}, None),
    "collapse_whitespace": ({"silver", "gold"}, None),
    "cast": ({"bronze", "silver", "gold"}, DATA_TYPES),
    "to_date": ({"silver", "gold"}, {"*"}),   # optional strptime format
    "date_part": ({"silver", "gold"}, set(DATE_PARTS)),
    "clean_number": ({"silver", "gold"}, {"eu"}),  # "₹1,234.50" or "(12)" -> a number; clean_number:eu reads "1.234,56"
    "map_values": ({"silver", "gold"}, {"*"}),    # map_values:ny=new york;n.y.=new york (blank replacement = null)
    "valid_range": ({"silver", "gold"}, {"*"}),   # valid_range:0..120, values outside become null (counted)
    "abs": ({"silver", "gold"}, None),            # absolute value (fix a sign convention)
    "round": ({"silver", "gold"}, {"*"}),         # round:<digits 0-6>, default 0
    "dropna": ({"silver"}, None),             # drop rows where this column is null
    "fillna": ({"silver"}, _FILLNA_ARGS),
    "dedupe": ({"silver"}, None),             # column is part of the dedup key
}


def parse_value_map(arg: str) -> dict[str, str]:
    """Parse the argument of ``map_values``: ``value=replacement;value=replacement`` (blank replacement = null)."""
    mapping: dict[str, str] = {}
    for pair in arg.split(";"):
        if not pair.strip():
            continue
        value, separator, replacement = pair.partition("=")
        value, replacement = value.strip(), replacement.strip()
        if not separator or not value:
            raise ValueError("map_values takes pairs like map_values:n.y.=ny;new york=ny (a blank replacement makes the value null)")
        if value in mapping:
            raise ValueError(f"map_values lists {value!r} twice")
        mapping[value] = replacement
    if not mapping:
        raise ValueError("map_values needs at least one value=replacement pair")
    if len(mapping) > MAX_MAP_PAIRS:
        raise ValueError(f"map_values accepts at most {MAX_MAP_PAIRS} pairs")
    return mapping


def parse_value_range(arg: str) -> tuple[float | None, float | None]:
    """Parse the argument of ``valid_range``: ``min..max`` where either side may be left out."""
    low_text, separator, high_text = arg.partition("..")
    try:
        low = float(low_text) if low_text.strip() else None
        high = float(high_text) if high_text.strip() else None
    except ValueError:
        low = high = None
        separator = ""
    if not separator or (low is None and high is None) or (low is not None and high is not None and low > high):
        raise ValueError("valid_range takes min..max, for example valid_range:0..120 or valid_range:0.. (min must not exceed max)")
    return low, high


@dataclass(frozen=True)
class Operation:
    """One parsed transformation step."""

    name: str
    arg: str = ""


def normalize_data_type(value: str) -> str:
    """Return the canonical data type name or raise ValueError."""
    key = (value or "").strip().lower()
    key = DATA_TYPE_ALIASES.get(key, key)
    if key not in DATA_TYPES:
        raise ValueError(f"Unsupported data_type {value!r}; choose from {sorted(DATA_TYPES)}")
    return key


def parse_transformation(text: str, layer: str | None = None) -> list[Operation]:
    """Parse ``op[:arg]|op[:arg]`` into operations, rejecting anything not allowlisted."""
    operations: list[Operation] = []
    for raw in (text or "").split("|"):
        raw = raw.strip()
        if not raw:
            continue
        name, _, arg = raw.partition(":")
        name, arg = name.strip().lower(), arg.strip()
        if name not in OPERATIONS:
            raise ValueError(f"Unsupported transformation {name!r}; allowed: {sorted(OPERATIONS)}")
        layers, args = OPERATIONS[name]
        if layer is not None and layer not in layers:
            raise ValueError(f"Transformation {name!r} is not allowed in the {layer} layer")
        if args is None:
            if arg:
                raise ValueError(f"Transformation {name!r} takes no argument")
        elif name == "cast":
            arg = normalize_data_type(arg)
        elif name == "date_part":
            arg = arg.lower()
            if arg not in DATE_PARTS:
                raise ValueError(f"date_part argument must be one of {sorted(DATE_PARTS)}")
        elif name == "clean_number":
            arg = arg.lower()
            if arg not in args and arg != "":
                raise ValueError("clean_number takes no argument, or eu for decimal commas (1.234,56)")
        elif name == "map_values":
            parse_value_map(arg)
        elif name == "valid_range":
            parse_value_range(arg)
        elif name == "round":
            if arg == "":
                arg = "0"
            if not arg.isdigit() or int(arg) > ROUND_MAX_DIGITS:
                raise ValueError(f"round takes a number of digits from 0 to {ROUND_MAX_DIGITS}")
        elif name == "fillna":
            if arg not in args and not arg.startswith("value=") or arg == "value=":
                raise ValueError("fillna argument must be mean, median, mode, zero, empty or value=<text>")
        # to_date accepts any optional strptime format string.
        operations.append(Operation(name, arg))
    return operations


def _normalize_row(row: dict[str, Any], index: int) -> dict[str, str]:
    missing = [c for c in REQUIRED_COLUMNS if not str(row.get(c) or "").strip()]
    if missing:
        raise ValueError(f"STTM row {index}: missing {', '.join(missing)}")
    normalized = {c: str(row.get(c) or "").strip() for c in STTM_COLUMNS}
    normalized["layer"] = normalized["layer"].lower()
    normalized["data_type"] = normalize_data_type(normalized["data_type"])
    normalized["aggregation"] = normalized["aggregation"].lower()
    normalized["nullable"] = (normalized["nullable"] or "true").lower()
    if normalized["nullable"] not in {"true", "false"}:
        raise ValueError(f"STTM row {index}: nullable must be true or false")
    return normalized


def validate_rows(
    rows: Iterable[dict[str, Any]],
    layer: str,
    available_columns: dict[str, set[str]] | None = None,
    require_all_sources: bool = False,
) -> list[dict[str, str]]:
    """Validate and normalize STTM rows for one layer.

    `available_columns` maps source_table -> set of columns present in that
    source (from the profile or previous-layer Parquet). When given, every
    source column and table must exist, so the layer agents never guess; with
    `require_all_sources` every source table must also be mapped (Bronze and
    Silver produce one output per input). Raises one ValueError that lists all
    problems found (up to MAX_PROBLEMS), so a caller can fix them in one pass.
    """
    if layer not in LAYERS:
        raise ValueError("layer must be bronze, silver or gold")
    problems: list[str] = []
    normalized: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        try:
            normalized.append(_normalize_row(row, index))
        except ValueError as error:
            problems.append(str(error))
    if not normalized and not problems:
        problems.append("STTM must contain at least one mapping")

    seen_targets: set[tuple[str, str]] = set()
    tables: dict[str, set[str]] = {}
    for index, row in enumerate(normalized, start=1):
        def problem(message: str, index: int = index) -> None:
            problems.append(f"STTM row {index} ({row['source_table']}.{row['source_col']}): {message}")

        if row["layer"] != layer:
            problem(f"layer {row['layer']!r} does not match {layer!r}")
        key = (row["target_table"], row["target_col"])
        if key in seen_targets:
            problem(f"duplicate target {row['target_table']}.{row['target_col']}")
        seen_targets.add(key)
        if layer == "gold" and not TABLE_NAME.fullmatch(row["target_table"]):
            problem(
                f"gold target_table {row['target_table']!r} must start with a letter and use only "
                "letters, digits and _ (it becomes a SQL table name)"
            )
        if layer in {"bronze", "silver"} and row["target_table"] != row["source_table"]:
            problem(f"{layer} keeps one table per source, so target_table must equal source_table")

        try:
            operations = parse_transformation(row["transformation_logic"], layer)
        except ValueError as error:
            problem(str(error))
            operations = []
        for operation in operations:
            if operation.name == "cast" and operation.arg != row["data_type"]:
                problem(f"cast:{operation.arg} conflicts with data_type {row['data_type']}; data_type is the final type")
            if (
                operation.name == "to_date"
                and row["data_type"] not in {"date", "datetime"}
                and not any(op.name in {"date_part", "clean_number", "abs", "round", "valid_range"} for op in operations)
            ):
                problem("to_date requires data_type date or datetime (unless date_part follows it)")
            if (
                operation.name == "fillna"
                and operation.arg in {"mean", "median", "zero"}
                and row["data_type"] not in {"integer", "float"}
            ):
                problem(f"fillna:{operation.arg} requires a numeric data_type")
            if operation.name == "fillna" and operation.arg == "empty" and row["data_type"] != "string":
                problem("fillna:empty requires data_type string")
            if operation.name == "date_part" and row["data_type"] != DATE_PARTS[operation.arg]:
                problem(f"date_part:{operation.arg} produces {DATE_PARTS[operation.arg]} values, so data_type must be {DATE_PARTS[operation.arg]}")
            if operation.name in {"clean_number", "abs", "round", "valid_range"} and row["data_type"] not in {"integer", "float"}:
                problem(f"{operation.name} produces a number, so data_type must be integer or float")

        aggregation = row["aggregation"]
        if aggregation not in AGGREGATIONS:
            problem(f"unsupported aggregation {aggregation!r}")
        elif aggregation and layer != "gold":
            problem("aggregation is only allowed in the gold layer")
        elif aggregation in {"sum", "avg", "mean"} and row["data_type"] not in {"integer", "float"}:
            problem(f"{aggregation} produces a number, so data_type must be integer or float")
        elif aggregation in {"count", "nunique"} and row["data_type"] != "integer":
            problem(f"{aggregation} produces a whole number, so data_type must be integer")

        if available_columns is not None:
            if row["source_table"] not in available_columns:
                problem(f"unknown source_table {row['source_table']!r}; available: {sorted(available_columns)}")
            elif row["source_col"] not in available_columns[row["source_table"]]:
                problem(
                    f"column {row['source_col']!r} does not exist in {row['source_table']!r}; "
                    f"available: {sorted(available_columns[row['source_table']])}"
                )
        tables.setdefault(row["target_table"], set()).add(row["source_table"])

    if layer == "gold":
        for table, sources in tables.items():
            keys = {
                part.strip()
                for r in normalized if r["target_table"] == table
                for part in r["join_key"].split(",") if part.strip()
            }
            if len(sources) > 1 and not keys:
                problems.append(f"Gold table {table!r} combines {sorted(sources)} but no join_key is given")
            if available_columns is not None:
                for source in sorted(sources & set(available_columns)):
                    absent = sorted(keys - available_columns[source])
                    if len(sources) > 1 and absent:
                        problems.append(f"Gold table {table!r}: join_key {absent} is not a column of {source!r}")
    if available_columns is not None and require_all_sources:
        unmapped = sorted(set(available_columns) - {r["source_table"] for r in normalized})
        if unmapped:
            problems.append(f"Every source table needs at least one mapped column; missing: {unmapped}")

    if problems:
        shown = "; ".join(problems[:MAX_PROBLEMS])
        extra = f" (+{len(problems) - MAX_PROBLEMS} more)" if len(problems) > MAX_PROBLEMS else ""
        raise ValueError(f"{len(problems)} STTM problem(s): {shown}{extra}")
    return normalized


def read_sttm(path: str | Path, layer: str | None = None) -> list[dict[str, str]]:
    """Read an STTM CSV; validate it when `layer` is given."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as sttm_file:
        reader = csv.DictReader(sttm_file)
        header = [h.strip() for h in (reader.fieldnames or [])]
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise ValueError(f"STTM is missing required columns: {', '.join(missing)}")
        rows = [{(k or "").strip(): v for k, v in row.items()} for row in reader]
    if not rows:
        raise ValueError("STTM must contain at least one mapping")
    return validate_rows(rows, layer or (rows[0].get("layer") or "").strip().lower())


def write_sttm(path: str | Path, rows: Iterable[dict[str, Any]], layer: str) -> Path:
    """Validate rows for `layer` and write them as an STTM CSV."""
    validated = validate_rows(rows, layer)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as sttm_file:
        writer = csv.DictWriter(sttm_file, fieldnames=STTM_COLUMNS)
        writer.writeheader()
        writer.writerows(validated)
    return destination
