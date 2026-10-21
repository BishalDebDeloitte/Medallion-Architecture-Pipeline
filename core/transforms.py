"""Deterministic execution of STTM rules on pandas data.

Everything an STTM can ask for is implemented here as an allowlisted
operation (see core/sttm_schema.py). Nothing is evaluated as code. Values that
cannot be converted become missing and are *counted and reported*, never
silently dropped.

Conventions
- Blank strings (after any text operation) are treated as missing.
- Integers: non-integral numbers are invalid. Numbers with thousands
  separators or currency symbols need the ``clean_number`` operation first
  (``clean_number:eu`` for decimal commas). Text that is still not a number after
  cleaning ("12abc", "1.2.3") becomes missing and is counted, never guessed.
- ``map_values`` replaces exact values (a blank replacement turns a placeholder
  such as "unknown" into null); ``valid_range:min..max`` turns impossible values
  (age 250, negative quantity) into counted nulls.
- Dates: ambiguous day/month strings are parsed month-first; use
  ``to_date:%d/%m/%Y`` for day-first data. Timezone-aware values are converted
  to UTC and stored naive. ``date`` is normalised to midnight.
- ``fillna:mean|median`` on an integer column is rounded to a whole number.
"""

import warnings
from typing import Any

import pandas as pd

from core.sttm_schema import Operation, parse_transformation, parse_value_map, parse_value_range

TEXT_OPERATIONS = {"trim", "lowercase", "uppercase", "titlecase", "collapse_whitespace"}
NUMERIC_OPERATIONS = {"clean_number", "abs", "round", "valid_range"}
COLUMN_OPERATIONS = {"identity", "to_date", "cast", "date_part", "map_values"} | TEXT_OPERATIONS | NUMERIC_OPERATIONS
CONVERSION_OPERATIONS = {"to_date", "date_part", "clean_number", "valid_range"}   # a value they cannot convert is counted and reported
_CURRENCY_WORDS = r"(?<![a-z])(?:usd|eur|gbp|inr|jpy|cny|aud|cad|chf|rs|rupees?|dollars?|euros?)(?![a-z])\.?"
TABLE_OPERATIONS = {"fillna", "dropna", "dedupe"}   # applied by cleanse(), after typing
TRUE_TOKENS = {"true", "t", "yes", "y", "1"}
FALSE_TOKENS = {"false", "f", "no", "n", "0"}


# --------------------------------------------------------------------------- casting
def _to_datetime(series: pd.Series, fmt: str | None = None) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, utc=True)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(
                series.astype("string"), errors="coerce", utc=True, format=fmt or "mixed",
            )
    return parsed.dt.tz_localize(None)


def _to_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    text = series.astype("string").str.strip().str.lower()
    mapping = {**{token: True for token in TRUE_TOKENS}, **{token: False for token in FALSE_TOKENS}}
    return text.map(mapping, na_action="ignore").astype("boolean")


def cast_series(series: pd.Series, data_type: str) -> tuple[pd.Series, int]:
    """Cast to an STTM data_type; return (converted, count of non-null values that became null)."""
    if data_type == "string":
        converted = series.astype("string")
    elif data_type in {"integer", "float"}:
        numbers = pd.to_numeric(series.astype("object"), errors="coerce")
        if data_type == "integer":
            converted = numbers.where(numbers % 1 == 0).astype("Int64")
        else:
            converted = numbers.astype("Float64")
    elif data_type == "boolean":
        converted = _to_boolean(series)
    elif data_type in {"date", "datetime"}:
        converted = _to_datetime(series)
        if data_type == "date":
            converted = converted.dt.normalize()
    else:
        raise ValueError(f"Unsupported data_type {data_type!r}")
    invalid = int((series.notna() & converted.isna()).sum())
    return converted, invalid


# --------------------------------------------------------------------------- per-column operations
def _date_part(dates: pd.Series, part: str) -> pd.Series:
    """Extract a component of a datetime series (integers as Int64, labels as string)."""
    if part == "year_month":
        return dates.dt.strftime("%Y-%m").astype("string")
    if part == "year_quarter":
        return (dates.dt.year.astype("Int64").astype("string") + "-Q" + dates.dt.quarter.astype("Int64").astype("string"))
    if part == "month_name":
        return dates.dt.month_name().astype("string")
    if part == "day_name":
        return dates.dt.day_name().astype("string")
    if part == "week":
        return dates.dt.isocalendar().week.astype("Int64")
    attribute = "weekday" if part == "weekday" else part
    return getattr(dates.dt, attribute).astype("Int64")


def _clean_number(series: pd.Series, decimal_comma: bool = False) -> pd.Series:
    """Strip currency symbols/codes, thousands separators and spaces; "(12)" becomes -12.

    Commas are thousands separators (1,234.50) unless ``decimal_comma`` is set (1.234,56).
    Anything that is still not a plain number afterwards ("12abc", "N/A", "1.2.3") becomes missing,
    so the caller counts it instead of silently keeping a wrong value.
    """
    text = series.astype("string").str.strip()
    negative = text.str.match(r"^\(.*\)$").fillna(False).astype(bool)
    text = text.str.replace("\u2212", "-", regex=False)
    text = text.str.replace(_CURRENCY_WORDS, "", case=False, regex=True)
    text = text.str.replace(r"[$€£¥₹%()\s]", "", regex=True)
    if decimal_comma:
        text = text.str.replace(r"\.(?=\d{3}(?:\D|$))", "", regex=True).str.replace(",", ".", regex=False)
    else:
        text = text.str.replace(",", "", regex=False)
    text = text.where(text.str.fullmatch(r"-?(?:\d+\.?\d*|\.\d+)").fillna(False).astype(bool))
    return text.mask(negative & text.notna(), "-" + text.str.lstrip("-"))


def apply_operation(series: pd.Series, operation: Operation) -> pd.Series:
    """Apply one column-level operation (text cleaning or date parsing)."""
    name = operation.name
    if name == "identity":
        return series
    if name == "to_date":
        return _to_datetime(series, operation.arg or None)
    if name == "date_part":
        return _date_part(_to_datetime(series), operation.arg)
    if name == "clean_number":
        return _clean_number(series, operation.arg == "eu")
    if name == "map_values":
        mapping = parse_value_map(operation.arg)
        text = series.astype("string")
        lookup = {value: mapping.get(value, value) for value in text.dropna().unique()}
        mapped = text.map(lookup).astype("string")
        return mapped.mask(mapped.str.strip() == "")
    if name == "valid_range":
        low, high = parse_value_range(operation.arg)
        numbers = pd.to_numeric(series.astype("object"), errors="coerce")
        inside = pd.Series(True, index=numbers.index)
        if low is not None:
            inside &= numbers >= low
        if high is not None:
            inside &= numbers <= high
        return numbers.where(inside)
    if name in {"abs", "round"}:
        numbers = pd.to_numeric(series.astype("object"), errors="coerce")
        return numbers.abs() if name == "abs" else numbers.round(int(operation.arg or 0))
    if name in TEXT_OPERATIONS:
        text = series.astype("string")
        if name == "trim":
            text = text.str.strip()
        elif name == "lowercase":
            text = text.str.lower()
        elif name == "uppercase":
            text = text.str.upper()
        elif name == "titlecase":
            text = text.str.title()
        else:
            text = text.str.replace(r"\s+", " ", regex=True)
        return text.mask(text.str.strip() == "")
    raise ValueError(f"Operation {name!r} is not a column-level operation")


def apply_column_operations(
    series: pd.Series, row: dict[str, str], layer: str,
) -> tuple[pd.Series, int]:
    """Apply a row's text/date operations (not the final data_type cast).

    Returns (series, count of values a ``to_date`` could not parse). Blank-to-null
    from text rules is intentional and not counted.
    """
    invalid = 0
    for operation in parse_transformation(row["transformation_logic"], layer):
        if operation.name in COLUMN_OPERATIONS and operation.name != "cast":
            converted = apply_operation(series, operation)
            if operation.name in CONVERSION_OPERATIONS:
                invalid += int((series.notna() & converted.isna()).sum())
            series = converted
    return series, invalid


def transform_columns(
    source: pd.DataFrame, rows: list[dict[str, str]], layer: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the target columns of one table from its STTM rows (text/date ops, then data_type cast)."""
    columns: dict[str, pd.Series] = {}
    coerced: dict[str, int] = {}
    for row in rows:
        series, invalid = apply_column_operations(source[row["source_col"]], row, layer)
        series, cast_invalid = cast_series(series, row["data_type"])
        invalid += cast_invalid
        if invalid:
            coerced[row["target_col"]] = invalid
        columns[row["target_col"]] = series
    frame = pd.DataFrame(columns, index=source.index).reset_index(drop=True)
    return frame, {"coerced_to_null": coerced}


# --------------------------------------------------------------------------- Silver cleansing
def _fill_value(series: pd.Series, argument: str, data_type: str) -> Any:
    non_null = series.dropna()
    if argument in {"mean", "median"}:
        if non_null.empty:
            return None
        value = float(non_null.mean() if argument == "mean" else non_null.median())
        return int(round(value)) if data_type == "integer" else value
    if argument == "zero":
        return 0 if data_type == "integer" else 0.0
    if argument == "mode":
        modes = non_null.mode()
        return None if modes.empty else modes.iloc[0]
    if argument == "empty":
        return ""
    if argument.startswith("value="):
        converted, invalid = cast_series(pd.Series([argument[len("value="):]], dtype="string"), data_type)
        if invalid or converted.isna().iloc[0]:
            raise ValueError(f"fillna value {argument[len('value='):]!r} is not a valid {data_type}")
        return converted.iloc[0]
    raise ValueError(f"Unsupported fillna argument {argument!r}")


def cleanse(
    frame: pd.DataFrame, rows: list[dict[str, str]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Silver cleansing in a fixed order: fillna, then dropna, then dedupe on the `dedupe` columns."""
    frame = frame.copy()
    filled: dict[str, int] = {}
    dropna_columns: list[str] = []
    dedupe_columns: list[str] = []
    for row in rows:
        target = row["target_col"]
        for operation in parse_transformation(row["transformation_logic"], "silver"):
            if operation.name == "fillna":
                before = int(frame[target].isna().sum())
                value = _fill_value(frame[target], operation.arg, row["data_type"])
                if value is not None and before:
                    frame[target] = frame[target].fillna(value)
                    filled[target] = filled.get(target, 0) + before - int(frame[target].isna().sum())
            elif operation.name == "dropna":
                dropna_columns.append(target)
            elif operation.name == "dedupe":
                dedupe_columns.append(target)

    rows_in = len(frame)
    if dropna_columns:
        frame = frame.dropna(subset=dropna_columns)
    after_dropna = len(frame)
    if dedupe_columns:
        frame = frame.drop_duplicates(subset=dedupe_columns, keep="first")
    return frame.reset_index(drop=True), {
        "filled": filled,
        "rows_dropped_dropna": rows_in - after_dropna,
        "rows_dropped_dedupe": after_dropna - len(frame),
    }


def enforce_nullability(frame: pd.DataFrame, rows: list[dict[str, str]], table: str) -> None:
    """Fail if a column declared nullable=false still contains missing values."""
    problems = [
        f"{table}.{row['target_col']} has {int(frame[row['target_col']].isna().sum())} missing value(s) "
        f"but is nullable=false"
        for row in rows
        if row["nullable"] == "false" and frame[row["target_col"]].isna().any()
    ]
    if problems:
        raise ValueError(
            "; ".join(problems)
            + ". Rows cannot be fixed by the agent: edit the STTM (e.g. add dropna/fillna or allow nulls)."
        )
