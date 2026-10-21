"""Pure profiling logic: types, date formats, transformation hints, time coverage, join discovery.

No agents or files here, only pandas. The Profiler agent (agents/profiler.py) assembles these into the
profile that the STTM agent designs mappings from, so the signals computed here (for example "this column
holds yyyymmdd integers", "April 2026 has only 2 days of data", "orders.order_date matches calendar.full_date")
are what let the pipeline transform and join data correctly.
"""

import re
import warnings
from itertools import combinations
from typing import Any

import pandas as pd

HIGH_NULL_PCT = 30.0
LOW_CARDINALITY_MAX = 5
LOW_CARDINALITY_MIN_ROWS = 20
DATE_FORMAT_SAMPLE = 2000
MIN_DAYS_FOR_COVERAGE = 60            # a daily-ish series, not a handful of birthdays
EDGE_PARTIAL_BELOW = 0.9              # first/last month with less than this share of its days is partial
INTERIOR_DENSITY_MIN = 0.8            # interior months must be mostly full for edge gaps to mean anything
NAME_JOIN_MIN_OVERLAP = 0.5           # same-name key columns may have orphans
VALUE_JOIN_MIN_OVERLAP = 0.95         # differently named columns must match almost completely
MIN_KEY_DISTINCT = 2

_DATE_PATTERN = re.compile(r"^(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})")
_DAY_FIRST_OR_MONTH_FIRST = re.compile(r"^(\d{1,2})([/.-])(\d{1,2})\2(\d{2,4})(?:[ T].*)?$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?")
_YMD_SLASH = re.compile(r"^\d{4}/\d{1,2}/\d{1,2}$")
_BOOLEAN_TOKENS = {"true", "false", "yes", "no", "y", "n", "t", "f"}
_DATE_NAME = re.compile(r"(date|day|dt|time|period|dob|birth)", re.IGNORECASE)
_ROW_NUMBER_NAME = re.compile(
    r"^(index|idx|row|row_?(num|number|no|id)|rownum|serial(_?no)?|sr_?no|s_?no|unnamed_?\d*|level_\d+)$"
)
_NUMERIC_TEXT = re.compile(r"^\(?\s*[$€£¥₹]?\s*-?\s*[\d,]*\.?\d+\s*\)?\s*%?$")
_NUMBER_DECORATION = re.compile(r"[$€£¥₹,()%]")
_EU_NUMBER = re.compile(r"^\(?\s*[$€£¥₹]?\s*-?\s*(?:\d{1,3}(?:\.\d{3})+|\d+),\d{1,2}\s*\)?\s*%?$")
_CODE_NAME = re.compile(r"(phone|mobile|zip|postal|pin|code|sku)", re.IGNORECASE)
_AGE_NAME = re.compile(r"(^|_)age($|_)")
_NON_NEGATIVE_NAME = re.compile(r"(^|_)(qty|quantity|price|unit_?price|units|stock|weight|height|count)($|_)")


# --------------------------------------------------------------------------- names and roles
def looks_like_key(name: str) -> bool:
    """True for names such as order_id, CustomerID, productKey, sku."""
    return bool(
        re.search(r"(^|[_\W])(id|key|uuid|code|sku)$", name, re.IGNORECASE)
        or re.search(r"[a-z0-9]Id$|[A-Z]ID$", name)
    )


def normalise_key_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def snake(name: str) -> str:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    return re.sub(r"[^a-z0-9]+", "_", spaced.lower()).strip("_")


# --------------------------------------------------------------------------- dates
def _to_datetime(values: pd.Series, fmt: str | None = None) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(values.astype(str).str.strip(), errors="coerce", format=fmt or "mixed")


def _looks_yyyymmdd(non_null: pd.Series, name: str) -> bool:
    """Integers such as 20250402 in a column whose name says date."""
    if len(non_null) < 5 or not _DATE_NAME.search(name):
        return False
    numbers = pd.to_numeric(non_null, errors="coerce")
    if numbers.isna().any() or not ((numbers >= 19000101) & (numbers <= 21001231)).all():
        return False
    return _to_datetime(numbers.astype("int64"), "%Y%m%d").notna().mean() >= 0.99


def detect_date_format(values: pd.Series) -> tuple[str | None, str]:
    """Return (strptime format, certainty) for a column of date text or yyyymmdd integers.

    certainty: "iso" (unambiguous year-first), "unambiguous" (a value proves day-first or month-first),
    "ambiguous" (every day/month value is 12 or less: the order cannot be told apart) or "unknown".
    """
    non_null = values.dropna()
    if non_null.empty:
        return None, "unknown"
    if pd.api.types.is_numeric_dtype(non_null):
        return "%Y%m%d", "unambiguous"
    text = non_null.astype(str).str.strip().head(DATE_FORMAT_SAMPLE)
    if text.str.match(_ISO_DATETIME).all():
        return ("%Y-%m-%d %H:%M:%S" if text.str.count(":").eq(2).all() else "%Y-%m-%d %H:%M"), "iso"
    if text.str.match(_ISO_DATE).all():
        return "%Y-%m-%d", "iso"
    if text.str.match(_YMD_SLASH).all():
        return "%Y/%m/%d", "iso"
    parts = text.str.extract(_DAY_FIRST_OR_MONTH_FIRST)
    if parts[0].notna().all():
        separator = parts[1].iloc[0]
        year = "%Y" if parts[3].str.len().eq(4).all() else "%y"
        first_max, second_max = parts[0].astype(int).max(), parts[2].astype(int).max()
        if first_max > 12 and second_max <= 12:
            return f"%d{separator}%m{separator}{year}", "unambiguous"
        if second_max > 12 and first_max <= 12:
            return f"%m{separator}%d{separator}{year}", "unambiguous"
        if first_max <= 12 and second_max <= 12:
            return f"%m{separator}%d{separator}{year}", "ambiguous"
    return None, "unknown"


def time_coverage(dates: pd.Series) -> dict[str, Any] | None:
    """Describe a dense daily date column and flag partial first/last months.

    Returns None for sparse columns (birthdays, signup dates) where partial periods mean nothing.
    """
    days = dates.dropna().dt.normalize()
    if days.nunique() < MIN_DAYS_FOR_COVERAGE:
        return None
    present = days.drop_duplicates().groupby(days.drop_duplicates().dt.to_period("M")).size()
    if len(present) < 3:
        return None
    interior = present.iloc[1:-1]
    density = (interior.values / [p.days_in_month for p in interior.index])
    if float(pd.Series(density).median()) < INTERIOR_DENSITY_MIN:
        return None
    partial = [
        {"period": str(period), "days_present": int(present[period]), "days_in_month": int(period.days_in_month)}
        for period in (present.index[0], present.index[-1])
        if present[period] / period.days_in_month < EDGE_PARTIAL_BELOW
    ]
    return {"first": days.min().date().isoformat(), "last": days.max().date().isoformat(), "partial_periods": partial}


# --------------------------------------------------------------------------- types
def inferred_type(series: pd.Series, name: str = "") -> str:
    """Classify a column as integer, float, boolean, date, datetime, string or unknown."""
    non_null = series.dropna()
    if non_null.empty:
        return "unknown"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series) or (pd.api.types.is_float_dtype(series) and (non_null % 1 == 0).all()):
        return "date" if _looks_yyyymmdd(non_null, name) else "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    text = non_null.astype(str).str.strip()
    if text.str.lower().isin(_BOOLEAN_TOKENS).all():
        return "boolean"
    if text.str.match(_DATE_PATTERN).mean() >= 0.9 and _to_datetime(text).notna().mean() >= 0.9:
        return "datetime" if text.str.contains(r"\d:\d", regex=True).any() else "date"
    return "string"


def is_row_number(name: str, series: pd.Series, inferred: str) -> bool:
    """A sequential 0/1..n counter named like Index or Row: meaningless as a join key."""
    if inferred != "integer" or not _ROW_NUMBER_NAME.match(snake(name)):
        return False
    numbers = pd.to_numeric(series.dropna(), errors="coerce")
    count = len(numbers)
    return bool(
        count >= 2 and numbers.is_unique and int(numbers.min()) in (0, 1) and int(numbers.max()) - int(numbers.min()) + 1 == count
    )


def semantic_role(name: str, inferred: str, unique_count: int, non_null_count: int, row_number: bool = False) -> str:
    """Heuristic role used as a fallback and as a hint for the STTM agent."""
    if row_number:
        return "row_number"
    if looks_like_key(name):
        return "identifier"
    if inferred in {"date", "datetime"}:
        return "date"
    if inferred == "boolean":
        return "flag"
    if inferred in {"integer", "float"}:
        return "measure"
    if non_null_count >= LOW_CARDINALITY_MIN_ROWS and unique_count / non_null_count > 0.9:
        return "text"
    return "dimension"


# --------------------------------------------------------------------------- transformation hints
def category_variants(non_null: pd.Series) -> tuple[list[list[str]], dict[str, str]]:
    """Find spellings of the same category.

    Returns (groups that differ only by letter case, {lower-cased variant: most common lower-cased form}
    for groups that also differ by punctuation or spacing, e.g. "n.y." and "ny").
    """
    counts = non_null.astype(str).str.strip().value_counts()
    groups: dict[str, list[str]] = {}
    for value in counts.index:
        key = re.sub(r"[\W_]+", "", value.casefold())
        if key:
            groups.setdefault(key, []).append(value)
    case_only: list[list[str]] = []
    replacements: dict[str, str] = {}
    for variants in groups.values():
        if len(variants) < 2:
            continue
        frequency: dict[str, int] = {}
        for variant in variants:
            frequency[variant.lower()] = frequency.get(variant.lower(), 0) + int(counts[variant])
        if len(frequency) < len(variants):
            case_only.append(variants[:3])
        if len(frequency) > 1:
            canonical = max(frequency, key=frequency.get)
            replacements.update({form: canonical for form in frequency if form != canonical})
    return case_only, replacements


def _usable_in_map(text: str) -> bool:
    return bool(text) and not any(char in text for char in "|;=")


def transformation_hints(
    name: str, series: pd.Series, inferred: str, padded: int, date_format: tuple[str | None, str] | None,
    unique_count: int, non_null_count: int,
) -> list[dict[str, str]]:
    """Deterministic suggestions of STTM operations that this column's data calls for."""
    hints: list[dict[str, str]] = []
    non_null = series.dropna()
    if inferred == "boolean" and not pd.api.types.is_bool_dtype(series):
        hints.append({"op": "cast:boolean", "reason": "yes/no or true/false text (Y, yes and TRUE all become true): use data_type boolean"})
    if inferred == "string":
        if padded:
            hints.append({"op": "trim", "reason": f"{padded} value(s) have leading/trailing whitespace"})
        if 2 <= unique_count <= 200 and not looks_like_key(name):
            case_only, replacements = category_variants(non_null)
            if case_only and not replacements:
                examples = ", ".join(repr(v) for v in case_only[0])
                hints.append({"op": "lowercase", "reason": f"the same value appears in different letter cases (e.g. {examples})"})
            usable = {a: b for a, b in replacements.items() if _usable_in_map(a) and _usable_in_map(b)}
            if usable:
                pairs = ";".join(f"{a}={b}" for a, b in list(usable.items())[:8])
                hints.append({
                    "op": f"trim|lowercase|map_values:{pairs}",
                    "reason": f"{len(replacements)} spelling/punctuation variant(s) of the same category (check before applying)",
                })
        sample = non_null.astype(str).str.strip().head(DATE_FORMAT_SAMPLE)
        share = sample.str.match(_NUMERIC_TEXT).mean() if len(sample) else 0
        decorated = sample.str.contains(_NUMBER_DECORATION).mean() if len(sample) else 0
        eu_form = sample.str.match(_EU_NUMBER) if len(sample) else pd.Series(dtype=bool)
        plain = sample.str.fullmatch(r"-?\d+") if len(sample) else pd.Series(dtype=bool)
        if len(sample) and eu_form.any() and (eu_form | plain).mean() >= 0.95:
            hints.append({"op": "clean_number:eu", "reason": "numbers stored as text with decimal commas such as 1.234,56 (use data_type float)"})
        elif share >= 0.8 and decorated > 0:
            hints.append({"op": "clean_number", "reason": "numbers stored as text with symbols or separators (use data_type float/integer)"})
        elif len(sample) and not looks_like_key(name) and not _CODE_NAME.search(name):
            parsed = pd.to_numeric(sample, errors="coerce")
            bad = int(parsed.isna().sum())
            if 0 < bad and bad / len(sample) <= 0.1:
                examples = ", ".join(repr(v) for v in sample[parsed.isna()].unique()[:3])
                kind = "integer" if (parsed.dropna() % 1 == 0).all() else "float"
                hints.append({
                    "op": f"cast:{kind}",
                    "reason": f"{bad} of {len(sample)} sampled value(s) are not numbers (e.g. {examples}); they become null and are counted - consider fillna or dropna",
                })
    if inferred in {"date", "datetime"} and date_format and date_format[0]:
        fmt, certainty = date_format
        is_text = not pd.api.types.is_datetime64_any_dtype(series)
        if certainty == "iso" and is_text:
            hints.append({"op": "to_date", "reason": "dates stored as ISO text"})
        elif certainty == "unambiguous":
            hints.append({"op": f"to_date:{fmt}", "reason": f"dates stored as text in the format {fmt}"})
        elif certainty == "ambiguous":
            hints.append({"op": f"to_date:{fmt}", "reason": f"day/month order cannot be told from the data; assuming {fmt} - confirm with the user"})
    if inferred in {"integer", "float"} and non_null_count:
        numbers = pd.to_numeric(non_null, errors="coerce")
        nonzero = int((numbers != 0).sum())
        negative = int((numbers < 0).sum())
        label = snake(name)
        if nonzero >= 5 and negative / nonzero >= 0.95:
            hints.append({"op": "abs", "reason": "nearly all non-zero values are negative (a sign convention?)"})
        elif _AGE_NAME.search(label):
            outside = int(((numbers < 0) | (numbers > 120)).sum())
            if outside:
                hints.append({"op": "valid_range:0..120", "reason": f"{outside} age value(s) outside 0-120; they become null and are counted"})
        elif negative and _NON_NEGATIVE_NAME.search(label):
            hints.append({"op": "valid_range:0..", "reason": f"{negative} negative value(s) in a column that should not be negative; they become null and are counted"})
    if looks_like_key(name) and 0.5 <= unique_count / max(non_null_count, 1) < 1:
        hints.append({"op": "dedupe", "reason": "a key-like column with duplicate values"})
    return hints


# --------------------------------------------------------------------------- join discovery
def _key_kind(column: dict[str, Any]) -> str | None:
    kind = column["inferred_type"]
    if kind in {"date", "datetime"}:
        return "date"
    return kind if kind in {"integer", "string"} else None


def _comparable(series: pd.Series, column: dict[str, Any]) -> set[str]:
    """Distinct values in a form that compares equal across files (ISO dates, trimmed text, plain ints)."""
    distinct = pd.Series(series.dropna().unique())
    if _key_kind(column) == "date":
        fmt = column.get("date_format") if column.get("date_certainty") != "ambiguous" else None
        parsed = _to_datetime(distinct if not pd.api.types.is_numeric_dtype(distinct) else distinct.astype("int64"), fmt)
        return set(parsed.dropna().dt.strftime("%Y-%m-%d"))
    if pd.api.types.is_float_dtype(distinct) and len(distinct) and (distinct % 1 == 0).all():
        distinct = distinct.astype("int64")
    return set(distinct.astype(str).str.strip())


_GENERIC_TOKENS = {"id", "key", "code", "no", "num", "number", "ref", "idx", "fk", "pk"}


def _tokens(name: str) -> set[str]:
    return {t[:-1] if len(t) > 3 and t.endswith("s") else t for t in snake(name).split("_") if t and t not in _GENERIC_TOKENS}


def names_related(fk_name: str, pk_name: str, pk_table: str) -> bool:
    """True if the foreign key's name points at the primary key's column or table (cust_ref -> customers.customer_id)."""
    fk_tokens, pk_tokens = _tokens(fk_name), _tokens(pk_name) | _tokens(pk_table)
    return any(f == p or (min(len(f), len(p)) >= 4 and (f.startswith(p) or p.startswith(f))) for f in fk_tokens for p in pk_tokens)


def _different_name_link_is_credible(fk: dict[str, Any], fk_table: str, pk: dict[str, Any], pk_table: str) -> bool:
    """Evidence needed to link two differently named columns, because containment alone misleads.

    A small-range foreign key always sits inside a long sequential id column, and two sequential ids contain
    each other; so integers need related names, text keys need related names or a large exact match, and the
    "many" side must repeat. Dates are compared as dates and need a calendar-like set of distinct values.
    """
    if fk["kind"] == "date":
        return pk["unique_count"] >= 10
    if fk["unique"]:
        return False
    if names_related(fk["name"], pk["name"], pk_table):
        return True
    return pk["kind"] == "string" and fk["kind"] == "string" and pk["unique_count"] >= 20


def discover_join_keys(frames: dict[str, pd.DataFrame], datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find foreign-key style links between tables by value containment, whatever the columns are called.

    A link is a unique column (the "one" side) whose values contain nearly all the values of a column in another
    table (the "many" side). Same-named key columns may have orphans (>=50% overlap); differently named
    columns must match >=95% and pass `_different_name_link_is_credible`. Row-number columns, floats and tiny
    key sets are never keys. Dates written as ISO text, other text formats or yyyymmdd integers are compared
    as dates.
    """
    info = {d["table"]: {c["name"]: c for c in d["columns"]} for d in datasets}
    rows = {d["table"]: d["row_count"] for d in datasets}
    cache: dict[tuple[str, str], set[str]] = {}

    def values(table: str, column: str) -> set[str]:
        if (table, column) not in cache:
            cache[(table, column)] = _comparable(frames[table][column], info[table][column])
        return cache[(table, column)]

    def eligible(table: str) -> list[dict[str, Any]]:
        out = []
        for column in info[table].values():
            if column.get("row_number") or _key_kind(column) is None or column["null_pct"] >= 100:
                continue
            non_null = rows[table] - column["null_count"]
            out.append({**column, "kind": _key_kind(column), "unique": column["unique_count"] == non_null and non_null >= MIN_KEY_DISTINCT})
        return out

    found: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for table_a, table_b in combinations(frames, 2):
        for a in eligible(table_a):
            for b in eligible(table_b):
                if not (a["kind"] == b["kind"] or {a["kind"], b["kind"]} <= {"string", "integer"}):
                    continue
                same_name = normalise_key_name(a["name"]) == normalise_key_name(b["name"])
                both_key_like = looks_like_key(a["name"]) and looks_like_key(b["name"])
                threshold = NAME_JOIN_MIN_OVERLAP if same_name else VALUE_JOIN_MIN_OVERLAP
                orientations = [(a, table_a, b, table_b)] * a["unique"] + [(b, table_b, a, table_a)] * b["unique"]
                if not orientations and same_name and both_key_like:
                    orientations = [(a, table_a, b, table_b)]               # many-to-many, reported for information
                best = None
                for pk, pk_table, fk, fk_table in orientations:
                    if pk["unique_count"] < MIN_KEY_DISTINCT or fk["unique_count"] > pk["unique_count"] / threshold + 1:
                        continue
                    if not same_name and not _different_name_link_is_credible(fk, fk_table, pk, pk_table):
                        continue
                    fk_values, pk_values = values(fk_table, fk["name"]), values(pk_table, pk["name"])
                    if not fk_values:
                        continue
                    contained = len(fk_values & pk_values) / len(fk_values)
                    if contained >= threshold and (best is None or contained > best[0]):
                        best = (contained, pk, pk_table, fk, fk_table)
                if best is None:
                    continue
                contained, pk, pk_table, fk, fk_table = best
                relationship = (
                    "one_to_one" if pk["unique"] and fk["unique"]
                    else "many_to_one" if pk["unique"] else "many_to_many"
                )
                signature = frozenset({(pk_table, pk["name"]), (fk_table, fk["name"])})
                if signature in seen:
                    continue
                seen.add(signature)
                entry = {
                    "left_table": fk_table, "left_column": fk["name"],
                    "right_table": pk_table, "right_column": pk["name"],
                    "overlap_ratio": round(contained, 4), "relationship": relationship,
                }
                alignment = alignment_steps(fk_table, fk, pk_table, pk)
                if alignment:
                    entry["alignment"] = alignment
                if relationship == "many_to_many":
                    entry["warning"] = "the key repeats on both sides: joining the two would multiply rows"
                found.append(entry)
    return sorted(_best_link_per_column(found), key=lambda item: (-item["overlap_ratio"], item["left_table"], item["left_column"]))


def _best_link_per_column(links: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one link per (column, other table): a calendar with date_id and full_date would otherwise match twice.

    Preference: higher overlap, then the same column name, then a key-like name (date_id over full_date).
    """
    best: dict[tuple, dict[str, Any]] = {}
    for link in links:
        key = (link["left_table"], link["left_column"], link["right_table"])
        rank = (
            link["overlap_ratio"],
            normalise_key_name(link["left_column"]) == normalise_key_name(link["right_column"]),
            looks_like_key(link["right_column"]),
        )
        if key not in best or rank > best[key][0]:
            best[key] = (rank, link)
    return [link for _, link in best.values()]


def alignment_steps(left_table: str, left: dict[str, Any], right_table: str, right: dict[str, Any]) -> list[str]:
    """What Silver must do so that Gold can join on one shared column name and type."""
    steps = []
    if normalise_key_name(left["name"]) != normalise_key_name(right["name"]):
        steps.append(
            f"names differ: give {left_table}.{left['name']} and {right_table}.{right['name']} the same target_col "
            f"in Silver (for example {snake(right['name'])}) so one join_key works"
        )
    if _key_kind(left) == "date" or _key_kind(right) == "date":
        for table, column in ((left_table, left), (right_table, right)):
            fmt = column.get("date_format")
            conversion = f"to_date:{fmt}" if fmt and fmt != "%Y-%m-%d" else "to_date"
            steps.append(f"{table}.{column['name']}: {conversion} with data_type date, so both sides are real dates")
    elif left["inferred_type"] != right["inferred_type"]:
        steps.append("types differ (integer vs string): give both the same data_type in Silver")
    return steps
