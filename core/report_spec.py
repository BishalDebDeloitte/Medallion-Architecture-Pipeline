"""The report specification the Reporter agent returns, and its validation.

The agent never types data into a chart. It names a query it executed
(`query_id`) and the result columns to plot; the numbers come from the stored
query result. Validation also encodes the chart-form rules (no one-bar charts or
two-slice pies, series caps per chart type) so a violation is fed back to the
agent for correction instead of rendering a misleading chart.
"""

import json
import re
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

MAX_KPIS = 4
MAX_INSIGHTS = 8
MAX_CHARTS = 6
MAX_SERIES_TOTAL = 8          # distinct colour-coded entities across the whole report
BAR_MAX_CATEGORIES = 30
LINE_MAX_POINTS = 500
SCATTER_MAX_POINTS = 2000
PIE_MIN_SLICES, PIE_MAX_SLICES = 3, 6
COLOR_LIMIT = {"bar": 6, "line": 6, "scatter": 3}     # scatter is an all-pairs form: cap at 3


class ReportSpecError(ValueError):
    """The agent's final answer cannot be used as a report."""


@dataclass
class QueryResult:
    """One executed query and its full result."""

    id: str
    sql: str
    frame: pd.DataFrame


class _Model(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")


class KpiSpec(_Model):
    """A headline number read from the first row of a one-row query."""

    label: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    column: str = Field(min_length=1)
    prefix: str = ""
    suffix: str = ""


class InsightSpec(_Model):
    """A finding, optionally backed by a query."""

    title: str = Field(min_length=1)
    detail: str = Field(min_length=1)
    query_id: str | None = None


class ChartSpec(_Model):
    """A chart drawn from the columns of an executed query."""

    title: str = Field(min_length=1)
    type: Literal["bar", "line", "pie", "scatter"]
    query_id: str = Field(min_length=1)
    x: str = Field(min_length=1)
    y: str = Field(min_length=1)
    color: str | None = None
    description: str = ""


class ReportSpec(_Model):
    """The executive report content."""

    headline: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    kpis: list[KpiSpec] = Field(default_factory=list)
    insights: list[InsightSpec] = Field(default_factory=list)
    charts: list[ChartSpec] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


def humanize(name: str) -> str:
    """Turn a column name such as total_amount into 'Total amount'."""
    text = re.sub(r"[_\s]+", " ", str(name)).strip()
    return text[:1].upper() + text[1:] if text else str(name)


def extract_json_object(text: str) -> dict:
    """Pull the JSON object out of a model reply (plain, fenced, or surrounded by prose)."""
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    attempts = [candidate]
    start, end = candidate.find("{"), candidate.rfind("}")
    if 0 <= start < end:
        attempts.append(candidate[start:end + 1])
    for attempt in attempts:
        try:
            parsed = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ReportSpecError("The final answer must be a single JSON object with the report fields and nothing else")


def parse_report_spec(text: str) -> ReportSpec:
    """Parse and structurally validate the agent's final answer."""
    payload = extract_json_object(text)
    try:
        return ReportSpec.model_validate(payload)
    except ValidationError as error:
        details = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors()[:8])
        raise ReportSpecError(f"The report JSON is invalid: {details}") from error


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _chart_problems(chart: ChartSpec, frame: pd.DataFrame) -> list[str]:
    label = f"Chart {chart.title!r}"
    problems = []
    needed = [chart.x, chart.y] + ([chart.color] if chart.color else [])
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        return [f"{label}: column(s) {missing} are not in query {chart.query_id} (columns: {list(frame.columns)})"]
    if chart.type == "pie" and chart.color:
        problems.append(f"{label}: a pie takes no color column (slices are colored by x)")
    if chart.type in {"bar", "line", "pie", "scatter"} and not _is_numeric(frame[chart.y]):
        problems.append(f"{label}: y column {chart.y!r} must be numeric")
    if chart.type == "scatter" and not _is_numeric(frame[chart.x]):
        problems.append(f"{label}: scatter x column {chart.x!r} must be numeric")
    rows = len(frame)
    if chart.type in {"bar", "line"} and chart.color is None and rows < 2:
        problems.append(f"{label}: a {chart.type} chart needs at least 2 rows; show a single value as a KPI instead")
    if chart.type == "bar" and rows > BAR_MAX_CATEGORIES:
        problems.append(f"{label}: {rows} bars is too many; aggregate or LIMIT the query (top {BAR_MAX_CATEGORIES} at most)")
    if chart.type == "line" and rows > LINE_MAX_POINTS:
        problems.append(f"{label}: {rows} points is too many; aggregate the query to at most {LINE_MAX_POINTS}")
    if chart.type == "scatter" and rows > SCATTER_MAX_POINTS:
        problems.append(f"{label}: {rows} points is too many for a scatter; sample or aggregate to {SCATTER_MAX_POINTS}")
    if chart.type == "pie":
        if not PIE_MIN_SLICES <= rows <= PIE_MAX_SLICES:
            problems.append(
                f"{label}: a pie needs {PIE_MIN_SLICES}-{PIE_MAX_SLICES} slices (got {rows}); use a bar chart or "
                "KPIs for other counts"
            )
        if _is_numeric(frame[chart.y]) and (frame[chart.y] < 0).any():
            problems.append(f"{label}: a pie cannot show negative values")
    if chart.color and chart.type in COLOR_LIMIT:
        distinct = frame[chart.color].nunique(dropna=True)
        if distinct > COLOR_LIMIT[chart.type]:
            problems.append(
                f"{label}: {distinct} color series exceeds the limit of {COLOR_LIMIT[chart.type]} for a "
                f"{chart.type} chart; filter or aggregate to fewer series"
            )
    return problems


def chart_entities(chart: ChartSpec, frame: pd.DataFrame) -> list[str]:
    """The colour-coded entities of a chart, in order of appearance (empty for a single series)."""
    column = chart.x if chart.type == "pie" else chart.color
    if column is None or column not in frame.columns:
        return []
    return [str(v) for v in dict.fromkeys(frame[column].dropna().tolist())]


def validate_spec(spec: ReportSpec, queries: dict[str, QueryResult]) -> list[str]:
    """Return every problem found in the spec (empty when it can be rendered)."""
    problems: list[str] = []
    if not spec.charts:
        problems.append("At least one chart is required")
    for name, items, limit in (
        ("charts", spec.charts, MAX_CHARTS), ("insights", spec.insights, MAX_INSIGHTS), ("kpis", spec.kpis, MAX_KPIS),
    ):
        if len(items) > limit:
            problems.append(f"Too many {name} ({len(items)}); at most {limit}")

    def known(query_id: str, what: str) -> QueryResult | None:
        if query_id not in queries:
            problems.append(f"{what} cites query {query_id!r}, which was not executed (executed: {sorted(queries)})")
        return queries.get(query_id)

    entities: dict[str, None] = {}
    for chart in spec.charts:
        result = known(chart.query_id, f"Chart {chart.title!r}")
        if result is not None:
            chart_issues = _chart_problems(chart, result.frame)
            problems.extend(chart_issues)
            if not chart_issues:
                entities.update(dict.fromkeys(chart_entities(chart, result.frame)))
    if len(entities) > MAX_SERIES_TOTAL:
        problems.append(
            f"The report color-codes {len(entities)} distinct entities; at most {MAX_SERIES_TOTAL} can keep a "
            "stable color. Reduce the series in the charts."
        )
    for insight in spec.insights:
        if insight.query_id:
            known(insight.query_id, f"Insight {insight.title!r}")
    for kpi in spec.kpis:
        result = known(kpi.query_id, f"KPI {kpi.label!r}")
        if result is None:
            continue
        frame = result.frame
        if len(frame) != 1:
            problems.append(f"KPI {kpi.label!r}: query {kpi.query_id} must return exactly 1 row (got {len(frame)})")
        elif kpi.column not in frame.columns:
            problems.append(f"KPI {kpi.label!r}: column {kpi.column!r} is not in query {kpi.query_id}")
        elif not _is_numeric(frame[kpi.column]) or frame[kpi.column].isna().iloc[0]:
            problems.append(f"KPI {kpi.label!r}: {kpi.column!r} must be a non-null number")
    return problems
