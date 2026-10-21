"""Render a validated ReportSpec into a self-contained HTML report and a matching JSON document.

Chart design follows the data-viz method: one colour for a single series, the
validated categorical palette in fixed order for several (colour follows the
entity across the whole report, never its rank), thin marks with a 2px surface
gap, recessive solid hairline grid, a legend whenever there are two or more
series, hover tooltips, and a table twin for every chart (tooltips never gate a
value). The report is light-themed only; it is a printable executive document.
"""

import json
import os
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from html import escape
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.report_spec import ChartSpec, QueryResult, ReportSpec, chart_entities, humanize

# Reference palette, light mode (validated set, used in its documented order).
SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

PLOTLY_JS = os.getenv("REPORT_PLOTLY_JS", "inline").strip().lower()   # "inline" (offline-safe) or "cdn"
TABLE_ROWS = 50
JSON_QUERY_ROWS = 100
LONG_LABEL = 14
MANY_BARS = 8


# --------------------------------------------------------------------------- formatting
def format_number(value: Any) -> str:
    """Format a number for display: grouped thousands, up to 2 decimals."""
    if value is None or pd.isna(value):
        return "—"
    number = float(value)
    if number == int(number) and abs(number) < 1e15:
        return f"{int(number):,}"
    exact = Decimal(repr(number))                       # repr gives the shortest text that round-trips: 463.625
    if abs(number) >= 1000:
        return f"{exact.quantize(Decimal(1), rounding=ROUND_HALF_UP):,}"
    return f"{exact.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):,}".rstrip("0").rstrip(".")


def _cell(value: Any) -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return format_number(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d") if value == value.normalize() else value.strftime("%Y-%m-%d %H:%M")
    return str(value)


def table_html(frame: pd.DataFrame, max_rows: int = TABLE_ROWS) -> str:
    """An accessible HTML table of the first rows of a frame (the chart's table twin)."""
    shown = frame.head(max_rows)
    head = "".join(f"<th scope=\"col\">{escape(humanize(c))}</th>" for c in shown.columns)
    body = "".join(
        "<tr>" + "".join(
            f"<td class=\"{'num' if pd.api.types.is_numeric_dtype(frame[c]) else ''}\">{escape(_cell(v))}</td>"
            for c, v in zip(shown.columns, row)
        ) + "</tr>"
        for row in shown.itertuples(index=False, name=None)
    )
    note = f"<p class=\"note\">Showing the first {max_rows} of {len(frame)} rows.</p>" if len(frame) > max_rows else ""
    return f"<div class=\"table-wrap\"><table class=\"data\"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{note}"


# --------------------------------------------------------------------------- colours
def assign_colors(spec: ReportSpec, queries: dict[str, QueryResult]) -> dict[str, str]:
    """Give each colour-coded entity a palette slot, by first appearance, stable across charts."""
    colors: dict[str, str] = {}
    for chart in spec.charts:
        for entity in chart_entities(chart, queries[chart.query_id].frame):
            if entity not in colors and len(colors) < len(SERIES):
                colors[entity] = SERIES[len(colors)]
    return colors


# --------------------------------------------------------------------------- figures
def build_figure(chart: ChartSpec, frame: pd.DataFrame, colors: dict[str, str]) -> go.Figure:
    """Build one Plotly figure from a chart spec and its query result."""
    df = frame.copy()
    labels = {c: humanize(c) for c in df.columns}
    multi = chart.color is not None and chart.type != "pie"
    if multi:
        df[chart.color] = df[chart.color].astype(str)
    if chart.type == "pie":
        df[chart.x] = df[chart.x].astype(str)
    series_colors = {str(k): v for k, v in colors.items()}

    if chart.type == "bar":
        horizontal = df[chart.x].astype(str).str.len().max() > LONG_LABEL or df[chart.x].nunique() > MANY_BARS
        kwargs: dict[str, Any] = dict(
            x=chart.y if horizontal else chart.x, y=chart.x if horizontal else chart.y,
            orientation="h" if horizontal else "v", labels=labels,
        )
        if multi:
            figure = px.bar(df, color=chart.color, color_discrete_map=series_colors, barmode="group", **kwargs)
        else:
            figure = px.bar(df, color_discrete_sequence=[SERIES[0]], **kwargs)
        figure.update_layout(bargap=0.45, bargroupgap=0.12)
        figure.update_traces(marker_line_width=2, marker_line_color=SURFACE)
        try:
            figure.update_traces(marker_cornerradius=4)
        except ValueError:     # older plotly without rounded bar ends
            pass
        if horizontal:
            figure.update_yaxes(autorange="reversed")
            figure.update_layout(height=max(320, 30 * len(df) + 130))
    elif chart.type == "line":
        df = df.sort_values(chart.x)
        if multi:
            figure = px.line(df, x=chart.x, y=chart.y, color=chart.color, color_discrete_map=series_colors,
                             markers=True, labels=labels)
        else:
            figure = px.line(df, x=chart.x, y=chart.y, color_discrete_sequence=[SERIES[0]], markers=True, labels=labels)
        figure.update_traces(line_width=2, marker=dict(size=8, line=dict(width=2, color=SURFACE)))
        figure.update_layout(hovermode="x unified")
    elif chart.type == "pie":
        figure = px.pie(df, names=chart.x, values=chart.y, color=chart.x, color_discrete_map=series_colors, labels=labels)
        # Labels sit outside in ink: white-on-yellow/pink slices fail contrast and small slices clip text.
        figure.update_traces(
            sort=False, textinfo="percent", textposition="outside", textfont=dict(color=INK_2, size=13),
            marker=dict(line=dict(width=2, color=SURFACE)),
        )
    else:
        if multi:
            figure = px.scatter(df, x=chart.x, y=chart.y, color=chart.color, color_discrete_map=series_colors, labels=labels)
        else:
            figure = px.scatter(df, x=chart.x, y=chart.y, color_discrete_sequence=[SERIES[0]], labels=labels)
        figure.update_traces(marker=dict(size=10, line=dict(width=2, color=SURFACE)))

    showlegend = chart.type == "pie" or multi
    figure.update_layout(
        title=None, showlegend=showlegend, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_2, size=13), margin=dict(l=56, r=24, t=36 if showlegend else 16, b=56),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None, font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=GRID, font=dict(family=FONT, color=INK)),
        height=figure.layout.height or 380,
    )
    if chart.type != "pie":
        figure.update_xaxes(showgrid=False, linecolor=AXIS, ticks="outside", tickcolor=AXIS,
                            tickfont=dict(color=MUTED), title_font=dict(color=INK_2), automargin=True)
        figure.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=False, showline=False,
                            tickfont=dict(color=MUTED), title_font=dict(color=INK_2), automargin=True,
                            separatethousands=True)
    return figure


# --------------------------------------------------------------------------- HTML
CSS = f"""
:root {{ color-scheme: light; --surface: {SURFACE}; --page: {PAGE}; --ink: {INK}; --ink-2: {INK_2};
  --muted: {MUTED}; --grid: {GRID}; --axis: {AXIS}; --accent: {SERIES[0]}; --ring: rgba(11,11,11,0.10); }}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; background: var(--page); color: var(--ink); font-family: {FONT}; line-height: 1.5; }}
main {{ max-width: 960px; margin: 0 auto; padding: 32px 16px 56px; }}
.eyebrow {{ color: var(--muted); font-size: 12px; letter-spacing: .08em; text-transform: uppercase; margin: 0 0 8px; }}
h1 {{ font-size: 28px; line-height: 1.25; margin: 0 0 8px; }}
h2 {{ font-size: 18px; margin: 40px 0 12px; }}
h3 {{ font-size: 16px; margin: 0 0 4px; }}
.meta {{ color: var(--muted); font-size: 13px; margin: 0 0 24px; }}
.card {{ background: var(--surface); border: 1px solid var(--ring); border-radius: 8px; padding: 20px; }}
.headline {{ border-left: 4px solid var(--accent); font-size: 20px; font-weight: 600; line-height: 1.35; }}
.summary {{ color: var(--ink-2); margin: 16px 0 0; font-size: 16px; }}
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 16px; }}
.kpi {{ background: var(--surface); border: 1px solid var(--ring); border-radius: 8px; padding: 16px; }}
.kpi .label {{ color: var(--ink-2); font-size: 13px; }}
.kpi .value {{ font-size: 36px; font-weight: 600; line-height: 1.15; margin-top: 4px; }}
.insights {{ list-style: none; padding: 0; margin: 0; display: grid; gap: 12px; }}
.insights li {{ background: var(--surface); border: 1px solid var(--ring); border-radius: 8px; padding: 14px 16px; }}
.insights strong {{ display: block; margin-bottom: 2px; }}
.insights span, .chart-desc {{ color: var(--ink-2); font-size: 14px; }}
.chart {{ margin-bottom: 20px; }}
.chart-desc {{ margin: 0 0 8px; }}
details {{ margin-top: 10px; }}
summary {{ cursor: pointer; color: var(--ink-2); font-size: 13px; }}
.table-wrap {{ overflow-x: auto; margin-top: 8px; }}
table.data {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
table.data th, table.data td {{ padding: 6px 10px; border-bottom: 1px solid var(--grid); text-align: left; }}
table.data th {{ color: var(--ink-2); font-weight: 600; }}
table.data td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.note {{ color: var(--muted); font-size: 12px; margin: 6px 0 0; }}
.chart-note {{ color: var(--ink-2); background: #fff7e0; border-left: 3px solid {SERIES[3]}; font-size: 13px; margin: 8px 0 0; padding: 6px 10px; }}
pre {{ background: #f1f0ec; border-radius: 6px; padding: 10px 12px; overflow-x: auto; font-size: 12px; }}
ul.plain {{ margin: 0; padding-left: 20px; color: var(--ink-2); }}
footer {{ color: var(--muted); font-size: 12px; margin-top: 48px; }}
@media print {{ body {{ background: #fff; }} main {{ padding: 0; }} .card, .kpi, .insights li {{ break-inside: avoid; }} }}
"""


def _plotly_mode(first: bool) -> Any:
    if not first:
        return False
    return "cdn" if PLOTLY_JS == "cdn" else True


def partial_period_notes(chart: ChartSpec, frame: pd.DataFrame, partial_periods: dict[str, str]) -> list[str]:
    """Warnings for a chart whose x values include a month that has only part of its days of data."""
    if not partial_periods or chart.x not in frame.columns:
        return []
    shown = {str(value)[:7] for value in frame[chart.x].dropna()}
    return [
        f"{period} covers only {coverage}, so it is not comparable with full periods."
        for period, coverage in partial_periods.items() if period in shown
    ]


def render_report(
    spec: ReportSpec,
    queries: dict[str, QueryResult],
    *,
    run_id: str,
    business_intent: str,
    tables: list[dict[str, Any]],
    generated_at: datetime | None = None,
    partial_periods: dict[str, dict[str, str]] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return (html, report_json) for a spec that passed validate_spec.

    `partial_periods` maps a query id to {'YYYY-MM': '2 of 30 days'} for the months of that query's data that
    are only partly covered; any chart showing such a month gets a warning note.
    """
    generated_at = generated_at or datetime.now(timezone.utc)
    colors = assign_colors(spec, queries)

    figures = [build_figure(c, queries[c.query_id].frame, colors) for c in spec.charts]
    partial_periods = partial_periods or {}
    chart_notes = [partial_period_notes(c, queries[c.query_id].frame, partial_periods.get(c.query_id, {})) for c in spec.charts]
    chart_blocks = []
    for index, (chart, figure) in enumerate(zip(spec.charts, figures)):
        plot = figure.to_html(
            full_html=False, include_plotlyjs=_plotly_mode(index == 0), div_id=f"chart-{index + 1}",
            config={"displaylogo": False, "responsive": True},
        )
        frame = queries[chart.query_id].frame
        columns = [c for c in dict.fromkeys([chart.x, chart.y, chart.color]) if c]
        chart_blocks.append(
            f"<section class=\"card chart\" aria-labelledby=\"chart-title-{index + 1}\">"
            f"<h3 id=\"chart-title-{index + 1}\">{escape(chart.title)}</h3>"
            + (f"<p class=\"chart-desc\">{escape(chart.description)}</p>" if chart.description else "")
            + f"{plot}"
            + "".join(f"<p class=\"chart-note\">\u26a0 {escape(note)}</p>" for note in chart_notes[index])
            + f"<details><summary>View data table</summary>{table_html(frame[columns])}</details></section>"
        )

    kpi_html = "".join(
        f"<div class=\"kpi\"><div class=\"label\">{escape(k.label)}</div>"
        f"<div class=\"value\">{escape(k.prefix)}{escape(format_number(queries[k.query_id].frame[k.column].iloc[0]))}{escape(k.suffix)}</div></div>"
        for k in spec.kpis
    )
    insight_html = "".join(
        f"<li><strong>{escape(i.title)}</strong><span>{escape(i.detail)}</span></li>" for i in spec.insights
    )
    used_queries = list(queries.values())
    query_html = "".join(
        f"<details><summary>{escape(q.id)}: {len(q.frame)} row(s)</summary><pre>{escape(q.sql)}</pre>{table_html(q.frame, 10)}</details>"
        for q in used_queries
    )
    source_html = "".join(
        f"<li><code>{escape(t['name'])}</code>: {t['row_count']:,} rows, {len(t['columns'])} columns</li>" for t in tables
    )
    html = (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{escape(business_intent)}</title><style>{CSS}</style></head><body><main>"
        "<p class=\"eyebrow\">Executive report</p>"
        f"<h1>{escape(business_intent)}</h1>"
        f"<p class=\"meta\">Run {escape(run_id)} · {generated_at.strftime('%Y-%m-%d %H:%M UTC')}</p>"
        f"<section class=\"card headline\"><div>{escape(spec.headline)}</div></section>"
        + (f"<div class=\"kpis\">{kpi_html}</div>" if kpi_html else "")
        + f"<p class=\"summary\">{escape(spec.summary)}</p>"
        + (f"<h2>Key insights</h2><ul class=\"insights\">{insight_html}</ul>" if insight_html else "")
        + f"<h2>Charts</h2>{''.join(chart_blocks)}"
        + (f"<h2>Limitations</h2><ul class=\"plain\">{''.join(f'<li>{escape(x)}</li>' for x in spec.limitations)}</ul>"
           if spec.limitations else "")
        + f"<h2>Supporting queries</h2><div class=\"card\">{query_html}</div>"
        + f"<h2>Data sources</h2><ul class=\"plain\">{source_html}</ul>"
        + f"<footer>Generated from the Gold layer. Every figure above comes from the queries listed; "
          f"narrative text is written by an AI model and should be checked against them.</footer>"
        "</main></body></html>"
    )

    report_json = {
        "run_id": run_id,
        "business_intent": business_intent,
        "generated_at": generated_at.isoformat(),
        "headline": spec.headline,
        "summary": spec.summary,
        "kpis": [
            {**k.model_dump(), "value": float(queries[k.query_id].frame[k.column].iloc[0])} for k in spec.kpis
        ],
        "insights": [i.model_dump() for i in spec.insights],
        "charts": [
            {**c.model_dump(), "notes": notes, "figure": json.loads(f.to_json())}
            for c, f, notes in zip(spec.charts, figures, chart_notes)
        ],
        "limitations": spec.limitations,
        "queries": [
            {
                "id": q.id, "sql": q.sql, "row_count": len(q.frame), "columns": [str(c) for c in q.frame.columns],
                "rows": json.loads(q.frame.head(JSON_QUERY_ROWS).to_json(orient="records", date_format="iso")),
            }
            for q in used_queries
        ],
        "tables": tables,
        "color_assignments": colors,
        "partial_periods": partial_periods,
    }
    return html, report_json
