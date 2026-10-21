"""Offline tests for report specification validation and rendering (Step 6)."""

import json
import re
import unittest
from unittest.mock import patch

import pandas as pd

from core import report_render
from core.report_render import SERIES, assign_colors, build_figure, format_number, render_report, table_html
from core.report_spec import (
    ChartSpec, KpiSpec, QueryResult, ReportSpec, ReportSpecError, extract_json_object, humanize,
    parse_report_spec, validate_spec,
)

CATEGORIES = pd.DataFrame({"customer": list("abcde"), "revenue": [50.0, 40.0, 30.0, 20.0, 10.0]})
TREND = pd.DataFrame({
    "month": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"] * 2),
    "region": ["North"] * 3 + ["South"] * 3,
    "revenue": [10.0, 12.0, 15.0, 8.0, 9.0, 14.0],
})


def queries(**frames):
    return {f"q{i}": QueryResult(f"q{i}", f"SELECT {i}", frame) for i, frame in enumerate(frames.values(), start=1)}


def chart(**overrides):
    base = {"title": "Revenue by customer", "type": "bar", "query_id": "q1", "x": "customer", "y": "revenue"}
    return ChartSpec(**{**base, **overrides})


def spec(**overrides):
    base = {"headline": "Customer A leads.", "summary": "Revenue is concentrated.", "charts": [chart()]}
    return ReportSpec(**{**base, **overrides})


class ParsingTests(unittest.TestCase):
    def test_plain_fenced_and_prose_wrapped_json(self):
        payload = '{"headline": "h", "summary": "s", "charts": []}'
        for text in (payload, f"```json\n{payload}\n```", f"Here you go:\n{payload}\nThanks"):
            self.assertEqual(extract_json_object(text)["headline"], "h")

    def test_non_json_is_rejected(self):
        with self.assertRaises(ReportSpecError):
            extract_json_object("The revenue was high.")
        with self.assertRaises(ReportSpecError):
            extract_json_object("[1, 2, 3]")

    def test_missing_fields_and_bad_chart_type_are_reported_readably(self):
        with self.assertRaisesRegex(ReportSpecError, "headline"):
            parse_report_spec('{"summary": "s", "charts": []}')
        bad = '{"headline": "h", "summary": "s", "charts": [{"title": "t", "type": "donut", "query_id": "q1", "x": "a", "y": "b"}]}'
        with self.assertRaisesRegex(ReportSpecError, "charts.0.type"):
            parse_report_spec(bad)

    def test_unknown_fields_are_ignored_and_whitespace_trimmed(self):
        parsed = parse_report_spec('{"headline": "  h  ", "summary": "s", "extra": 1, "charts": []}')
        self.assertEqual(parsed.headline, "h")

    def test_humanize(self):
        self.assertEqual(humanize("total_amount"), "Total amount")
        self.assertEqual(humanize("customer_id"), "Customer id")


class ValidationTests(unittest.TestCase):
    def problems(self, report, **frames):
        return validate_spec(report, queries(**frames))

    def test_valid_spec_has_no_problems(self):
        self.assertEqual(self.problems(spec(), a=CATEGORIES), [])

    def test_a_chart_is_required_and_queries_must_exist(self):
        self.assertIn("At least one chart", " ".join(self.problems(spec(charts=[]), a=CATEGORIES)))
        text = " ".join(self.problems(spec(charts=[chart(query_id="q9")]), a=CATEGORIES))
        self.assertIn("'q9'", text)
        self.assertIn("['q1']", text)

    def test_columns_must_exist_and_y_must_be_numeric(self):
        self.assertIn("not in query q1", " ".join(self.problems(spec(charts=[chart(x="nope")]), a=CATEGORIES)))
        self.assertIn("must be numeric", " ".join(self.problems(spec(charts=[chart(y="customer")]), a=CATEGORIES)))

    def test_one_bar_and_two_slice_pie_are_refused(self):
        one = CATEGORIES.head(1)
        self.assertIn("at least 2 rows", " ".join(self.problems(spec(), a=one)))
        two = CATEGORIES.head(2)
        self.assertIn("3-6 slices", " ".join(self.problems(spec(charts=[chart(type="pie")]), a=two)))
        self.assertEqual(self.problems(spec(charts=[chart(type="pie")]), a=CATEGORIES.head(4)), [])
        self.assertIn("3-6 slices", " ".join(self.problems(spec(charts=[chart(type="pie")]), a=pd.concat([CATEGORIES] * 2))))

    def test_series_caps_per_chart_type(self):
        many = pd.DataFrame({"month": list(range(14)), "series": [f"s{i % 7}" for i in range(14)], "v": range(14)})
        line = chart(type="line", x="month", y="v", color="series")
        self.assertIn("7 color series exceeds the limit of 6", " ".join(self.problems(spec(charts=[line]), a=many)))
        scatter = pd.DataFrame({"x": range(8), "y": range(8), "g": ["a", "b", "c", "d"] * 2})
        self.assertIn("limit of 3", " ".join(self.problems(spec(charts=[chart(type="scatter", x="x", y="y", color="g")]), a=scatter)))

    def test_too_many_bars_and_pie_with_color_and_negative_values(self):
        big = pd.DataFrame({"c": [f"c{i}" for i in range(31)], "v": range(31)})
        self.assertIn("too many", " ".join(self.problems(spec(charts=[chart(x="c", y="v")]), a=big)))
        self.assertIn("no color column", " ".join(self.problems(spec(charts=[chart(type="pie", color="customer")]), a=CATEGORIES.head(4))))
        negative = CATEGORIES.head(4).assign(revenue=[1.0, -2.0, 3.0, 4.0])
        self.assertIn("negative", " ".join(self.problems(spec(charts=[chart(type="pie")]), a=negative)))

    def test_total_colour_coded_entities_across_charts_are_capped(self):
        wide_a = pd.DataFrame({"m": list(range(10)), "s": [f"a{i % 5}" for i in range(10)], "v": range(10)})
        wide_b = pd.DataFrame({"m": list(range(10)), "s": [f"b{i % 5}" for i in range(10)], "v": range(10)})
        charts = [chart(title="A", type="line", x="m", y="v", color="s"),
                  chart(title="B", type="line", x="m", y="v", color="s", query_id="q2")]
        self.assertIn("at most 8", " ".join(self.problems(spec(charts=charts), a=wide_a, b=wide_b)))

    def test_kpis_need_a_single_numeric_row(self):
        one = pd.DataFrame({"total": [800.0]})
        good = spec(kpis=[KpiSpec(label="Total", query_id="q2", column="total")])
        self.assertEqual(self.problems(good, a=CATEGORIES, b=one), [])
        self.assertIn("exactly 1 row", " ".join(self.problems(good, a=CATEGORIES, b=CATEGORIES)))
        text_kpi = spec(kpis=[KpiSpec(label="Name", query_id="q2", column="name")])
        self.assertIn("non-null number", " ".join(self.problems(text_kpi, a=CATEGORIES, b=pd.DataFrame({"name": ["x"]}))))

    def test_insight_queries_must_exist(self):
        report = spec(insights=[{"title": "t", "detail": "d", "query_id": "q7"}])
        self.assertIn("'q7'", " ".join(self.problems(report, a=CATEGORIES)))


class FigureTests(unittest.TestCase):
    def test_single_series_uses_one_colour_and_no_legend(self):
        figure = build_figure(chart(), CATEGORIES, {})
        self.assertFalse(figure.layout.showlegend)
        self.assertEqual({trace.marker.color for trace in figure.data}, {SERIES[0]})

    def test_multi_series_colours_follow_the_entity_in_fixed_order(self):
        line = chart(type="line", x="month", y="revenue", color="region", query_id="q1")
        colors = assign_colors(spec(charts=[line]), queries(a=TREND))
        self.assertEqual(colors, {"North": SERIES[0], "South": SERIES[1]})
        figure = build_figure(line, TREND, colors)
        self.assertTrue(figure.layout.showlegend)
        self.assertEqual({t.name: t.line.color for t in figure.data}, {"North": SERIES[0], "South": SERIES[1]})
        # filtering a series out must not repaint the survivor
        south_only = build_figure(line, TREND[TREND["region"] == "South"], colors)
        self.assertEqual(south_only.data[0].line.color, SERIES[1])

    def test_colours_are_shared_across_charts_by_entity(self):
        a = chart(title="A", type="line", x="month", y="revenue", color="region")
        b = chart(title="B", type="bar", x="month", y="revenue", color="region", query_id="q2")
        reordered = TREND.iloc[::-1].reset_index(drop=True)
        colors = assign_colors(spec(charts=[a, b]), queries(a=TREND, b=reordered))
        self.assertEqual(colors, {"North": SERIES[0], "South": SERIES[1]})

    def test_mark_specs(self):
        line = build_figure(chart(type="line", x="month", y="revenue", color="region"), TREND, {"North": SERIES[0], "South": SERIES[1]})
        self.assertTrue(all(t.line.width == 2 and t.marker.size == 8 for t in line.data))
        self.assertTrue(all(t.marker.line.width == 2 and t.marker.line.color == report_render.SURFACE for t in line.data))
        bar = build_figure(chart(), CATEGORIES, {})
        self.assertEqual(bar.data[0].marker.line.width, 2)
        self.assertEqual(bar.layout.yaxis.gridcolor, report_render.GRID)

    def test_long_labels_or_many_bars_go_horizontal(self):
        long = pd.DataFrame({"customer": ["A very long customer name", "Another long customer"], "revenue": [2.0, 1.0]})
        figure = build_figure(chart(), long, {})
        self.assertEqual(figure.data[0].orientation, "h")
        self.assertEqual(figure.layout.yaxis.autorange, "reversed")
        self.assertEqual(build_figure(chart(), CATEGORIES, {}).data[0].orientation, "v")

    def test_pie_and_scatter(self):
        entities = {str(c): SERIES[i] for i, c in enumerate(CATEGORIES["customer"])}
        pie = build_figure(chart(type="pie"), CATEGORIES, entities)
        self.assertTrue(pie.layout.showlegend)
        self.assertEqual(list(pie.data[0].marker.colors), SERIES[:5])
        scatter = build_figure(chart(type="scatter", x="revenue", y="revenue"), CATEGORIES, {})
        self.assertFalse(scatter.layout.showlegend)


class RenderTests(unittest.TestCase):
    def render(self, report, **frames):
        return render_report(
            report, queries(**frames), run_id="run-1", business_intent="Which customer buys most?",
            tables=[{"name": "sales", "row_count": 5, "columns": ["customer", "revenue"]}],
        )

    def test_html_contains_every_section_and_a_table_twin(self):
        report = spec(
            kpis=[KpiSpec(label="Total revenue", query_id="q2", column="total", prefix="$")],
            insights=[{"title": "Concentration", "detail": "Customer A is 33%.", "query_id": "q1"}],
            limitations=["Only five customers."],
        )
        html, data = self.render(report, a=CATEGORIES, b=pd.DataFrame({"total": [150.0]}))
        for expected in ("Which customer buys most?", "Customer A leads.", "Total revenue", "$150", "Concentration",
                         "Revenue by customer", "View data table", "Supporting queries", "Only five customers.", "sales"):
            self.assertIn(expected, html)
        self.assertIn("<table class=\"data\">", html)
        self.assertIn("color-scheme: light", html)
        self.assertEqual(data["kpis"][0]["value"], 150.0)
        self.assertEqual(data["charts"][0]["figure"]["data"][0]["type"], "bar")
        self.assertEqual([q["id"] for q in data["queries"]], ["q1", "q2"])
        json.dumps(data)                                    # fully serialisable

    def test_untrusted_text_and_data_are_escaped(self):
        evil = "</script><img src=x onerror=alert(1)>"
        frame = pd.DataFrame({"customer": [evil, "b"], "revenue": [2.0, 1.0]})
        report = spec(
            headline=f"<b>{evil}</b>", summary="<script>alert(1)</script>",
            charts=[chart(title=f"T {evil}", description=evil)],
        )
        html, _ = render_report(
            report, queries(a=frame), run_id="run-1", business_intent=evil, tables=[],
        )
        self.assertNotIn("<img src=x", html)
        self.assertNotIn("<script>alert(1)", html)
        self.assertEqual(html.count("</script>"), html.count("<script"))      # no injected extra tags

    def test_plotly_js_inline_or_cdn(self):
        with patch.object(report_render, "PLOTLY_JS", "cdn"):
            cdn, _ = self.render(spec(), a=CATEGORIES)
        self.assertIn("cdn.plot.ly", cdn)
        self.assertLess(len(cdn), 1_000_000)
        with patch.object(report_render, "PLOTLY_JS", "inline"):
            inline, _ = self.render(spec(charts=[chart(title="One"), chart(title="Two")]), a=CATEGORIES)
        self.assertGreater(len(inline), 1_000_000)
        self.assertEqual(len(re.findall(r'Plotly\.newPlot\(\s*"chart-', inline)), 2)
        self.assertLess(len(inline), 8_000_000)                              # two copies of the library would not fit

    def test_number_and_table_formatting(self):
        self.assertEqual(
            [format_number(v) for v in (1234567, 1234.56, 0.5, 12.0, 3.14159, None)],
            ["1,234,567", "1,235", "0.5", "12", "3.14", "—"],
        )
        # halves round up, as in a finance report (not to the even digit)
        self.assertEqual([format_number(v) for v in (463.625, 0.125, 2.675, 1234.5, -463.625)], ["463.63", "0.13", "2.68", "1,235", "-463.63"])
        html = table_html(pd.DataFrame({"a": range(60), "b": [None] * 60}), max_rows=5)
        self.assertEqual(html.count("<tr>"), 6)
        self.assertIn("Showing the first 5 of 60 rows", html)
        self.assertIn("—", html)


if __name__ == "__main__":
    unittest.main()
