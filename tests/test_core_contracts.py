"""Offline tests for the Step 1 core contracts: paths, state, scratchpad, STTM schema."""

import csv
import tempfile
import unittest
from pathlib import Path

from core import paths
from core.scratchpad import SCRATCHPAD_KEYS, new_scratchpad
from core.state import AWAITING_APPROVAL, PipelineState
from core.sttm_schema import (
    STTM_COLUMNS, Operation, parse_transformation, read_sttm, validate_rows, write_sttm,
)


def row(**overrides):
    base = {
        "layer": "silver", "target_table": "orders", "source_table": "orders",
        "source_col": "amount", "target_col": "amount", "data_type": "float",
        "transformation_logic": "", "aggregation": "", "nullable": "true",
        "join_key": "", "description": "",
    }
    return {**base, **overrides}


class PathTests(unittest.TestCase):
    def test_layout_matches_spec(self):
        self.assertEqual(paths.BRONZE_DIR.name, "bronze")
        self.assertEqual(paths.SILVER_DIR.name, "silver")
        self.assertEqual(paths.GOLD_DIR.name, "gold")
        self.assertEqual(paths.REPORTS_DIR, paths.DATA_DIR / "reports")

    def test_runtime_output_is_redirected_during_tests(self):
        self.assertNotEqual(paths.RUNTIME_DIR, paths.BASE_DIR)

    def test_safe_stem_sanitizes_upload_names(self):
        self.assertEqual(paths.safe_stem("My Sales (2024).csv"), "My_Sales_2024")
        with self.assertRaises(ValueError):
            paths.safe_stem("???.csv")

    def test_unsafe_run_id_is_rejected(self):
        with self.assertRaises(ValueError):
            paths.sttm_file("bronze", "../evil")
        with self.assertRaises(ValueError):
            paths.sttm_file("platinum", "run-1")


class StateTests(unittest.TestCase):
    def test_defaults(self):
        state = PipelineState(csv_paths=["a.csv"])
        self.assertEqual((state.csv_paths, state.status, state.current_phase), (["a.csv"], "pending", "phase_0"))

    def test_status_is_validated_on_creation_and_assignment(self):
        for bad in ("awaiting_aproval", "awaiting_bronze_approval", ""):
            with self.assertRaises(ValueError, msg=bad):
                PipelineState(status=bad)
        state = PipelineState()
        with self.assertRaises(ValueError):
            state.status = "done"
        state.status = "failed"
        self.assertEqual(state.status, "failed")

    def test_spec_fields_exist(self):
        fields = set(PipelineState.model_fields)
        for name in (
            "run_id", "business_intent", "csv_paths", "current_phase", "status", "profile_path",
            "sttm_bronze_path", "bronze_output_paths", "sttm_silver_path", "silver_output_paths",
            "sttm_gold_path", "gold_output_paths", "report_path", "errors", "pending_approval_path",
        ):
            self.assertIn(name, fields)

    def test_default_lists_are_not_shared(self):
        first, second = PipelineState(), PipelineState()
        first.errors.append("boom")
        self.assertEqual(second.errors, [])
        self.assertEqual(PipelineState(status=AWAITING_APPROVAL).status, "awaiting_approval")


class ScratchpadTests(unittest.TestCase):
    def test_keys_cover_every_handoff(self):
        for key in ("profile_path", "sttm_bronze_path", "bronze_output_paths", "sttm_silver_path",
                    "silver_output_paths", "sttm_gold_path", "gold_output_paths", "report_path"):
            self.assertIn(key, SCRATCHPAD_KEYS)
        self.assertEqual(new_scratchpad(), {})


class TransformationParsingTests(unittest.TestCase):
    def test_parses_chain_and_normalizes_cast_alias(self):
        ops = parse_transformation("trim | lowercase | cast:int", "silver")
        self.assertEqual(ops, [Operation("trim"), Operation("lowercase"), Operation("cast", "integer")])

    def test_fillna_forms(self):
        self.assertEqual(parse_transformation("fillna:median")[0].arg, "median")
        self.assertEqual(parse_transformation("fillna:value=Unknown")[0].arg, "value=Unknown")
        for bad in ("fillna", "fillna:max", "fillna:value="):
            with self.assertRaises(ValueError, msg=bad):
                parse_transformation(bad)

    def test_rejects_unknown_and_code_like_input(self):
        for bad in ("eval:os.system('x')", "drop_table", "trim:5", "cast:blob"):
            with self.assertRaises(ValueError, msg=bad):
                parse_transformation(bad)

    def test_layer_restrictions(self):
        parse_transformation("dropna", "silver")
        with self.assertRaises(ValueError):
            parse_transformation("dropna", "bronze")


class SttmValidationTests(unittest.TestCase):
    def test_valid_rows_are_normalized(self):
        rows = validate_rows([row(data_type="FLOAT64", nullable="")], "silver")
        self.assertEqual(rows[0]["data_type"], "float")
        self.assertEqual(rows[0]["nullable"], "true")

    def test_rejects_wrong_layer_duplicates_and_missing_fields(self):
        with self.assertRaises(ValueError):
            validate_rows([row(layer="bronze")], "silver")
        with self.assertRaises(ValueError):
            validate_rows([row(), row()], "silver")
        with self.assertRaises(ValueError):
            validate_rows([row(source_col="")], "silver")
        with self.assertRaises(ValueError):
            validate_rows([], "silver")

    def test_aggregation_only_in_gold(self):
        with self.assertRaises(ValueError):
            validate_rows([row(aggregation="sum")], "silver")
        validate_rows([row(layer="gold", aggregation="sum")], "gold")
        with self.assertRaises(ValueError):
            validate_rows([row(layer="gold", aggregation="median")], "gold")

    def test_source_columns_are_checked_against_available_columns(self):
        available = {"orders": {"amount"}}
        validate_rows([row()], "silver", available)
        with self.assertRaises(ValueError):
            validate_rows([row(source_col="missing")], "silver", available)
        with self.assertRaises(ValueError):
            validate_rows([row(source_table="other")], "silver", available)

    def test_gold_multi_source_table_requires_join_key(self):
        rows = [
            row(layer="gold", target_table="sales", source_table="orders", target_col="a", source_col="a"),
            row(layer="gold", target_table="sales", source_table="customers", target_col="b", source_col="b"),
        ]
        with self.assertRaises(ValueError):
            validate_rows(rows, "gold")
        rows[0]["join_key"] = "customer_id"
        validate_rows(rows, "gold")


class SttmRoundTripTests(unittest.TestCase):
    def test_write_then_read_round_trip_and_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_sttm(Path(directory) / "sub" / "sttm_silver_r.csv", [row()], "silver")
            with path.open(encoding="utf-8", newline="") as handle:
                self.assertEqual(tuple(next(csv.reader(handle))), STTM_COLUMNS)
            self.assertEqual(read_sttm(path, "silver")[0]["target_col"], "amount")
            self.assertEqual(read_sttm(path)[0]["layer"], "silver")

    def test_human_edit_with_missing_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("layer,target_table\nsilver,orders\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_sttm(path)


class SttmConsistencyRuleTests(unittest.TestCase):
    def problems(self, rows, layer="silver", **kwargs):
        with self.assertRaises(ValueError) as caught:
            validate_rows(rows, layer, **kwargs)
        return str(caught.exception)

    def test_all_problems_are_collected_in_one_error(self):
        message = self.problems([
            row(source_col="a", target_col="a", transformation_logic="explode"),
            row(source_col="b", target_col="b", aggregation="sum"),
            row(source_col="c", target_col="c", data_type="blob"),
        ])
        self.assertIn("3 STTM problem(s)", message)
        for fragment in ("explode", "only allowed in the gold layer", "blob"):
            self.assertIn(fragment, message)

    def test_cast_must_agree_with_data_type(self):
        self.assertIn("conflicts with data_type", self.problems([row(transformation_logic="cast:string")]))
        validate_rows([row(transformation_logic="cast:float")], "silver")

    def test_operation_arguments_must_fit_the_data_type(self):
        self.assertIn("to_date requires", self.problems([row(data_type="string", transformation_logic="to_date")]))
        self.assertIn("numeric data_type", self.problems([row(data_type="string", transformation_logic="fillna:mean")]))
        validate_rows([row(data_type="string", transformation_logic="fillna:mode")], "silver")

    def test_aggregation_result_types(self):
        gold = {"layer": "gold"}
        self.assertIn("integer or float", self.problems([row(**gold, data_type="string", aggregation="sum")], "gold"))
        self.assertIn("must be integer", self.problems([row(**gold, data_type="float", aggregation="count")], "gold"))
        validate_rows([row(**gold, data_type="string", aggregation="max")], "gold")

    def test_bronze_and_silver_are_one_table_per_source(self):
        self.assertIn("target_table must equal source_table", self.problems([row(target_table="other")]))
        validate_rows([row(layer="gold", target_table="other")], "gold")

    def test_unknown_columns_list_the_available_ones(self):
        message = self.problems([row(source_col="nope")], available_columns={"orders": {"amount", "qty"}})
        self.assertIn("'nope'", message)
        self.assertIn("['amount', 'qty']", message)

    def test_every_source_table_can_be_required(self):
        available = {"orders": {"amount"}, "customers": {"name"}}
        self.assertIn("customers", self.problems([row()], available_columns=available, require_all_sources=True))
        validate_rows([row()], "silver", available)

    def test_join_key_must_exist_in_every_source_table(self):
        rows = [
            row(layer="gold", target_table="sales", source_table="orders", source_col="amount",
                target_col="amount", join_key="customer_id"),
            row(layer="gold", target_table="sales", source_table="customers", source_col="name",
                target_col="name"),
        ]
        available = {"orders": {"amount", "customer_id"}, "customers": {"name"}}
        self.assertIn("not a column of 'customers'", self.problems(rows, "gold", available_columns=available))
        available["customers"].add("customer_id")
        validate_rows(rows, "gold", available)


class ArtifactNameTests(unittest.TestCase):
    def test_table_name_round_trips_through_artifact_paths(self):
        self.assertEqual(paths.table_from_artifact(paths.bronze_file("orders", "run-1"), "run-1", "bronze"), "orders")
        self.assertEqual(paths.table_from_artifact(paths.silver_file("my_sales", "run-1"), "run-1", "silver"), "my_sales")
        self.assertEqual(paths.table_from_artifact(paths.gold_file("sales_by_customer", "run-1"), "run-1", "gold"), "sales_by_customer")

    def test_rejects_other_layers_other_runs_and_unknown_layers(self):
        silver = paths.silver_file("orders", "run-1")
        gold = paths.gold_file("orders", "run-1")
        for args in (
            (silver, "run-1", "bronze"), (silver, "run-2", "silver"), (silver, "run-1", "gold"),
            (gold, "run-1", "silver"), (gold, "run-2", "gold"), (gold, "run-1", "platinum"),
        ):
            with self.assertRaises(ValueError, msg=args):
                paths.table_from_artifact(*args)


if __name__ == "__main__":
    unittest.main()
