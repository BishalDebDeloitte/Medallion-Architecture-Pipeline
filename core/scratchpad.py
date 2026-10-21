"""Typed in-run handoff data written by specialist tools."""

from typing import NotRequired, TypedDict


class Scratchpad(TypedDict, total=False):
    """Explicit artifact keys shared by tools within one run.

    Tools write here; the Orchestrator reads it after a phase and copies the
    declared keys into PipelineState. Nothing else is exchanged implicitly.
    """

    profile_path: str
    sttm_bronze_path: str
    bronze_output_paths: list[str]
    sttm_silver_path: str
    silver_output_paths: list[str]
    sttm_gold_path: str
    gold_output_paths: list[str]
    report_path: str
    report_json_path: NotRequired[str]
    bronze_report_path: NotRequired[str]
    silver_report_path: NotRequired[str]
    gold_report_path: NotRequired[str]


SCRATCHPAD_KEYS = tuple(Scratchpad.__annotations__)


def new_scratchpad() -> Scratchpad:
    """Return an empty scratchpad for one phase."""
    return Scratchpad()
