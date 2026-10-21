"""Pydantic model for state shared across pipeline phases.

Only the Orchestrator mutates PipelineState. Specialist tools write to the
scratchpad (core/scratchpad.py) and never see this object.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Canonical run statuses.
PENDING = "pending"
RUNNING = "running"
AWAITING_APPROVAL = "awaiting_approval"
COMPLETE = "complete"
FAILED = "failed"
RUN_STATUSES = (PENDING, RUNNING, AWAITING_APPROVAL, COMPLETE, FAILED)
RunStatus = Literal["pending", "running", "awaiting_approval", "complete", "failed"]

# Phase names in execution order; phase_0 is run creation.
PHASES = ("phase_0", "phase_1", "phase_2", "phase_3", "phase_4")


class PipelineState(BaseModel):
    """Represent the current data and status of one pipeline run.

    Assignments are validated, so an unknown `status` fails immediately.
    """

    model_config = ConfigDict(validate_assignment=True)

    run_id: str = Field(default="", description="Unique identifier for this pipeline run.")
    status: RunStatus = Field(default=PENDING, description="One of RUN_STATUSES.")
    current_phase: str = Field(default="phase_0", description="Phase currently selected by the orchestrator.")

    csv_paths: list[str] = Field(default_factory=list, description="Paths to uploaded CSV files.")
    business_intent: str = Field(default="", description="User's question or goal in plain English.")
    task_description: str = Field(default="", description="Current goal passed to the Supervisor.")

    profile_path: str = Field(default="", description="Path to the combined data profile JSON.")

    sttm_bronze_path: str = Field(default="", description="Approved Bronze STTM CSV.")
    sttm_silver_path: str = Field(default="", description="Approved Silver STTM CSV.")
    sttm_gold_path: str = Field(default="", description="Approved Gold STTM CSV.")
    pending_approval_path: str = Field(default="", description="STTM CSV awaiting human review.")
    pending_approval_layer: str = Field(default="", description="Layer of the STTM awaiting review.")

    bronze_output_paths: list[str] = Field(default_factory=list, description="Bronze Parquet files.")
    silver_output_paths: list[str] = Field(default_factory=list, description="Silver Parquet files.")
    gold_output_paths: list[str] = Field(default_factory=list, description="Gold Parquet files.")

    report_path: str = Field(default="", description="Path to the final HTML report.")
    report_json_path: str = Field(default="", description="Path to the report's JSON twin.")
    bronze_report_path: str = Field(default="", description="JSON report of what the Bronze step did.")
    silver_report_path: str = Field(default="", description="JSON report of what the Silver step did.")
    gold_report_path: str = Field(default="", description="JSON report of what the Gold step did.")
    cancelled: bool = Field(default=False, description="True if a reviewer rejected a mapping and stopped the run.")

    error: Optional[str] = Field(default=None, description="Most recent error message, if any.")
    errors: list[str] = Field(default_factory=list, description="All errors recorded during this run.")
