"""Wiring: build the phase-scoped Supervisor tools from the real specialist agents.

This module only connects things (paths, models, deferred inputs); it contains
no decisions. The Orchestrator owns state and sequencing and asks this toolkit
for the tools of one phase.
"""

from collections.abc import Callable, Mapping
from typing import Any

from agents.bronze_agent import make_bronze_agent_tool
from agents.gold_agent import make_gold_agent_tool
from agents.profiler import make_profiler_agent_tool
from agents.reporter import make_reporter_agent_tool
from agents.silver_agent import make_silver_agent_tool
from agents.sttm_generator import SttmInputs, available_columns, load_context, make_sttm_agent_tool
from core.scratchpad import Scratchpad
from core.state import PipelineState


def _from_scratchpad(scratchpad: Scratchpad, key: str, needs: str) -> Any:
    """Read a value an earlier tool in the same phase must have produced."""
    value = scratchpad.get(key)
    if not value:
        raise ValueError(f"{key} is not available yet: run {needs} first")
    return value


class AgentToolkit:
    """Builds the real agent tools; `models` is one model, a {role: model} mapping, or None for the configured LLM."""

    def __init__(self, models: Any | Mapping[str, Any] | None = None, agent_factory: Callable[..., Any] | None = None) -> None:
        self.models = models
        self.agent_factory = agent_factory

    def model(self, role: str) -> Any | None:
        """The model for an agent role (profiler, sttm, bronze, silver, gold, reporter, supervisor)."""
        return self.models.get(role) if isinstance(self.models, Mapping) else self.models

    # ------------------------------------------------------------------ STTM sources
    def sttm_inputs(self, layer: str, state: PipelineState, scratchpad: Scratchpad | None = None) -> SttmInputs:
        """Inputs for the STTM of a layer; with a scratchpad, sources are read from it at call time."""
        common = {"run_id": state.run_id, "business_intent": state.business_intent}
        if layer == "bronze":
            profile = (scratchpad or {}).get("profile_path") or state.profile_path
            if not profile:
                raise ValueError("profile_path is not available yet: run profiler_agent first")
            return SttmInputs("bronze", profile_path=profile, **common)
        if layer == "silver":
            paths = (scratchpad or {}).get("bronze_output_paths") or state.bronze_output_paths
            if not paths:
                raise ValueError("bronze_output_paths is not available yet: run bronze_agent first")
            return SttmInputs("silver", parquet_paths=tuple(paths), previous_sttm_path=state.sttm_bronze_path, **common)
        paths = (scratchpad or {}).get("silver_output_paths") or state.silver_output_paths
        if not paths:
            raise ValueError("silver_output_paths is not available yet: run silver_agent first")
        return SttmInputs("gold", parquet_paths=tuple(paths), previous_sttm_path=state.sttm_silver_path, **common)

    def available_columns(self, layer: str, state: PipelineState) -> dict[str, set[str]] | None:
        """Source tables and columns an edited STTM of this layer may reference."""
        return available_columns(load_context(self.sttm_inputs(layer, state)))

    def _sttm_tool(self, layer: str, state: PipelineState, scratchpad: Scratchpad) -> Callable[[str], str]:
        return make_sttm_agent_tool(
            lambda: self.sttm_inputs(layer, state, scratchpad), scratchpad,
            model=self.model("sttm"), agent_factory=self.agent_factory, layer=layer,
        )

    # ------------------------------------------------------------------ tools per phase
    def phase_tools(self, phase: str, state: PipelineState, scratchpad: Scratchpad) -> list[Callable[[str], str]]:
        """The two tools of a phase: the layer executor (or profiler) and the next layer's STTM agent."""
        run_id, intent, factory = state.run_id, state.business_intent, self.agent_factory
        if phase == "phase_1":
            return [
                make_profiler_agent_tool(state.csv_paths, intent, run_id, scratchpad, self.model("profiler"), factory),
                self._sttm_tool("bronze", state, scratchpad),
            ]
        if phase == "phase_2":
            return [
                make_bronze_agent_tool(
                    state.csv_paths, state.sttm_bronze_path, run_id, scratchpad,
                    approved=bool(state.sttm_bronze_path), model=self.model("bronze"), agent_factory=factory,
                ),
                self._sttm_tool("silver", state, scratchpad),
            ]
        if phase == "phase_3":
            return [
                make_silver_agent_tool(
                    state.bronze_output_paths, state.sttm_silver_path, run_id, scratchpad,
                    approved=bool(state.sttm_silver_path), model=self.model("silver"), agent_factory=factory,
                ),
                self._sttm_tool("gold", state, scratchpad),
            ]
        if phase == "phase_4":
            return [
                make_gold_agent_tool(
                    state.silver_output_paths, state.sttm_gold_path, intent, run_id, scratchpad,
                    approved=bool(state.sttm_gold_path), model=self.model("gold"), agent_factory=factory,
                ),
                make_reporter_agent_tool(
                    lambda: list(_from_scratchpad(scratchpad, "gold_output_paths", "gold_agent")),
                    intent, run_id, scratchpad, sttm_gold_path=state.sttm_gold_path,
                    model=self.model("reporter"), agent_factory=factory, profile_path=state.profile_path,
                ),
            ]
        raise ValueError(f"Unknown phase: {phase}")

    def regenerate_tools(self, layer: str, state: PipelineState, scratchpad: Scratchpad) -> list[Callable[[str], str]]:
        """Only the STTM agent of the layer under review (its sources already exist)."""
        return [self._sttm_tool(layer, state, scratchpad)]
