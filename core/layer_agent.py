"""Shared shell for specialist agents that execute an approved STTM (Bronze, Silver, Gold)."""

import json
import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

from core.agent_runner import final_text, invoke_agent
from core.observability import AgentTrace
from core.paths import layer_report_file
from core.scratchpad import Scratchpad
from core.tooling import err, ok

logger = logging.getLogger(__name__)


def write_layer_report(layer: str, run_id: str, tables: list[dict[str, Any]]) -> str:
    """Save what a layer step did (per table: rows in/out, rows dropped, values filled or unconvertible)."""
    destination = layer_report_file(layer, run_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"layer": layer, "run_id": run_id, "generated_at": datetime.now(timezone.utc).isoformat(), "tables": tables},
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return str(destination)


def run_layer_agent(
    *,
    agent_name: str,
    run_id: str,
    trace_input: dict[str, Any],
    scratchpad: Scratchpad,
    output_key: str,
    build_tools: Callable[[], Sequence[Callable[..., str]]],
    system_prompt: str,
    user_message: str,
    verify: Callable[[Any], dict[str, Any]],
    model: Any | None = None,
    agent_factory: Callable[..., Any] | None = None,
) -> str:
    """Run one agent, check that it produced `output_key`, verify the output, return JSON.

    The scratchpad entry is cleared first, so success always means this
    invocation produced the output. `verify` inspects the produced artifacts
    and returns a summary dict (it raises if they are wrong). Never raises.
    """
    trace = AgentTrace(agent_name, run_id).set_input(**trace_input)
    try:
        scratchpad.pop(output_key, None)
        messages = invoke_agent(build_tools(), system_prompt, user_message, model, agent_factory)
        trace.extract_from_messages(messages)
        output = scratchpad.get(output_key)
        if not output:
            explanation = final_text(messages) or "no explanation given"
            raise ValueError(f"The {agent_name} agent did not produce {output_key}: {explanation}")
        summary = verify(output)
        trace.set_output(**{output_key: output}, **summary).complete()
        return ok(**{output_key: output}, **summary, message=final_text(messages))
    except Exception as error:
        logger.exception("%s agent failed for run %s", agent_name, run_id)
        trace.fail(error)
        return err(str(error))
