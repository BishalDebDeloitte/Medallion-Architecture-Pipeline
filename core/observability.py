"""Capture observable agent message and tool activity as JSON traces."""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from core.paths import TRACES_DIR, trace_file


TRACES_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)


def load_traces(run_id: str, directory: Path | None = None) -> list[dict[str, Any]]:
    """Return every saved trace record of a run (all agents), ordered by start time."""
    base = directory or TRACES_DIR
    records: list[dict[str, Any]] = []
    for path in base.glob(f"trace_*_{run_id[:8]}.json"):
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("Unreadable trace file: %s", path)
            continue
        records.extend(r for r in (loaded if isinstance(loaded, list) else [loaded]) if r.get("run_id") == run_id)
    return sorted(records, key=lambda record: record.get("started_at", ""))


def summarize_trace(record: dict[str, Any]) -> dict[str, Any]:
    """A compact view of one trace record for display."""
    return {
        "agent": record.get("agent"),
        "status": record.get("status"),
        "started_at": record.get("started_at"),
        "duration_seconds": round(record.get("duration_seconds") or 0, 2),
        "tools": [call.get("name") for call in record.get("tool_calls", [])],
        "plan": record.get("plan"),
        "error": record.get("error"),
        "output": record.get("output", {}),
    }


class AgentTrace:
    """Collect an agent run's inputs, messages, tool calls, and output."""

    def __init__(self, agent_name: str, run_id: str) -> None:
        self.agent_name = agent_name
        self.run_id = run_id
        self.start_time = time.time()
        self.trace: dict[str, Any] = {
            "agent": agent_name,
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "input": {},
            "plan": None,
            "tool_calls": [],
            "reasoning_steps": [],
            "output": {},
            "duration_seconds": None,
            "status": "running",
        }

    def set_input(self, **kwargs: Any) -> "AgentTrace":
        """Store the agent's input parameters."""
        self.trace["input"] = kwargs
        return self

    def set_plan(self, plan_str: str) -> "AgentTrace":
        """Store the agent's stated plan."""
        self.trace["plan"] = plan_str
        return self

    def set_output(self, **kwargs: Any) -> "AgentTrace":
        """Store the agent's output values."""
        self.trace["output"] = kwargs
        return self

    def extract_from_messages(self, messages: Sequence[BaseMessage]) -> "AgentTrace":
        """Record observable content and tool activity from message history."""
        for message in messages:
            if isinstance(message, HumanMessage):
                self.trace["reasoning_steps"].append(
                    {"type": "task_input", "content": message.content}
                )
            elif isinstance(message, AIMessage):
                tool_calls = getattr(message, "tool_calls", []) or []
                self.trace["tool_calls"].extend(tool_calls)
                content = message.content
                if content:
                    self.trace["reasoning_steps"].append(
                        {"type": "ai_reasoning", "content": content}
                    )
                    if self.trace["plan"] is None:
                        self.trace["plan"] = (
                            content
                            if isinstance(content, str)
                            else json.dumps(content, ensure_ascii=False)
                        )
            elif isinstance(message, ToolMessage):
                self.trace["reasoning_steps"].append(
                    {
                        "type": "tool_result",
                        "tool_call_id": message.tool_call_id,
                        "name": message.name,
                        "content": message.content,
                    }
                )
        return self

    def complete(self, status: str = "success") -> Path:
        """Finalize and save the trace, then print a concise run summary."""
        self.trace["duration_seconds"] = time.time() - self.start_time
        self.trace["status"] = status
        trace_path = trace_file(self.agent_name, self.run_id, TRACES_DIR)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        records = []
        if trace_path.exists():
            prior = json.loads(trace_path.read_text(encoding="utf-8"))
            records = prior if isinstance(prior, list) else [prior]
        records.append(self.trace)
        with trace_path.open("w", encoding="utf-8") as output_file:
            json.dump(records, output_file, ensure_ascii=False, indent=2)
        logger.info(
            "Trace saved: agent=%s run_id=%s status=%s duration=%.3fs",
            self.agent_name, self.run_id, status, self.trace["duration_seconds"],
        )
        return trace_path

    def fail(self, error: Exception | str) -> Path:
        """Record an error and save this run with failed status."""
        self.trace["error"] = str(error)
        return self.complete(status="failed")
