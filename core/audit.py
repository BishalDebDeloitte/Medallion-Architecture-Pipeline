"""JSONL audit logging for pipeline runs."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.paths import AUDIT_DIR, audit_file


class AuditLogger:
    """Append and retrieve audit events for one pipeline run."""

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or str(uuid.uuid4())
        self.log_path = audit_file(self.run_id, AUDIT_DIR)

    def log(self, agent: str, action: str, **kwargs: Any) -> None:
        """Append one timestamped event to this run's JSONL file."""
        entry = {
            **kwargs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "agent": agent,
            "action": action,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_logs(self) -> list[dict[str, Any]]:
        """Return all events for this run, or an empty list if none exist."""
        if not self.log_path.exists():
            return []

        with self.log_path.open("r", encoding="utf-8") as log_file:
            return [json.loads(line) for line in log_file if line.strip()]
