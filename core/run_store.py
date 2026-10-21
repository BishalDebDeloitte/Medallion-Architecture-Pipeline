"""Persist PipelineState to disk so a run survives a Streamlit rerun, refresh or restart."""

import logging
import os
from datetime import datetime, timezone
from typing import Any

from core import paths
from core.state import PipelineState

logger = logging.getLogger(__name__)


def save_state(state: PipelineState) -> None:
    """Write the state atomically (temp file then rename) so a crash never leaves a torn file."""
    destination = paths.state_file(state.run_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    os.replace(temporary, destination)


def load_state(run_id: str) -> PipelineState | None:
    """Load a run's state, or None if it does not exist or cannot be read."""
    try:
        return PipelineState.model_validate_json(paths.state_file(run_id).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as error:
        logger.warning("Could not load state for run %s: %s", run_id, error)
        return None


def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    """Summarise saved runs, newest first (for a 'resume a run' picker)."""
    runs = []
    for path in sorted(paths.RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        state = load_state(path.stem)
        if state is None or state.run_id != path.stem:      # only real run files
            continue
        runs.append(
            {
                "run_id": state.run_id, "status": state.status, "phase": state.current_phase,
                "business_intent": state.business_intent,
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            }
        )
    return runs
