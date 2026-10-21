"""Helpers shared by every agent's tools: structured JSON results, never raising."""

import functools
import json
import logging
from collections.abc import Callable, Iterable
from typing import Any

logger = logging.getLogger(__name__)


def ok(**fields: Any) -> str:
    """Return a structured success result as a JSON string."""
    return json.dumps({"status": "success", **fields}, ensure_ascii=False, default=str)


def err(message: str, **fields: Any) -> str:
    """Return a structured error result as a JSON string."""
    return json.dumps({"status": "error", "message": message, **fields}, ensure_ascii=False, default=str)


def structured_tool(tool: Callable[..., str]) -> Callable[..., str]:
    """Make a tool return {"status": "error", ...} instead of raising.

    The wrapper keeps the original name, signature and docstring, which
    LangChain uses as the tool name, argument schema and description.
    """
    @functools.wraps(tool)
    def wrapped(*args: Any, **kwargs: Any) -> str:
        try:
            result = tool(*args, **kwargs)
            if isinstance(result, dict):
                return ok(**result) if "status" not in result else json.dumps(result, default=str)
            if not isinstance(result, str):
                raise TypeError(f"{tool.__name__} must return a JSON string, got {type(result).__name__}")
            return result
        except Exception as error:
            logger.warning("Tool %s failed: %s", tool.__name__, error)
            return err(str(error))

    return wrapped


class InspectGate:
    """Enforce the inspect-then-execute pattern inside one agent invocation."""

    def __init__(self, inspect_tool_name: str) -> None:
        self.inspect_tool_name = inspect_tool_name
        self._inspected = False

    def mark(self) -> None:
        """Record that the lightweight inspect tool ran."""
        self._inspected = True

    def require(self) -> None:
        """Raise if the execute tool is called before the inspect tool."""
        if not self._inspected:
            raise ValueError(
                f"Call {self.inspect_tool_name} first and form a plan before running this tool."
            )


def resolve_allowed_paths(
    requested: Iterable[str] | None, allowed: Iterable[str], label: str = "file",
) -> list[str]:
    """Return the requested paths if all are allowed; default to every allowed path.

    Agents may only touch paths handed to them explicitly, so a hallucinated
    path produces an error that lists the valid ones.
    """
    allowed_list = list(allowed)
    if not requested:
        return allowed_list
    normalized = {str(path): path for path in allowed_list}
    unknown = [str(path) for path in requested if str(path) not in normalized]
    if unknown:
        raise ValueError(
            f"Unknown {label} path(s): {unknown}. Use only these paths: {allowed_list}"
        )
    return [str(path) for path in requested]
