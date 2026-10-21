"""Test doubles shared by agent tests."""

import json
from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


def tool_call(name: str, call_id: str = "call_1", **args: Any) -> dict[str, Any]:
    """Build a LangChain tool-call dict."""
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def ai(content: str = "", *tool_calls: dict[str, Any]) -> AIMessage:
    """Build an AIMessage with optional tool calls."""
    return AIMessage(content=content, tool_calls=list(tool_calls))


def last_tool_result(messages: list[BaseMessage]) -> dict[str, Any]:
    """Parse the JSON content of the most recent ToolMessage."""
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            return json.loads(message.content)
    raise AssertionError("No tool result in the conversation yet")


class ScriptedChatModel(BaseChatModel):
    """A chat model that replays scripted replies and works with create_agent.

    Each script item is an AIMessage or a callable(messages) -> AIMessage, so a
    reply can depend on earlier tool results.
    """

    script: list[Any]
    _position: int = PrivateAttr(default=0)
    _seen: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    @property
    def seen(self) -> list[list[BaseMessage]]:
        """The message list received on each model call."""
        return self._seen

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None,
        run_manager: Any = None, **kwargs: Any,
    ) -> ChatResult:
        self._seen.append(list(messages))
        if self._position >= len(self.script):
            raise AssertionError("ScriptedChatModel ran out of scripted replies")
        reply = self.script[self._position]
        self._position += 1
        message = reply(messages) if isinstance(reply, Callable) else reply
        return ChatResult(generations=[ChatGeneration(message=message)])
