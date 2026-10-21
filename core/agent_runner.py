"""Run one specialist ReAct agent over a fixed set of tools."""

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

DEFAULT_RECURSION_LIMIT = 20


def message_text(content: Any) -> str:
    """Flatten message content (a string or a list of content blocks) to text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        ).strip()
    return str(content or "").strip()


def final_text(messages: Sequence[BaseMessage]) -> str:
    """Return the text of the last AI message, or an empty string."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message_text(message.content):
            return message_text(message.content)
    return ""


def invoke_agent(
    tools: Sequence[Callable[..., str]],
    system_prompt: str,
    user_message: str,
    model: Any | None = None,
    agent_factory: Callable[..., Any] | None = None,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    history: Sequence[BaseMessage] | None = None,
) -> list[BaseMessage]:
    """Build a ReAct agent restricted to `tools`, run it, and return its messages.

    `model` and `agent_factory` are injectable so tests run offline. The
    recursion limit stops a confused agent from looping forever. Pass the
    messages of an earlier call as `history` to continue that conversation
    (for example to ask the agent to correct its final answer); the result
    then contains the history followed by the new messages.
    """
    if agent_factory is None:
        from langchain.agents import create_agent

        agent_factory = create_agent
    if model is None:
        from core.llm import create_chat_model

        model = create_chat_model()
    agent = agent_factory(model, list(tools), system_prompt=system_prompt)
    result = agent.invoke(
        {"messages": [*(history or []), {"role": "user", "content": user_message}]},
        config={"recursion_limit": recursion_limit},
    )
    return list(result.get("messages", []))
