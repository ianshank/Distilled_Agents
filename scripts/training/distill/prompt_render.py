"""Render harness chat strings. Keep enhanced_system/harness/prompt_render.py in sync."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence


def join_thought_action(thought: str, action: str) -> str:
    """Join thought and action the way assistant turns are stored."""
    return " ".join(part for part in (thought.strip(), action.strip()) if part)


def render_messages(messages: Sequence[dict[str, str]]) -> str:
    """Join role-tagged turns without a trailing generate suffix."""
    return "\n".join(
        f"{item.get('role') or 'user'}: {item.get('content') or ''}" for item in messages
    )


def render_prompt(messages: Sequence[dict[str, str]], prefix: Optional[str] = None) -> str:
    """Match TransformersBackend generate prompts (`assistant:` plus optional FTP)."""
    body = render_messages(messages)
    tail = "assistant:"
    if prefix:
        tail = f"{tail} {prefix}"
    if body:
        return f"{body}\n{tail}"
    return tail


def messages_from_steps(
    task: str,
    steps: Iterable[dict[str, Any]],
    *,
    instruction: str = "",
) -> list[dict[str, str]]:
    """Rebuild the runtime message list from stored trajectory steps."""
    messages: list[dict[str, str]] = []
    text = instruction.strip()
    if text:
        messages.append({"role": "system", "content": text})
    messages.append({"role": "user", "content": task})
    for step in steps:
        fault = str(step.get("fault") or "")
        observation = str(step.get("observation") or "")
        if fault:
            messages.append({"role": "user", "content": f"{fault}: {observation}"})
            continue
        content = join_thought_action(str(step.get("thought") or ""), str(step.get("action") or ""))
        if content:
            messages.append({"role": "assistant", "content": content})
        tool_id = str(step.get("tool_id") or "")
        if observation and tool_id != "final_answer":
            messages.append({"role": "user", "content": observation})
    return messages


def supervised_spans(
    task: str,
    steps: Iterable[dict[str, Any]],
    *,
    instruction: str = "",
) -> list[tuple[str, bool]]:
    """Yield (text, supervised) chunks matching render_messages(messages_from_steps)."""
    chunks: list[tuple[str, bool]] = []
    first = True

    def add_message(role: str, content: str, supervised: bool) -> None:
        nonlocal first
        prefix = "" if first else "\n"
        first = False
        chunks.append((f"{prefix}{role}: ", False))
        chunks.append((content, supervised))

    text = instruction.strip()
    if text:
        add_message("system", text, False)
    add_message("user", task, False)
    for step in steps:
        fault = str(step.get("fault") or "")
        observation = str(step.get("observation") or "")
        if fault:
            add_message("user", f"{fault}: {observation}", False)
            continue
        content = join_thought_action(str(step.get("thought") or ""), str(step.get("action") or ""))
        if content:
            add_message("assistant", content, True)
        tool_id = str(step.get("tool_id") or "")
        if observation and tool_id != "final_answer":
            add_message("user", observation, False)
    return chunks
