"""Parse/schema majority vote (not execution-consistent Kang SAG)."""

from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

from enhanced_system.harness.dispatch import DispatchError, parse_action, split_thought_action
from enhanced_system.harness.protocols import ModelBackend


def generate_action(
    backend: ModelBackend,
    messages: Sequence[dict[str, str]],
    allowed: set[str],
    *,
    samples: int,
    temperature: Optional[float],
    prefix: Optional[str] = None,
) -> tuple[str, str]:
    """Return (raw_text, parsed_tool_id) for the majority-valid sample."""
    count = samples if samples and samples > 0 else 1
    raw_list = backend.generate(messages, prefix=prefix, n=count, temperature=temperature)
    valid: list[str] = []
    actions: list[str] = []
    for item in raw_list:
        try:
            _thought, action = split_thought_action(item, allowed)
            parse_action(action, allowed)
            valid.append(item)
            actions.append(action)
        except DispatchError:
            continue
    if not valid:
        raise DispatchError("no parse-valid action among samples")
    winner_action, _ = Counter(actions).most_common(1)[0]
    winner = valid[actions.index(winner_action)]
    tool_id, _ = parse_action(winner_action, allowed)
    return winner, tool_id
