"""Extract Lean goal state at sorry locations using LSP."""

import re

from . import SorryLocation, LeanGoal


def get_goal_at_sorry(sorry: SorryLocation) -> LeanGoal | None:
    """
    Get the proof goal at a sorry location.

    This is a placeholder that returns None. In production, this would use
    the Lean LSP MCP tool `mcp__lean-lsp__lean_goal` to get the actual goal.

    For CLI usage, the agent will call MCP tools directly rather than
    going through this function.

    Args:
        sorry: SorryLocation with file path and position

    Returns:
        LeanGoal with target and hypotheses, or None if unavailable
    """
    # Note: The actual goal extraction happens via MCP tools when the agent
    # processes each sorry. This function exists for potential offline/cached
    # goal extraction in the future.
    return None


def parse_goal_response(goal_text: str) -> LeanGoal | None:
    """
    Parse a goal response from the Lean LSP into a LeanGoal object.

    Args:
        goal_text: Raw goal text from lean_goal MCP tool

    Returns:
        Parsed LeanGoal or None if parsing fails
    """
    if not goal_text or "no goals" in goal_text.lower():
        return None

    lines = goal_text.strip().split("\n")
    hypotheses = []
    target = ""

    # Goal format is typically:
    # hyp1 : Type1
    # hyp2 : Type2
    # ⊢ target_type
    #
    # or with case splits:
    # case name
    # hyp1 : Type1
    # ⊢ target_type

    in_hypotheses = True

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip case labels
        if line.startswith("case "):
            continue

        # Target line starts with ⊢ (turnstile)
        if line.startswith("⊢"):
            target = line[1:].strip()
            in_hypotheses = False
            continue

        # Hypothesis line: name : type
        if in_hypotheses and " : " in line:
            parts = line.split(" : ", 1)
            if len(parts) == 2:
                name = parts[0].strip()
                typ = parts[1].strip()
                hypotheses.append({"name": name, "type": typ})

    if not target:
        return None

    return LeanGoal(target_type=target, hypotheses=hypotheses)


def format_goal_for_prompt(goal: LeanGoal) -> str:
    """
    Format a LeanGoal for inclusion in an LLM prompt.

    Args:
        goal: The LeanGoal to format

    Returns:
        Formatted string representation
    """
    lines = []

    if goal.hypotheses:
        lines.append("Hypotheses:")
        for hyp in goal.hypotheses:
            lines.append(f"  {hyp['name']} : {hyp['type']}")
        lines.append("")

    lines.append(f"Goal: ⊢ {goal.target_type}")

    return "\n".join(lines)
