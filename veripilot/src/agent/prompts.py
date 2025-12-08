"""
Prompt templates for the Prover Agent.

Provides model-specific prompt templates for Lean proof generation,
with special handling for Aeneas-generated verification code.
"""

from typing import Optional

from parser import SorryLocation
from .context_formatter import format_error_context


# System prompts for different models
SYSTEM_PROMPT_DEFAULT = """You are an expert in the Lean 4 theorem prover, specializing in program verification.

Your task is to generate valid Lean 4 proof tactics to replace `sorry` placeholders.

Key principles:
- Use Lean automation (grind, simp, omega) whenever possible for stability
- Follow the unfold → progress* → grind pattern for Aeneas code
- Return ONLY the proof tactics, no explanations or markdown
- Do not introduce axioms
- Prefer concise, robust proofs over verbose manual ones"""

SYSTEM_PROMPT_GEMINI = """You are an expert Lean 4 theorem prover for program verification.

TASK: Generate proof tactics to replace a `sorry` placeholder.

RULES:
1. Return ONLY Lean 4 tactic code, no markdown or explanations
2. Use automation (grind, simp, omega, scalar_tac) over manual proofs
3. For Aeneas code: unfold → progress* → handle side goals
4. No axioms allowed
5. Keep proofs concise

OUTPUT FORMAT:
```
tactic1
tactic2
...
```"""

SYSTEM_PROMPT_CLAUDE = """You are an expert Lean 4 theorem prover. Generate proof tactics to fill a sorry placeholder.

Requirements:
- Return only tactic code (no markdown, no explanation)
- Prefer automation: grind, simp, omega, scalar_tac
- For Aeneas/verification: unfold → progress* → automation
- No axioms"""


def build_system_prompt(model: str = "default") -> str:
    """
    Build the system prompt for a specific model.

    Args:
        model: Model identifier (gemini, claude, aristotle, default)

    Returns:
        System prompt string
    """
    prompts = {
        "gemini": SYSTEM_PROMPT_GEMINI,
        "claude": SYSTEM_PROMPT_CLAUDE,
        "default": SYSTEM_PROMPT_DEFAULT,
    }
    # Aristotle doesn't use system prompts (file-based API)
    return prompts.get(model, SYSTEM_PROMPT_DEFAULT)


def build_user_prompt(
    sorry: SorryLocation,
    context: str,
    goal: Optional[str] = None,
) -> str:
    """
    Build the user prompt for proof generation.

    Args:
        sorry: The sorry location
        context: Formatted context from context_formatter
        goal: Optional goal state text

    Returns:
        User prompt string
    """
    lines = [
        f"# Proof Task",
        "",
        f"Replace the `sorry` at line {sorry.line} in theorem `{sorry.theorem_name}` with a valid proof.",
        "",
    ]

    # Add goal if available
    if goal:
        lines.extend([
            "## Current Goal",
            "```",
            goal,
            "```",
            "",
        ])

    # Add formatted context
    lines.append(context)

    # Add response format instruction
    lines.extend([
        "",
        "## Your Response",
        "",
        "Return ONLY the tactic code to replace the sorry. No markdown code fences, no explanations.",
        "If multiple tactics are needed, put each on its own line.",
        "",
        "Example response:",
        "unfold my_function",
        "progress",
        "grind",
    ])

    return "\n".join(lines)


def build_retry_prompt(
    sorry: SorryLocation,
    context: str,
    prev_proof: str,
    error: str,
    attempt: int,
    goal: Optional[str] = None,
) -> str:
    """
    Build a retry prompt after a failed attempt.

    Args:
        sorry: The sorry location
        context: Formatted context
        prev_proof: Previous proof attempt
        error: Error message from Lean
        attempt: Current attempt number (2, 3, or 4)
        goal: Optional goal state

    Returns:
        User prompt for retry
    """
    lines = [
        f"# Proof Task (Attempt {attempt}/4)",
        "",
        f"Replace the `sorry` at line {sorry.line} in theorem `{sorry.theorem_name}`.",
        "",
    ]

    # Add error context
    lines.append(format_error_context(error, prev_proof))
    lines.append("")

    # Add goal if available
    if goal:
        lines.extend([
            "## Current Goal",
            "```",
            goal,
            "```",
            "",
        ])

    # Add original context
    lines.append(context)

    # Add attempt-specific guidance
    if attempt == 2:
        lines.extend([
            "",
            "## Retry Guidance",
            "- Try simpler tactics first",
            "- Check lemma names carefully",
            "- Consider using `decide` for simple propositions",
        ])
    elif attempt == 3:
        lines.extend([
            "",
            "## Retry Guidance",
            "- Try a completely different approach",
            "- Use more aggressive automation: `progress* ; grind`",
            "- Check if you need to unfold more definitions",
        ])
    elif attempt >= 4:
        lines.extend([
            "",
            "## Final Attempt Guidance",
            "- Focus on the core goal only",
            "- Try breaking into smaller subgoals with `have`",
            "- Consider if the theorem is even provable",
        ])

    lines.extend([
        "",
        "## Your Response",
        "",
        "Return ONLY the tactic code. No markdown, no explanations.",
    ])

    return "\n".join(lines)


def extract_proof_from_response(response: str) -> str:
    """
    Extract clean proof tactics from LLM response.

    Handles:
    - Markdown code fences
    - Explanatory text before/after
    - Extra whitespace

    Args:
        response: Raw LLM response

    Returns:
        Clean proof tactic code
    """
    text = response.strip()

    # Remove markdown code fences
    if "```" in text:
        # Extract content between first and last ```
        parts = text.split("```")
        if len(parts) >= 3:
            # Take the first code block
            code = parts[1]
            # Remove language tag if present (e.g., "lean")
            if code.startswith("lean"):
                code = code[4:]
            text = code.strip()
        elif len(parts) == 2:
            # Only one fence, take content after it
            text = parts[1].strip()
            if text.startswith("lean"):
                text = text[4:].strip()

    # Remove common preamble phrases
    prefixes_to_remove = [
        "here is the proof:",
        "here are the tactics:",
        "the proof is:",
        "proof:",
        "tactics:",
    ]
    text_lower = text.lower()
    for prefix in prefixes_to_remove:
        if text_lower.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    # Remove trailing explanations (usually after double newline)
    if "\n\n" in text:
        parts = text.split("\n\n")
        # Keep only the first part if it looks like code
        first_part = parts[0].strip()
        if first_part and not first_part[0].isupper():  # Tactics don't start with capitals
            text = first_part

    return text.strip()
