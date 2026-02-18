"""
Rich-based console output for per-attempt verification status.

Pure presentation layer -- no agent code dependencies.
Called from nodes.py and agent.py during the iterative loop.
"""

from rich.console import Console
from rich.text import Text

console = Console()


def print_attempt_start(
    sorry_idx: int,
    attempt: int,
    max_attempts: int,
    goal_state: str,
    verbose: bool = False,
) -> None:
    """Print start of an attempt: shows goal state (truncated) and attempt number."""
    header = Text()
    header.append(f"  sorry #{sorry_idx} ", style="bold")
    header.append(f"attempt {attempt}/{max_attempts}", style="dim")
    console.print(header)

    if goal_state:
        preview = goal_state[:120].replace("\n", " ")
        if len(goal_state) > 120:
            preview += "..."
        goal_line = Text()
        goal_line.append("  goal: ", style="dim")
        goal_line.append(preview, style="cyan")
        console.print(goal_line)

    if verbose and goal_state and len(goal_state) > 120:
        console.print(f"  [dim]{goal_state}[/dim]")


def print_sliding_window(
    attempt_history: list[dict],
    window_size: int = 3,
) -> None:
    """Print the sliding window of recent attempts (verbose mode only).

    Shows the last ``window_size`` attempts with snippet, error, and
    suggestion so the user can see what the LLM is learning from.
    """
    if not attempt_history:
        return
    recent = attempt_history[-window_size:]
    console.print(
        "  [dim]--- sliding window "
        f"(last {len(recent)}) ---[/dim]"
    )
    for a in recent:
        snippet_preview = a.get("snippet", "?")[:80]
        error = a.get("normalized_error", "?")[:80]
        suggestion = a.get("suggestion", "")[:60]
        line = Text()
        line.append(f"    #{a.get('number', '?')} ", style="dim")
        line.append(f"`{snippet_preview}`", style="cyan")
        console.print(line)
        console.print(f"      err: {error}", style="red dim")
        if suggestion:
            console.print(
                f"      hint: {suggestion}",
                style="yellow dim",
            )
        rag = a.get("rag_suggestions", [])
        if rag:
            console.print(
                f"      did you mean: {', '.join(rag[:3])}",
                style="green dim",
            )
    console.print("  [dim]--- end window ---[/dim]")


def print_attempt_trying(
    sorry_idx: int,
    attempt: int,
    snippet: str,
    verbose: bool = False,
) -> None:
    """Print the snippet being tried. Default: short preview. Verbose: full snippet."""
    preview = snippet[:80].replace("\n", " ; ")
    if len(snippet) > 80 and not verbose:
        preview += "..."
    line = Text()
    line.append("  trying: ", style="dim")
    line.append(preview)
    console.print(line)

    if verbose and len(snippet) > 80:
        for sl in snippet.split("\n"):
            console.print(f"    {sl}", style="dim")


def print_attempt_success(
    sorry_idx: int,
    attempt: int,
    max_attempts: int,
    elapsed: float,
) -> None:
    """Print green checkmark success line."""
    line = Text()
    line.append("  \u2713 ", style="green bold")
    line.append(f"sorry #{sorry_idx} solved ", style="green bold")
    line.append(f"(attempt {attempt}/{max_attempts}, {elapsed:.1f}s)", style="dim")
    console.print(line)


def print_attempt_failure(
    sorry_idx: int,
    attempt: int,
    max_attempts: int,
    error_summary: str,
    verbose: bool = False,
) -> None:
    """Print red X failure line. Verbose adds Lean's full error."""
    summary = error_summary[:100] if not verbose else error_summary[:300]
    line = Text()
    line.append("  \u2717 ", style="red bold")
    line.append(f"sorry #{sorry_idx} attempt {attempt}/{max_attempts}: ", style="red")
    line.append(summary, style="dim")
    console.print(line)


def print_sorry_exhausted(
    sorry_idx: int,
    max_attempts: int,
    tried_summary: str,
) -> None:
    """Print dot exhausted line with summary of what was tried."""
    line = Text()
    line.append("  \u2022 ", style="yellow")
    line.append(f"sorry #{sorry_idx} exhausted ", style="yellow")
    line.append(f"({max_attempts} attempts): ", style="dim")
    line.append(tried_summary[:120], style="dim")
    console.print(line)


def print_session_summary(
    solved: int,
    total: int,
    elapsed: float,
) -> None:
    """Print final session summary (N/M sorries solved in Xs)."""
    console.print()
    line = Text()
    if solved > 0:
        line.append(f"  {solved}/{total} sorries solved", style="green bold")
    else:
        line.append(f"  0/{total} sorries solved", style="red")
    line.append(f" in {elapsed:.1f}s", style="dim")
    console.print(line)
