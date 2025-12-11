"""
Self-auditing logic for proof verification (Poetiq pattern).

Implements autonomous termination decisions based on:
- Divergence: goal complexity increasing
- Oscillation: same error pattern repeating
- Budget exhaustion: iteration/token limits

Reference: docs/claude-helpers/resources/POETIQ_deep_dive.md Section 2.3
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AuditConfig:
    """Configuration for self-auditing."""

    max_iterations: int = 4
    max_tokens: int = 50000  # Budget-conscious default
    complexity_growth_threshold: float = 1.5  # 50% growth = divergence
    oscillation_window: int = 3  # Check last N errors for repetition
    min_iterations_before_divergence_check: int = 2


@dataclass
class AuditState:
    """State tracked across verification attempts."""

    attempt: int = 0
    error_history: list[str] = field(default_factory=list)
    goal_complexity_history: list[int] = field(default_factory=list)
    token_used: int = 0
    successful_tactics: list[str] = field(default_factory=list)
    failed_tactics: list[str] = field(default_factory=list)


class SelfAuditingController:
    """
    Controller for autonomous termination decisions.

    Implements Poetiq's self-auditing pattern:
    - Monitor progress and autonomously decide when to terminate
    - Average <2 requests per problem despite allowing more attempts
    """

    def __init__(self, config: Optional[AuditConfig] = None):
        self.config = config or AuditConfig()
        self.state = AuditState()

    def record_attempt(
        self,
        error: Optional[str],
        goal_complexity: int,
        tokens: int = 0,
        tactic: str = "",
        success: bool = False,
    ) -> None:
        """
        Record an attempt for auditing.

        Args:
            error: Error message (None if successful)
            goal_complexity: Estimated complexity of the goal
            tokens: Tokens used in this attempt
            tactic: The tactic that was tried
            success: Whether this attempt succeeded
        """
        self.state.attempt += 1

        if error:
            self.state.error_history.append(self._normalize_error(error))
            if tactic:
                self.state.failed_tactics.append(tactic)
        elif tactic:
            self.state.successful_tactics.append(tactic)

        self.state.goal_complexity_history.append(goal_complexity)
        self.state.token_used += tokens

    def should_continue(self) -> tuple[bool, str]:
        """
        Determine if verification should continue.

        Returns:
            (should_continue, reason if stopping)
        """
        # Check iteration limit
        if self.state.attempt >= self.config.max_iterations:
            return False, "max_iterations_reached"

        # Check token budget
        if self.state.token_used >= self.config.max_tokens:
            return False, "token_budget_exhausted"

        # Check divergence (complexity growing) - only after minimum attempts
        if self.state.attempt >= self.config.min_iterations_before_divergence_check:
            if self._detect_divergence():
                return False, "divergence_detected"

        # Check oscillation (same errors repeating)
        if self._detect_oscillation():
            return False, "oscillation_detected"

        return True, ""

    def _detect_divergence(self) -> bool:
        """
        Detect if goal complexity is increasing significantly.

        Poetiq pattern: Stop if goal complexity grows beyond threshold,
        indicating we're moving away from a solution.
        """
        history = self.state.goal_complexity_history
        if len(history) < 2:
            return False

        initial = history[0]
        current = history[-1]

        if initial == 0:
            return False

        growth = current / initial
        return growth > self.config.complexity_growth_threshold

    def _detect_oscillation(self) -> bool:
        """
        Detect if errors are repeating in a pattern.

        Poetiq pattern: If the same normalized error repeats N times,
        we're likely stuck in a loop.
        """
        errors = self.state.error_history
        window = self.config.oscillation_window

        if len(errors) < window:
            return False

        recent = errors[-window:]
        # Check if all recent errors are identical (after normalization)
        return len(set(recent)) == 1

    def _normalize_error(self, error: str) -> str:
        """
        Normalize error for comparison (remove line numbers, etc.).

        Makes error comparison robust to trivial differences.
        """
        # Remove line:col references
        normalized = re.sub(r":\d+:\d+:", "::", error)
        # Remove file paths
        normalized = re.sub(r"/[^\s:]+\.lean", "FILE.lean", normalized)
        # Remove specific variable names (often change between attempts)
        normalized = re.sub(r"\b[a-z]_\d+\b", "VAR", normalized)
        # Truncate for comparison
        return normalized.strip()[:200]

    def get_summary(self) -> dict:
        """Get a summary of the audit state for logging."""
        return {
            "attempts": self.state.attempt,
            "tokens_used": self.state.token_used,
            "error_count": len(self.state.error_history),
            "successful_tactics": len(self.state.successful_tactics),
            "failed_tactics": len(self.state.failed_tactics),
            "complexity_trend": self._get_complexity_trend(),
        }

    def _get_complexity_trend(self) -> str:
        """Get human-readable complexity trend."""
        history = self.state.goal_complexity_history
        if len(history) < 2:
            return "insufficient_data"

        initial = history[0]
        current = history[-1]

        if initial == 0:
            return "unknown"

        ratio = current / initial
        if ratio > 1.5:
            return "diverging"
        elif ratio < 0.7:
            return "converging"
        else:
            return "stable"

    def reset(self) -> None:
        """Reset the audit state for a new verification task."""
        self.state = AuditState()


def estimate_goal_complexity(error_message: str) -> int:
    """
    Estimate goal complexity from error message.

    Heuristic: count unique identifiers and nesting depth.
    This is a simple proxy for how "complex" the remaining goal is.

    Args:
        error_message: Error message or goal state text

    Returns:
        Complexity score (higher = more complex)
    """
    if not error_message:
        return 0

    # Count identifiers
    identifiers = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", error_message)
    unique_idents = len(set(identifiers))

    # Count nesting (brackets, parens)
    nesting = (
        error_message.count("(")
        + error_message.count("{")
        + error_message.count("[")
    )

    # Count arrows (function types add complexity)
    arrows = error_message.count("→") + error_message.count("->")

    return unique_idents + nesting * 2 + arrows * 3


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a text string.

    Simple heuristic: ~4 characters per token on average.

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    return len(text) // 4
