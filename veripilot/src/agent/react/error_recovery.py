"""
OpenManus-style error recovery for ReAct proof verification agent.

Implements intelligent error classification and recovery strategies:
- ErrorSeverity: Classifies errors as FATAL, RECOVERABLE, or TRANSIENT
- RecoveryStrategy: Defines recovery actions (RETRY_SAME, TRY_ALTERNATIVE, etc.)
- ErrorRecoveryController: Maps error types to recovery strategies
- TacticModifier: Generates modified tactics based on recovery strategy

Based on OpenManus error recovery patterns from:
docs/claude-helpers/resources/ROMA_et_al_veriplot.md
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """Severity classification for Lean errors."""

    FATAL = "fatal"           # Cannot recover: syntax errors, missing imports
    RECOVERABLE = "recoverable"  # Can try alternatives: type_mismatch, tactic_failed
    TRANSIENT = "transient"   # May succeed on retry: timeout, resource limits


class RecoveryStrategy(str, Enum):
    """Recovery strategies for different error types."""

    RETRY_SAME = "retry_same"           # Same tactic with minor variation (add try wrapper)
    TRY_ALTERNATIVE = "try_alternative"  # Different tactic from same family
    UNFOLD_MORE = "unfold_more"         # Expand definitions before retry
    SIMPLIFY_FIRST = "simplify_first"   # Run simp before main tactic
    BACKTRACK = "backtrack"             # Restore to earlier checkpoint
    ESCALATE = "escalate"               # Flag for human review
    ABORT = "abort"                     # Fatal error, no recovery possible


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt for tracking and analysis."""

    step: int
    error_type: str
    severity: str
    strategy: str
    original_tactic: str
    modified_tactic: str
    success: bool = False
    timestamp: float = 0.0


@dataclass
class RecoveryContext:
    """Context for making recovery decisions."""

    tried_tactics: list[str] = field(default_factory=list)
    definitions_to_unfold: list[str] = field(default_factory=list)
    successful_tactics: list[str] = field(default_factory=list)
    error_content: str = ""
    goal_state: str = ""
    attempt_count: int = 0


class ErrorRecoveryController:
    """
    OpenManus-style error recovery controller.

    Maps error types from error_normalizer.py to recovery strategies,
    implements multi-stage recovery, and tracks recovery attempts.

    Multi-stage recovery:
    - Stage 0: Primary strategy (first attempt at this error)
    - Stage 1: Fallback strategy (second attempt)
    - Stage 2+: Escalate to backtrack or abort
    """

    # Error type -> (Severity, Primary Strategy, Fallback Strategy)
    ERROR_STRATEGY_MAP: dict[str, tuple[ErrorSeverity, RecoveryStrategy, RecoveryStrategy]] = {
        # Type errors - try alternative tactics or unfold definitions
        "type_mismatch": (
            ErrorSeverity.RECOVERABLE,
            RecoveryStrategy.TRY_ALTERNATIVE,
            RecoveryStrategy.UNFOLD_MORE,
        ),
        "application_mismatch": (
            ErrorSeverity.RECOVERABLE,
            RecoveryStrategy.UNFOLD_MORE,
            RecoveryStrategy.TRY_ALTERNATIVE,
        ),

        # Unknown identifiers - try alternatives or backtrack
        "unknown_identifier": (
            ErrorSeverity.RECOVERABLE,
            RecoveryStrategy.TRY_ALTERNATIVE,
            RecoveryStrategy.BACKTRACK,
        ),
        "unknown_tactic": (
            ErrorSeverity.RECOVERABLE,
            RecoveryStrategy.TRY_ALTERNATIVE,
            RecoveryStrategy.TRY_ALTERNATIVE,
        ),

        # Tactic failures - try alternatives or simplify first
        "tactic_failed": (
            ErrorSeverity.RECOVERABLE,
            RecoveryStrategy.TRY_ALTERNATIVE,
            RecoveryStrategy.SIMPLIFY_FIRST,
        ),
        "unsolved_goals": (
            ErrorSeverity.RECOVERABLE,
            RecoveryStrategy.RETRY_SAME,
            RecoveryStrategy.TRY_ALTERNATIVE,
        ),

        # Motive errors (induction/cases) - try rcases or different approach
        "motive_error": (
            ErrorSeverity.RECOVERABLE,
            RecoveryStrategy.TRY_ALTERNATIVE,
            RecoveryStrategy.BACKTRACK,
        ),

        # Syntax errors are fatal - can't recover
        "syntax_error": (
            ErrorSeverity.FATAL,
            RecoveryStrategy.ABORT,
            RecoveryStrategy.ABORT,
        ),

        # Timeouts are transient - may succeed on retry
        "timeout": (
            ErrorSeverity.TRANSIENT,
            RecoveryStrategy.RETRY_SAME,
            RecoveryStrategy.TRY_ALTERNATIVE,
        ),

        # Sorry indicates incomplete proof - try alternatives
        "sorry": (
            ErrorSeverity.RECOVERABLE,
            RecoveryStrategy.TRY_ALTERNATIVE,
            RecoveryStrategy.TRY_ALTERNATIVE,
        ),

        # Unknown errors - conservative recovery
        "unknown": (
            ErrorSeverity.RECOVERABLE,
            RecoveryStrategy.TRY_ALTERNATIVE,
            RecoveryStrategy.ESCALATE,
        ),
    }

    def __init__(self, max_recovery_per_error: int = 2):
        """
        Initialize error recovery controller.

        Args:
            max_recovery_per_error: Maximum recovery attempts per error type
        """
        self.max_recovery_per_error = max_recovery_per_error
        self.recovery_history: list[RecoveryAttempt] = []

    def classify_error(self, error_type: str) -> tuple[ErrorSeverity, RecoveryStrategy]:
        """
        Get severity and primary strategy for an error type.

        Args:
            error_type: Error type from error_normalizer (e.g., "type_mismatch")

        Returns:
            Tuple of (severity, primary_strategy)
        """
        entry = self.ERROR_STRATEGY_MAP.get(
            error_type,
            self.ERROR_STRATEGY_MAP["unknown"]
        )
        return entry[0], entry[1]

    def get_recovery_strategy(
        self,
        error_type: str,
        recovery_stage: int,
        context: Optional[RecoveryContext] = None,
    ) -> RecoveryStrategy:
        """
        Determine recovery strategy based on error and history.

        Multi-stage recovery:
        - Stage 0: Primary strategy
        - Stage 1: Fallback strategy
        - Stage 2+: Backtrack or escalate

        Args:
            error_type: Error type from error_normalizer
            recovery_stage: Current stage (0=first attempt, 1=fallback, 2+=escalate)
            context: Optional recovery context with tried tactics, etc.

        Returns:
            Recommended RecoveryStrategy
        """
        entry = self.ERROR_STRATEGY_MAP.get(
            error_type,
            self.ERROR_STRATEGY_MAP["unknown"]
        )
        severity, primary, fallback = entry

        # Fatal errors always abort
        if severity == ErrorSeverity.FATAL:
            return RecoveryStrategy.ABORT

        # Stage-based strategy selection
        if recovery_stage == 0:
            strategy = primary
        elif recovery_stage == 1:
            strategy = fallback
        else:
            # Beyond fallback - escalate
            strategy = RecoveryStrategy.BACKTRACK

        # Context-aware adjustments
        if context:
            strategy = self._adjust_for_context(strategy, context)

        return strategy

    def _adjust_for_context(
        self,
        strategy: RecoveryStrategy,
        context: RecoveryContext,
    ) -> RecoveryStrategy:
        """
        Adjust strategy based on context.

        - If all alternatives exhausted, backtrack
        - If successful tactics exist, try variations of those
        """
        # If we've tried many tactics, consider backtracking
        if len(context.tried_tactics) >= 5:
            return RecoveryStrategy.BACKTRACK

        # If we have successful tactics from earlier, prefer variations
        if context.successful_tactics and strategy == RecoveryStrategy.TRY_ALTERNATIVE:
            # Keep TRY_ALTERNATIVE but TacticModifier will use successful_tactics
            pass

        return strategy

    def record_attempt(
        self,
        step: int,
        error_type: str,
        strategy: RecoveryStrategy,
        original_tactic: str,
        modified_tactic: str,
        success: bool = False,
    ) -> RecoveryAttempt:
        """Record a recovery attempt for tracking."""
        import time

        severity, _ = self.classify_error(error_type)
        attempt = RecoveryAttempt(
            step=step,
            error_type=error_type,
            severity=severity.value,
            strategy=strategy.value,
            original_tactic=original_tactic,
            modified_tactic=modified_tactic,
            success=success,
            timestamp=time.time(),
        )
        self.recovery_history.append(attempt)
        return attempt

    def get_recovery_stats(self) -> dict[str, Any]:
        """Get statistics about recovery attempts."""
        if not self.recovery_history:
            return {"total": 0, "success_rate": 0.0}

        total = len(self.recovery_history)
        successes = sum(1 for a in self.recovery_history if a.success)

        by_strategy: dict[str, dict[str, int]] = {}
        for attempt in self.recovery_history:
            if attempt.strategy not in by_strategy:
                by_strategy[attempt.strategy] = {"total": 0, "success": 0}
            by_strategy[attempt.strategy]["total"] += 1
            if attempt.success:
                by_strategy[attempt.strategy]["success"] += 1

        return {
            "total": total,
            "successes": successes,
            "success_rate": successes / total if total > 0 else 0.0,
            "by_strategy": by_strategy,
        }


class TacticModifier:
    """
    Generate modified tactics based on recovery strategy.

    Implements OpenManus-style tactic transformation for error recovery.
    """

    # Alternative tactics for each common tactic family
    TACTIC_ALTERNATIVES: dict[str, list[str]] = {
        # Heavy automation
        "grind": ["simp_all", "try omega", "decide", "rfl", "native_decide"],
        "aesop": ["simp_all", "try grind", "decide"],

        # Simplification
        "simp": ["simp only [*]", "simp_all", "simp [*]"],
        "simp_all": ["simp only [*]", "simp [*]", "try grind"],

        # Arithmetic
        "omega": ["try scalar_tac", "decide", "try ring", "try linarith"],
        "ring": ["try omega", "try ring_nf", "try norm_num"],
        "linarith": ["try omega", "try nlinarith"],
        "norm_num": ["try omega", "decide", "rfl"],

        # Equality/rewriting
        "rfl": ["rfl", "try rfl", "try decide"],
        "rw": ["simp only", "conv => rw", "try rfl"],
        "conv": ["simp only", "rw"],

        # Application tactics
        "exact": ["apply", "refine", "use"],
        "apply": ["exact", "refine", "have h := "],
        "refine": ["exact", "apply"],

        # Case analysis
        "cases": ["rcases", "match", "by_cases"],
        "rcases": ["cases", "obtain", "match"],
        "induction": ["induction' ", "cases", "rcases"],

        # Introduction
        "intro": ["intros", "rintro"],
        "intros": ["intro", "rintro"],
        "rintro": ["intro", "intros"],

        # Finishing
        "trivial": ["try trivial", "try rfl", "try decide"],
        "decide": ["try native_decide", "rfl", "try trivial"],
        "assumption": ["try assumption", "exact h", "trivial"],

        # Scalar tactics (Mathlib)
        "scalar_tac": ["try omega", "try simp", "try ring"],
        "progress": ["simp", "try scalar_tac", "try omega"],
    }

    # Wrappers to add based on error patterns
    ERROR_WRAPPERS: dict[str, str] = {
        "timeout": "try {tactic}",
        "tactic_failed": "try {tactic}",
        "unknown_tactic": "{alternative}",
    }

    def __init__(self):
        """Initialize tactic modifier."""
        self._used_alternatives: dict[str, int] = {}

    def apply_strategy(
        self,
        current_tactic: str,
        strategy: RecoveryStrategy,
        context: RecoveryContext,
    ) -> str:
        """
        Generate modified tactic based on strategy.

        Args:
            current_tactic: The tactic that failed
            strategy: Recovery strategy to apply
            context: Recovery context with tried tactics, etc.

        Returns:
            Modified tactic string
        """
        if strategy == RecoveryStrategy.ABORT:
            return current_tactic  # No modification for abort

        if strategy == RecoveryStrategy.RETRY_SAME:
            return self._apply_retry_same(current_tactic, context)

        if strategy == RecoveryStrategy.TRY_ALTERNATIVE:
            return self._apply_try_alternative(current_tactic, context)

        if strategy == RecoveryStrategy.UNFOLD_MORE:
            return self._apply_unfold_more(current_tactic, context)

        if strategy == RecoveryStrategy.SIMPLIFY_FIRST:
            return self._apply_simplify_first(current_tactic, context)

        if strategy == RecoveryStrategy.BACKTRACK:
            # Backtracking is handled at graph level, not tactic level
            # Return a safe default
            return "try grind"

        if strategy == RecoveryStrategy.ESCALATE:
            # Escalation handled at graph level
            return current_tactic

        return current_tactic

    def _apply_retry_same(self, tactic: str, context: RecoveryContext) -> str:
        """Add try wrapper or minor variation."""
        # If already has try, don't double-wrap
        if tactic.strip().startswith("try"):
            # Try adding <;> combinator
            if "<;>" not in tactic:
                return f"{tactic} <;> try trivial"
            return tactic

        # Add try wrapper
        return f"try {tactic}"

    def _apply_try_alternative(self, tactic: str, context: RecoveryContext) -> str:
        """Get alternative tactic from same family."""
        tried = set(context.tried_tactics)

        # First, check if we have successful tactics to build on
        if context.successful_tactics:
            for success in reversed(context.successful_tactics):
                if success not in tried:
                    return success

        # Find the base tactic
        base_tactic = self._extract_base_tactic(tactic)

        # Get alternatives
        alternatives = self.TACTIC_ALTERNATIVES.get(base_tactic, [])

        # Find first untried alternative
        for alt in alternatives:
            if alt not in tried and alt != tactic:
                return alt

        # If all alternatives tried, try composition with simp
        if "simp" not in tactic:
            return f"simp only [*] <;> {tactic}"

        # Last resort: grind with safety wrapper
        if "grind" not in tried:
            return "try grind"

        return tactic

    def _apply_unfold_more(self, tactic: str, context: RecoveryContext) -> str:
        """Prepend unfold/simp to expand definitions."""
        # Check if we have specific definitions to unfold
        if context.definitions_to_unfold:
            unfolds = context.definitions_to_unfold[:2]  # Max 2 unfolds
            unfold_str = "; ".join(f"unfold {d}" for d in unfolds)
            return f"{unfold_str}; {tactic}"

        # Default: use simp only to unfold
        if "simp" not in tactic:
            return f"simp only [*]; {tactic}"

        # If simp already present, try simp_all
        return f"simp_all; {tactic}"

    def _apply_simplify_first(self, tactic: str, context: RecoveryContext) -> str:
        """Run simplification before main tactic."""
        # Don't double-simplify
        if tactic.startswith("simp"):
            return f"try grind <;> {tactic}"

        return f"simp only [*]; {tactic}"

    def _extract_base_tactic(self, tactic: str) -> str:
        """Extract the base tactic name from a complex tactic string."""
        # Remove try wrapper
        tactic = tactic.strip()
        if tactic.startswith("try"):
            tactic = tactic[3:].strip()

        # Get first word
        match = re.match(r"(\w+)", tactic)
        if match:
            return match.group(1).lower()

        return tactic.split()[0].lower() if tactic else ""

    def extract_definitions_from_error(self, error_content: str) -> list[str]:
        """
        Extract definition names from error message for unfolding.

        Looks for patterns like:
        - "unknown identifier 'FooBar'"
        - "failed to synthesize instance ... for 'SomeDef'"
        """
        definitions: list[str] = []

        # Pattern: identifier in quotes
        quoted = re.findall(r"'(\w+(?:\.\w+)*)'", error_content)
        for name in quoted:
            # Filter to likely definition names (capitalized)
            if name[0].isupper():
                definitions.append(name)

        # Pattern: explicit "definition" mentions
        def_pattern = re.findall(r"definition\s+(\w+)", error_content)
        definitions.extend(def_pattern)

        return definitions[:3]  # Limit to 3


# Convenience functions for use in nodes.py


def get_recovery_controller() -> ErrorRecoveryController:
    """Get a fresh ErrorRecoveryController instance."""
    return ErrorRecoveryController()


def get_tactic_modifier() -> TacticModifier:
    """Get a fresh TacticModifier instance."""
    return TacticModifier()
