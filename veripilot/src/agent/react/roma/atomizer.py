"""
ROMA Atomizer - decides whether to solve a goal directly or decompose it.

The Atomizer is the first decision point in the ROMA architecture:
- ATOMIC goals are sent directly to the ReAct loop
- COMPLEX goals are sent to the Planner for decomposition

Decision factors:
- Complexity score from GoalComplexityAnalyzer
- Prior attempt history (failed attempts suggest decomposition)
- Automation likelihood
- Goal structure patterns
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .complexity import (
    GoalComplexity,
    ComplexityScore,
    GoalComplexityAnalyzer,
    quick_complexity_check,
)

logger = logging.getLogger(__name__)


@dataclass
class AtomizerDecision:
    """Result of the Atomizer's decision."""

    should_decompose: bool
    """True if goal should be decomposed, False for direct solving."""

    reason: str
    """Human-readable explanation for the decision."""

    complexity_score: Optional[ComplexityScore] = None
    """Full complexity analysis (if performed)."""

    suggested_strategy: Optional[str] = None
    """Suggested decomposition strategy if decomposing (cases, induction, etc.)."""

    confidence: float = 0.5
    """Confidence in the decision (0.0 to 1.0)."""


class Atomizer:
    """
    Decides whether a proof goal should be solved directly or decomposed.

    The Atomizer uses complexity analysis and attempt history to make
    intelligent decisions about when to decompose goals.

    Decision Logic:
        1. If automation_likelihood > 0.7 AND attempts < 2: DIRECT
        2. If attempts >= 3 AND complexity >= MODERATE: DECOMPOSE
        3. If overall_score > 0.5: DECOMPOSE
        4. Otherwise: DIRECT

    Example:
        atomizer = Atomizer()
        decision = await atomizer.should_decompose(
            goal_state="⊢ ∀ n : Nat, n + 0 = n",
            context="",
            previous_attempts=0,
        )
        if decision.should_decompose:
            # Send to planner
            ...
        else:
            # Send to ReAct loop
            ...
    """

    # Thresholds for decision making
    COMPLEXITY_THRESHOLD = 0.5
    """Overall score above this triggers decomposition."""

    AUTOMATION_THRESHOLD = 0.7
    """Automation likelihood above this favors direct solving."""

    ATTEMPT_THRESHOLD = 3
    """After this many failed attempts, consider decomposition."""

    MODERATE_ATTEMPT_THRESHOLD = 2
    """For MODERATE goals, decompose after this many attempts."""

    def __init__(
        self,
        complexity_analyzer: Optional[GoalComplexityAnalyzer] = None,
        llm_provider: Optional[object] = None,
    ):
        """
        Initialize the Atomizer.

        Args:
            complexity_analyzer: Pre-configured analyzer. If None, creates one.
            llm_provider: Optional LLM for enhanced decision making.
        """
        self.analyzer = complexity_analyzer or GoalComplexityAnalyzer(llm_provider)
        self.llm_provider = llm_provider

    async def should_decompose(
        self,
        goal_state: str,
        context: str = "",
        previous_attempts: int = 0,
        previous_errors: Optional[list[str]] = None,
        tried_tactics: Optional[list[str]] = None,
    ) -> AtomizerDecision:
        """
        Decide whether to decompose a goal or solve directly.

        Args:
            goal_state: The current proof goal state from Lean.
            context: Surrounding code context (definitions, imports).
            previous_attempts: Number of prior attempts on this goal.
            previous_errors: Error messages from prior attempts.
            tried_tactics: List of tactics already tried.

        Returns:
            AtomizerDecision with the decision and reasoning.
        """
        previous_errors = previous_errors or []
        tried_tactics = tried_tactics or []

        # Quick check for obviously simple goals
        quick_check = quick_complexity_check(goal_state)
        if quick_check == GoalComplexity.ATOMIC and previous_attempts < 2:
            logger.debug("Quick check: ATOMIC goal, solving directly")
            return AtomizerDecision(
                should_decompose=False,
                reason="Goal appears trivial; attempting direct automation.",
                confidence=0.8,
            )

        # Full complexity analysis
        score = self.analyzer.analyze(
            goal_state=goal_state,
            context=context,
            previous_attempts=previous_attempts,
            previous_errors=previous_errors,
        )

        # Apply decision logic
        decision = self._make_decision(
            score=score,
            previous_attempts=previous_attempts,
            previous_errors=previous_errors,
            tried_tactics=tried_tactics,
        )

        logger.info(
            f"Atomizer decision: {'DECOMPOSE' if decision.should_decompose else 'DIRECT'} "
            f"(complexity={score.complexity.value}, score={score.overall_score:.2f})"
        )

        return decision

    def _make_decision(
        self,
        score: ComplexityScore,
        previous_attempts: int,
        previous_errors: list[str],
        tried_tactics: list[str],
    ) -> AtomizerDecision:
        """Apply decision logic based on complexity and history."""

        reasons = []
        should_decompose = False
        confidence = 0.5
        suggested_strategy = None

        # Rule 1: High automation likelihood with few attempts → DIRECT
        if (
            score.automation_likelihood > self.AUTOMATION_THRESHOLD
            and previous_attempts < 2
        ):
            reasons.append(
                f"High automation likelihood ({score.automation_likelihood:.0%})"
            )
            confidence = score.automation_likelihood
            should_decompose = False

        # Rule 2: Many failed attempts + moderate complexity → DECOMPOSE
        elif (
            previous_attempts >= self.ATTEMPT_THRESHOLD
            and score.complexity in (GoalComplexity.MODERATE, GoalComplexity.COMPLEX)
        ):
            reasons.append(
                f"Multiple failed attempts ({previous_attempts}) with "
                f"{score.complexity.value} complexity"
            )
            should_decompose = True
            confidence = 0.7
            suggested_strategy = self._suggest_strategy(score, previous_errors)

        # Rule 3: High complexity score → DECOMPOSE
        elif score.overall_score > self.COMPLEXITY_THRESHOLD:
            reasons.append(
                f"High complexity score ({score.overall_score:.2f} > {self.COMPLEXITY_THRESHOLD})"
            )
            should_decompose = True
            confidence = min(0.9, score.overall_score + 0.2)
            suggested_strategy = self._suggest_strategy(score, previous_errors)

        # Rule 4: Complex classification → DECOMPOSE
        elif score.complexity == GoalComplexity.COMPLEX:
            reasons.append(f"Goal classified as {score.complexity.value}")
            should_decompose = True
            confidence = 0.75
            suggested_strategy = self._suggest_strategy(score, previous_errors)

        # Rule 5: Moderate with some attempts → consider decomposition
        elif (
            score.complexity == GoalComplexity.MODERATE
            and previous_attempts >= self.MODERATE_ATTEMPT_THRESHOLD
        ):
            reasons.append(
                f"Moderate complexity with {previous_attempts} failed attempts"
            )
            should_decompose = True
            confidence = 0.6
            suggested_strategy = self._suggest_strategy(score, previous_errors)

        # Default: solve directly
        else:
            reasons.append(f"Goal appears manageable ({score.complexity.value})")
            should_decompose = False
            confidence = max(0.5, score.automation_likelihood)

        # Add complexity reasoning
        if score.reasoning:
            reasons.append(score.reasoning)

        return AtomizerDecision(
            should_decompose=should_decompose,
            reason=" ".join(reasons),
            complexity_score=score,
            suggested_strategy=suggested_strategy,
            confidence=confidence,
        )

    def _suggest_strategy(
        self,
        score: ComplexityScore,
        previous_errors: list[str],
    ) -> str:
        """Suggest a decomposition strategy based on goal patterns."""
        patterns = score.detected_patterns

        # Check for induction signals
        if any("Nat" in p or "List" in p for p in patterns):
            return "induction"

        # Check for case split signals
        if "mixed_quantifiers" in patterns or any("exists" in p for p in patterns):
            return "cases"

        # Check error patterns
        error_text = " ".join(previous_errors).lower()
        if "type mismatch" in error_text:
            return "lemma"  # May need helper lemma
        if "unknown identifier" in error_text:
            return "sequential"  # Build up definitions

        # Default based on complexity
        if score.overall_score > 0.7:
            return "hierarchical"  # Deep decomposition
        elif score.quantifier_score > 0.3:
            return "cases"  # Split on quantifiers
        else:
            return "sequential"  # Step-by-step

    async def should_decompose_with_llm(
        self,
        goal_state: str,
        context: str = "",
        previous_attempts: int = 0,
        previous_errors: Optional[list[str]] = None,
    ) -> AtomizerDecision:
        """
        LLM-enhanced decomposition decision.

        Uses the LLM to provide semantic understanding of when
        decomposition would be beneficial.

        Falls back to rule-based decision if no LLM available.
        """
        # Start with rule-based decision
        decision = await self.should_decompose(
            goal_state=goal_state,
            context=context,
            previous_attempts=previous_attempts,
            previous_errors=previous_errors,
        )

        if not self.llm_provider:
            return decision

        # TODO: Implement LLM enhancement
        # The LLM would:
        # 1. Identify the mathematical structure
        # 2. Suggest specific decomposition approaches
        # 3. Estimate success probability

        logger.debug("LLM-enhanced atomizer not yet implemented")
        return decision


def should_try_direct_first(
    goal_state: str,
    previous_attempts: int = 0,
) -> bool:
    """
    Quick check: should we try direct tactics before considering decomposition?

    This is a fast heuristic for the common case where goals are simple.

    Args:
        goal_state: The goal state text.
        previous_attempts: Number of prior attempts.

    Returns:
        True if direct tactics should be tried first.
    """
    # Always try direct first if no prior attempts
    if previous_attempts == 0:
        return True

    # Quick complexity check
    complexity = quick_complexity_check(goal_state)

    # ATOMIC goals should always try direct
    if complexity == GoalComplexity.ATOMIC:
        return True

    # SIMPLE goals try direct up to 2 attempts
    if complexity == GoalComplexity.SIMPLE and previous_attempts < 2:
        return True

    # Otherwise, might be time to decompose
    return False
