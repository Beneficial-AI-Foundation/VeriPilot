"""
Goal complexity analysis for ROMA hierarchical decomposition.

This module analyzes proof goals to determine their complexity level,
which informs the Atomizer's decision to solve directly vs decompose.

Complexity factors:
- Nesting depth (nested quantifiers, implications)
- Quantifier count (∀, ∃, Σ)
- Hypothesis count and complexity
- Type complexity (dependent types, universes)
- Automation likelihood (can grind/simp handle it?)
"""

from __future__ import annotations

import re
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class GoalComplexity(str, Enum):
    """Classification of proof goal complexity."""

    ATOMIC = "atomic"
    """Goal is trivially solvable by automation (grind, simp, decide, rfl)."""

    SIMPLE = "simple"
    """Goal requires 2-3 standard tactics but no decomposition."""

    MODERATE = "moderate"
    """Goal may benefit from helper lemmas or case splits."""

    COMPLEX = "complex"
    """Goal should be decomposed into subtasks."""


@dataclass
class ComplexityScore:
    """Detailed complexity scoring for a proof goal."""

    # Individual factor scores (0.0 to 1.0)
    nesting_score: float = 0.0
    """Score based on nesting depth of quantifiers/implications."""

    quantifier_score: float = 0.0
    """Score based on number and type of quantifiers."""

    hypothesis_score: float = 0.0
    """Score based on number and complexity of hypotheses."""

    type_complexity_score: float = 0.0
    """Score based on type structure (dependent types, universes)."""

    automation_likelihood: float = 0.5
    """Estimated probability that automation tactics will succeed."""

    # Computed aggregate
    overall_score: float = 0.0
    """Weighted aggregate score (0.0 = trivial, 1.0 = very complex)."""

    # Classification
    complexity: GoalComplexity = GoalComplexity.SIMPLE
    """Classified complexity level."""

    # Analysis details
    detected_patterns: list[str] = field(default_factory=list)
    """Patterns detected in the goal (e.g., 'nested_forall', 'dependent_type')."""

    reasoning: str = ""
    """Human-readable explanation of the complexity assessment."""

    def __post_init__(self):
        """Compute overall score and classification if not set."""
        if self.overall_score == 0.0 and any([
            self.nesting_score,
            self.quantifier_score,
            self.hypothesis_score,
            self.type_complexity_score,
        ]):
            self._compute_overall()

    def _compute_overall(self) -> None:
        """Compute weighted overall score and classify."""
        # Weights from the plan:
        # score = nesting*0.15 + quantifiers*0.2 + hypotheses*0.05
        #       + type_complexity*0.3 + (1-automation)*0.3
        self.overall_score = (
            self.nesting_score * 0.15
            + self.quantifier_score * 0.20
            + self.hypothesis_score * 0.05
            + self.type_complexity_score * 0.30
            + (1.0 - self.automation_likelihood) * 0.30
        )

        # Classify based on score and automation likelihood
        if self.automation_likelihood > 0.7 and self.overall_score < 0.2:
            self.complexity = GoalComplexity.ATOMIC
        elif self.overall_score < 0.3:
            self.complexity = GoalComplexity.SIMPLE
        elif self.overall_score < 0.6:
            self.complexity = GoalComplexity.MODERATE
        else:
            self.complexity = GoalComplexity.COMPLEX


class GoalComplexityAnalyzer:
    """
    Analyzes proof goals to determine complexity level.

    Uses pattern matching and heuristics to score goals on multiple
    dimensions, then classifies into ATOMIC/SIMPLE/MODERATE/COMPLEX.

    Example:
        analyzer = GoalComplexityAnalyzer()
        score = analyzer.analyze(goal_state, file_context)
        if score.complexity == GoalComplexity.COMPLEX:
            # Decompose into subtasks
            ...
    """

    # Patterns that suggest higher complexity
    QUANTIFIER_PATTERNS = [
        (r"∀\s*\w+", "forall"),
        (r"∃\s*\w+", "exists"),
        (r"Σ\s*\w+", "sigma"),
        (r"\\forall", "forall_tex"),
        (r"\\exists", "exists_tex"),
    ]

    # Patterns that suggest type complexity
    TYPE_PATTERNS = [
        (r"Type\s*\d+", "universe_level"),
        (r"Sort\s*\d+", "sort_level"),
        (r"@\[", "attribute"),
        (r"\{.*:.*\}", "dependent_type"),
        (r"→.*→.*→", "multiple_arrows"),
    ]

    # Patterns that suggest automation will work
    AUTOMATION_FRIENDLY = [
        (r"^\s*True\s*$", "trivial_true"),
        (r"^\s*\w+\s*=\s*\w+\s*$", "simple_equality"),
        (r"Nat\.add|Nat\.mul|Nat\.sub", "nat_arithmetic"),
        (r"List\.(append|length|map|filter)", "list_ops"),
        (r"∧|∨|¬|↔", "propositional"),
    ]

    # Patterns that resist automation
    AUTOMATION_RESISTANT = [
        (r"sorry", "contains_sorry"),
        (r"noncomputable", "noncomputable"),
        (r"axiom", "uses_axiom"),
        (r"Classical\.", "classical_logic"),
        (r"funext|propext|Quot", "extensionality"),
    ]

    def __init__(self, llm_provider: Optional[object] = None):
        """
        Initialize the complexity analyzer.

        Args:
            llm_provider: Optional LLM for semantic analysis.
                         If None, uses pattern matching only.
        """
        self.llm_provider = llm_provider
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for performance."""
        self._quantifier_re = [
            (re.compile(p, re.MULTILINE), name)
            for p, name in self.QUANTIFIER_PATTERNS
        ]
        self._type_re = [
            (re.compile(p, re.MULTILINE), name)
            for p, name in self.TYPE_PATTERNS
        ]
        self._auto_friendly_re = [
            (re.compile(p, re.MULTILINE | re.IGNORECASE), name)
            for p, name in self.AUTOMATION_FRIENDLY
        ]
        self._auto_resist_re = [
            (re.compile(p, re.MULTILINE), name)
            for p, name in self.AUTOMATION_RESISTANT
        ]

    def analyze(
        self,
        goal_state: str,
        context: str = "",
        previous_attempts: int = 0,
        previous_errors: Optional[list[str]] = None,
    ) -> ComplexityScore:
        """
        Analyze a proof goal's complexity.

        Args:
            goal_state: The current goal state from Lean (proof state text).
            context: Surrounding code context (definitions, imports).
            previous_attempts: Number of prior tactic attempts on this goal.
            previous_errors: List of error messages from prior attempts.

        Returns:
            ComplexityScore with detailed analysis and classification.
        """
        detected = []
        previous_errors = previous_errors or []

        # 1. Analyze nesting depth
        nesting_score = self._analyze_nesting(goal_state, detected)

        # 2. Count and score quantifiers
        quantifier_score = self._analyze_quantifiers(goal_state, detected)

        # 3. Analyze hypotheses
        hypothesis_score = self._analyze_hypotheses(goal_state, detected)

        # 4. Analyze type complexity
        type_score = self._analyze_type_complexity(goal_state, context, detected)

        # 5. Estimate automation likelihood
        auto_likelihood = self._estimate_automation_likelihood(
            goal_state, context, previous_attempts, previous_errors, detected
        )

        # Build the score
        score = ComplexityScore(
            nesting_score=nesting_score,
            quantifier_score=quantifier_score,
            hypothesis_score=hypothesis_score,
            type_complexity_score=type_score,
            automation_likelihood=auto_likelihood,
            detected_patterns=detected,
        )

        # Generate reasoning
        score.reasoning = self._generate_reasoning(score, detected)

        logger.debug(
            f"Complexity analysis: {score.complexity.value} "
            f"(score={score.overall_score:.2f}, auto={auto_likelihood:.2f})"
        )

        return score

    def _analyze_nesting(self, goal_state: str, detected: list[str]) -> float:
        """Analyze nesting depth of quantifiers and implications."""
        # Count nesting by tracking depth markers
        max_depth = 0
        current_depth = 0

        # Simple heuristic: parentheses and arrows indicate nesting
        for char in goal_state:
            if char in "({[":
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char in ")}]":
                current_depth = max(0, current_depth - 1)

        # Count arrow chains (→)
        arrow_count = goal_state.count("→") + goal_state.count("->")
        if arrow_count > 3:
            detected.append("deep_arrow_chain")

        # Normalize: depth 0-2 = low, 3-5 = medium, 6+ = high
        if max_depth > 5:
            detected.append("deeply_nested")
            return min(1.0, max_depth / 10.0)
        elif max_depth > 2:
            detected.append("moderately_nested")
            return max_depth / 10.0
        return max_depth / 20.0

    def _analyze_quantifiers(self, goal_state: str, detected: list[str]) -> float:
        """Count and score quantifiers."""
        total_quantifiers = 0

        for pattern, name in self._quantifier_re:
            matches = pattern.findall(goal_state)
            if matches:
                total_quantifiers += len(matches)
                detected.append(f"{name}:{len(matches)}")

        # Nested quantifiers are harder
        if "∀" in goal_state and "∃" in goal_state:
            detected.append("mixed_quantifiers")
            total_quantifiers += 2  # Penalty for mixing

        # Normalize: 0-1 = low, 2-4 = medium, 5+ = high
        return min(1.0, total_quantifiers / 8.0)

    def _analyze_hypotheses(self, goal_state: str, detected: list[str]) -> float:
        """Analyze hypothesis count and complexity."""
        # Count hypothesis lines (lines with : that aren't the goal)
        lines = goal_state.strip().split("\n")
        hypothesis_lines = [
            line for line in lines
            if ":" in line and "⊢" not in line and not line.strip().startswith("⊢")
        ]

        hyp_count = len(hypothesis_lines)
        if hyp_count > 10:
            detected.append("many_hypotheses")
        elif hyp_count > 5:
            detected.append("several_hypotheses")

        # Score based on count
        return min(1.0, hyp_count / 15.0)

    def _analyze_type_complexity(
        self,
        goal_state: str,
        context: str,
        detected: list[str],
    ) -> float:
        """Analyze type structure complexity."""
        score = 0.0
        combined = goal_state + "\n" + context

        for pattern, name in self._type_re:
            if pattern.search(combined):
                detected.append(name)
                score += 0.15

        # Dependent types are harder
        if "{" in goal_state and ":" in goal_state:
            brace_depth = 0
            has_dependent = False
            for char in goal_state:
                if char == "{":
                    brace_depth += 1
                elif char == "}":
                    brace_depth -= 1
                elif char == ":" and brace_depth > 0:
                    has_dependent = True
                    break
            if has_dependent:
                detected.append("dependent_type_binding")
                score += 0.2

        return min(1.0, score)

    def _estimate_automation_likelihood(
        self,
        goal_state: str,
        context: str,
        previous_attempts: int,
        previous_errors: list[str],
        detected: list[str],
    ) -> float:
        """Estimate probability that automation tactics will succeed."""
        likelihood = 0.5  # Start neutral

        combined = goal_state + "\n" + context

        # Boost for automation-friendly patterns
        for pattern, name in self._auto_friendly_re:
            if pattern.search(combined):
                detected.append(f"auto_friendly:{name}")
                likelihood += 0.1

        # Penalty for automation-resistant patterns
        for pattern, name in self._auto_resist_re:
            if pattern.search(combined):
                detected.append(f"auto_resistant:{name}")
                likelihood -= 0.15

        # Prior failed attempts reduce likelihood
        if previous_attempts > 0:
            likelihood -= 0.1 * min(previous_attempts, 3)
            detected.append(f"prior_attempts:{previous_attempts}")

        # Certain errors strongly indicate automation won't work
        error_text = " ".join(previous_errors).lower()
        if "tactic failed" in error_text or "no goals" in error_text:
            likelihood -= 0.1
        if "type mismatch" in error_text:
            likelihood -= 0.15
            detected.append("prior_type_mismatch")

        # Goal length heuristic: very short goals often work with automation
        goal_len = len(goal_state.strip())
        if goal_len < 50:
            likelihood += 0.1
            detected.append("short_goal")
        elif goal_len > 500:
            likelihood -= 0.1
            detected.append("long_goal")

        return max(0.0, min(1.0, likelihood))

    def _generate_reasoning(
        self,
        score: ComplexityScore,
        detected: list[str],
    ) -> str:
        """Generate human-readable explanation."""
        parts = []

        # Complexity level
        parts.append(f"Goal classified as {score.complexity.value.upper()}.")

        # Key factors
        if score.nesting_score > 0.3:
            parts.append("High nesting depth increases complexity.")
        if score.quantifier_score > 0.3:
            parts.append("Multiple quantifiers detected.")
        if score.type_complexity_score > 0.3:
            parts.append("Complex type structure present.")

        # Automation assessment
        if score.automation_likelihood > 0.7:
            parts.append("Automation tactics likely to succeed.")
        elif score.automation_likelihood < 0.3:
            parts.append("Automation unlikely; manual tactics needed.")

        # Notable patterns
        notable = [p for p in detected if not p.startswith("auto_")]
        if notable:
            parts.append(f"Detected: {', '.join(notable[:5])}")

        return " ".join(parts)

    async def analyze_with_llm(
        self,
        goal_state: str,
        context: str = "",
    ) -> ComplexityScore:
        """
        Analyze complexity using LLM for semantic understanding.

        Falls back to pattern-based analysis if no LLM provider.

        Args:
            goal_state: The current goal state from Lean.
            context: Surrounding code context.

        Returns:
            ComplexityScore with LLM-enhanced analysis.
        """
        # Start with pattern-based analysis
        score = self.analyze(goal_state, context)

        if not self.llm_provider:
            return score

        # TODO: Implement LLM-based refinement
        # This would ask the LLM to:
        # 1. Identify mathematical concepts in the goal
        # 2. Suggest decomposition strategies
        # 3. Refine automation likelihood estimate

        logger.debug("LLM-enhanced analysis not yet implemented; using patterns")
        return score


def quick_complexity_check(goal_state: str) -> GoalComplexity:
    """
    Quick complexity classification without full analysis.

    Useful for fast filtering before detailed analysis.

    Args:
        goal_state: The goal state text.

    Returns:
        GoalComplexity classification.
    """
    # Very short goals are likely atomic
    if len(goal_state.strip()) < 30:
        return GoalComplexity.ATOMIC

    # Many quantifiers suggest complexity
    quantifier_count = (
        goal_state.count("∀")
        + goal_state.count("∃")
        + goal_state.count("forall")
        + goal_state.count("exists")
    )
    if quantifier_count > 3:
        return GoalComplexity.COMPLEX

    # Deep nesting suggests complexity
    if goal_state.count("(") > 8 or goal_state.count("→") > 5:
        return GoalComplexity.MODERATE

    # Default to simple
    return GoalComplexity.SIMPLE
