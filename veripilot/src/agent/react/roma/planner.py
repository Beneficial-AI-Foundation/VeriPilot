"""
ROMA Planner - decomposes complex proof goals into manageable subtasks.

The Planner is invoked when the Atomizer decides a goal needs decomposition.
It analyzes the goal structure and creates an ordered plan of subtasks.

Decomposition Strategies:
- CASES: Split goal by cases (match, if-then-else, disjunction)
- INDUCTION: Set up induction with base case and inductive step
- SEQUENTIAL: Linear sequence of tactics building toward goal
- LEMMA: Extract helper lemma and prove separately
- HIERARCHICAL: Recursive decomposition of nested goals
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .complexity import ComplexityScore, GoalComplexity

logger = logging.getLogger(__name__)


class DecompositionStrategy(str, Enum):
    """Strategy for decomposing a complex goal."""

    CASES = "cases"
    """Split goal by case analysis (Or.elim, match, if-then-else)."""

    INDUCTION = "induction"
    """Induction on a recursive type (Nat, List, etc.)."""

    SEQUENTIAL = "sequential"
    """Linear sequence of tactics building toward goal."""

    LEMMA = "lemma"
    """Extract and prove a helper lemma."""

    HIERARCHICAL = "hierarchical"
    """Recursive decomposition into nested subgoals."""

    UNFOLD = "unfold"
    """Unfold definitions to expose structure."""


@dataclass
class SubTask:
    """A single subtask in a decomposition plan."""

    id: str
    """Unique identifier for this subtask (e.g., 'step_1', 'base_case')."""

    description: str
    """Human-readable description of what this subtask accomplishes."""

    goal_hint: str
    """Expected goal state or pattern after parent tactics are applied."""

    suggested_tactics: list[str] = field(default_factory=list)
    """Tactics likely to work for this subtask."""

    dependencies: list[str] = field(default_factory=list)
    """IDs of subtasks that must complete before this one."""

    is_critical: bool = True
    """If True, failure of this subtask fails the whole plan."""

    max_attempts: int = 5
    """Maximum attempts before giving up on this subtask."""

    metadata: dict = field(default_factory=dict)
    """Additional strategy-specific metadata."""


@dataclass
class DecompositionPlan:
    """A complete plan for decomposing a complex goal."""

    strategy: DecompositionStrategy
    """The decomposition strategy being used."""

    subtasks: list[SubTask]
    """Ordered list of subtasks to complete."""

    synthesis_strategy: str = "sequential"
    """How to combine sub-proofs: 'sequential', 'parallel', 'nested'."""

    entry_tactic: Optional[str] = None
    """Tactic to apply before starting subtasks (e.g., 'induction n')."""

    exit_tactic: Optional[str] = None
    """Tactic to apply after all subtasks complete."""

    estimated_complexity: float = 0.5
    """Estimated overall complexity of the decomposed problem."""

    reasoning: str = ""
    """Explanation of why this decomposition was chosen."""

    def get_ready_subtasks(self, completed: set[str]) -> list[SubTask]:
        """Get subtasks whose dependencies are all satisfied."""
        return [
            st for st in self.subtasks
            if st.id not in completed
            and all(dep in completed for dep in st.dependencies)
        ]

    def get_subtask(self, task_id: str) -> Optional[SubTask]:
        """Get a subtask by ID."""
        for st in self.subtasks:
            if st.id == task_id:
                return st
        return None


class RomaPlanner:
    """
    Plans the decomposition of complex proof goals.

    The planner analyzes goal structure and creates execution plans
    with properly ordered subtasks.

    Example:
        planner = RomaPlanner()
        plan = await planner.decompose(
            goal_state="⊢ ∀ n : Nat, n + 0 = n",
            context="",
            complexity_score=score,
        )
        for subtask in plan.subtasks:
            # Execute subtask
            ...
    """

    # Patterns that suggest specific strategies
    INDUCTION_SIGNALS = [
        (r"Nat\b", "Nat_induction"),
        (r"List\b", "List_induction"),
        (r"Fin\b", "Fin_induction"),
        (r"Vector\b", "Vector_induction"),
        (r"Tree\b", "Tree_induction"),
    ]

    CASES_SIGNALS = [
        (r"∨|\\or|Or\b", "disjunction"),
        (r"match\b", "pattern_match"),
        (r"if\s+.*\s+then", "conditional"),
        (r"Option\b", "option_cases"),
        (r"Bool\b", "bool_cases"),
        (r"Sum\b", "sum_cases"),
    ]

    LEMMA_SIGNALS = [
        (r"have\s+:", "have_statement"),
        (r"let\s+\w+\s*:", "let_binding"),
        (r"suffices\b", "suffices"),
    ]

    def __init__(self, llm_provider: Optional[object] = None):
        """
        Initialize the planner.

        Args:
            llm_provider: Optional LLM for enhanced planning.
        """
        self.llm_provider = llm_provider
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns."""
        self._induction_re = [
            (re.compile(p, re.MULTILINE), name)
            for p, name in self.INDUCTION_SIGNALS
        ]
        self._cases_re = [
            (re.compile(p, re.MULTILINE), name)
            for p, name in self.CASES_SIGNALS
        ]
        self._lemma_re = [
            (re.compile(p, re.MULTILINE), name)
            for p, name in self.LEMMA_SIGNALS
        ]

    async def decompose(
        self,
        goal_state: str,
        context: str = "",
        complexity_score: Optional[ComplexityScore] = None,
        suggested_strategy: Optional[str] = None,
        rag_results: Optional[list[dict]] = None,
    ) -> DecompositionPlan:
        """
        Create a decomposition plan for a complex goal.

        Args:
            goal_state: The current proof goal state.
            context: Surrounding code context.
            complexity_score: Pre-computed complexity analysis.
            suggested_strategy: Strategy hint from Atomizer.
            rag_results: Relevant examples from RAG system.

        Returns:
            DecompositionPlan with ordered subtasks.
        """
        rag_results = rag_results or []

        # Detect the best strategy
        strategy = self._detect_strategy(
            goal_state=goal_state,
            context=context,
            suggested=suggested_strategy,
        )

        # Create plan based on strategy
        if strategy == DecompositionStrategy.INDUCTION:
            plan = self._plan_induction(goal_state, context)
        elif strategy == DecompositionStrategy.CASES:
            plan = self._plan_cases(goal_state, context)
        elif strategy == DecompositionStrategy.LEMMA:
            plan = self._plan_lemma(goal_state, context)
        elif strategy == DecompositionStrategy.UNFOLD:
            plan = self._plan_unfold(goal_state, context)
        elif strategy == DecompositionStrategy.HIERARCHICAL:
            plan = self._plan_hierarchical(goal_state, context, complexity_score)
        else:  # SEQUENTIAL
            plan = self._plan_sequential(goal_state, context)

        # Enrich with RAG results if available
        if rag_results:
            self._enrich_with_rag(plan, rag_results)

        logger.info(
            f"Created {strategy.value} plan with {len(plan.subtasks)} subtasks"
        )

        return plan

    def _detect_strategy(
        self,
        goal_state: str,
        context: str,
        suggested: Optional[str],
    ) -> DecompositionStrategy:
        """Detect the best decomposition strategy."""
        combined = goal_state + "\n" + context

        # Use suggested strategy if valid
        if suggested:
            try:
                return DecompositionStrategy(suggested)
            except ValueError:
                pass

        # Check for induction signals
        for pattern, _ in self._induction_re:
            if pattern.search(combined):
                # Look for recursive structure in goal
                if "∀" in goal_state or "forall" in goal_state.lower():
                    return DecompositionStrategy.INDUCTION

        # Check for case split signals
        for pattern, _ in self._cases_re:
            if pattern.search(goal_state):
                return DecompositionStrategy.CASES

        # Check for lemma signals
        for pattern, _ in self._lemma_re:
            if pattern.search(goal_state):
                return DecompositionStrategy.LEMMA

        # Check for complex definitions that need unfolding
        if "def " in context or "@[simp]" in context:
            if len(goal_state) > 200:
                return DecompositionStrategy.UNFOLD

        # Default to sequential
        return DecompositionStrategy.SEQUENTIAL

    def _plan_induction(
        self,
        goal_state: str,
        context: str,
    ) -> DecompositionPlan:
        """Create an induction-based decomposition plan."""
        # Detect induction variable
        induction_var = self._detect_induction_variable(goal_state)

        subtasks = [
            SubTask(
                id="base_case",
                description=f"Prove base case ({induction_var} = 0 or empty)",
                goal_hint="Base case goal after 'induction'",
                suggested_tactics=["rfl", "simp", "decide", "trivial"],
                dependencies=[],
                is_critical=True,
            ),
            SubTask(
                id="inductive_step",
                description=f"Prove inductive step (assuming IH for {induction_var})",
                goal_hint="Goal with induction hypothesis in context",
                suggested_tactics=[
                    "simp [*]",
                    "rw [ih]",
                    "exact ih",
                    "apply ih",
                    "omega",
                ],
                dependencies=["base_case"],
                is_critical=True,
            ),
        ]

        return DecompositionPlan(
            strategy=DecompositionStrategy.INDUCTION,
            subtasks=subtasks,
            synthesis_strategy="sequential",
            entry_tactic=f"induction {induction_var}",
            reasoning=f"Induction on {induction_var} detected from goal structure.",
        )

    def _plan_cases(
        self,
        goal_state: str,
        context: str,
    ) -> DecompositionPlan:
        """Create a case-split decomposition plan."""
        # Detect what we're splitting on
        split_target = self._detect_case_target(goal_state)
        num_cases = self._estimate_case_count(goal_state, split_target)

        subtasks = []
        for i in range(num_cases):
            subtasks.append(
                SubTask(
                    id=f"case_{i + 1}",
                    description=f"Prove case {i + 1} of {num_cases}",
                    goal_hint=f"Case {i + 1} subgoal",
                    suggested_tactics=["simp", "trivial", "rfl", "exact h"],
                    dependencies=[],  # Cases are independent
                    is_critical=True,
                )
            )

        return DecompositionPlan(
            strategy=DecompositionStrategy.CASES,
            subtasks=subtasks,
            synthesis_strategy="parallel",  # Cases can be done in parallel
            entry_tactic=f"cases {split_target}" if split_target else "cases",
            reasoning=f"Case split on {split_target or 'hypothesis'} with {num_cases} cases.",
        )

    def _plan_lemma(
        self,
        goal_state: str,
        context: str,
    ) -> DecompositionPlan:
        """Create a helper-lemma decomposition plan."""
        subtasks = [
            SubTask(
                id="helper_lemma",
                description="Prove helper lemma (have statement)",
                goal_hint="The 'have' statement goal",
                suggested_tactics=["exact", "apply", "simp", "rfl"],
                dependencies=[],
                is_critical=True,
            ),
            SubTask(
                id="use_lemma",
                description="Use helper lemma to complete main goal",
                goal_hint="Main goal with helper in context",
                suggested_tactics=["exact this", "apply this", "simp [this]"],
                dependencies=["helper_lemma"],
                is_critical=True,
            ),
        ]

        return DecompositionPlan(
            strategy=DecompositionStrategy.LEMMA,
            subtasks=subtasks,
            synthesis_strategy="sequential",
            reasoning="Goal suggests extracting a helper lemma.",
        )

    def _plan_unfold(
        self,
        goal_state: str,
        context: str,
    ) -> DecompositionPlan:
        """Create an unfold-based decomposition plan."""
        # Detect definitions to unfold
        defs_to_unfold = self._detect_definitions(goal_state, context)

        subtasks = [
            SubTask(
                id="unfold_defs",
                description=f"Unfold definitions: {', '.join(defs_to_unfold[:3])}",
                goal_hint="Goal with definitions expanded",
                suggested_tactics=[
                    f"unfold {d}" for d in defs_to_unfold[:3]
                ] + ["simp only [*]"],
                dependencies=[],
                is_critical=True,
            ),
            SubTask(
                id="simplify",
                description="Simplify expanded goal",
                goal_hint="Simplified form",
                suggested_tactics=["simp", "ring", "omega", "decide"],
                dependencies=["unfold_defs"],
                is_critical=True,
            ),
        ]

        return DecompositionPlan(
            strategy=DecompositionStrategy.UNFOLD,
            subtasks=subtasks,
            synthesis_strategy="sequential",
            entry_tactic=f"unfold {defs_to_unfold[0]}" if defs_to_unfold else None,
            reasoning=f"Unfolding definitions to expose structure.",
        )

    def _plan_sequential(
        self,
        goal_state: str,
        context: str,
    ) -> DecompositionPlan:
        """Create a sequential tactic decomposition plan."""
        # Generic sequential plan
        subtasks = [
            SubTask(
                id="step_1",
                description="Apply initial simplification",
                goal_hint="Simplified goal",
                suggested_tactics=["simp", "simp_all", "norm_num"],
                dependencies=[],
                is_critical=False,  # Can skip if not helpful
            ),
            SubTask(
                id="step_2",
                description="Apply main proof tactic",
                goal_hint="Reduced goal or closed",
                suggested_tactics=["exact", "apply", "rw", "have"],
                dependencies=["step_1"],
                is_critical=True,
            ),
            SubTask(
                id="step_3",
                description="Close remaining goals",
                goal_hint="No goals",
                suggested_tactics=["trivial", "assumption", "rfl", "decide"],
                dependencies=["step_2"],
                is_critical=True,
            ),
        ]

        return DecompositionPlan(
            strategy=DecompositionStrategy.SEQUENTIAL,
            subtasks=subtasks,
            synthesis_strategy="sequential",
            reasoning="Using sequential tactic approach.",
        )

    def _plan_hierarchical(
        self,
        goal_state: str,
        context: str,
        complexity_score: Optional[ComplexityScore],
    ) -> DecompositionPlan:
        """Create a hierarchical decomposition for deeply nested goals."""
        # For very complex goals, create nested decomposition
        subtasks = [
            SubTask(
                id="outer_structure",
                description="Address outer goal structure",
                goal_hint="Exposed inner goals",
                suggested_tactics=["intro", "apply", "constructor"],
                dependencies=[],
                is_critical=True,
                metadata={"may_spawn_subagent": True},
            ),
            SubTask(
                id="inner_goals",
                description="Solve inner subgoals (may recurse)",
                goal_hint="Inner goal solved",
                suggested_tactics=["exact", "apply", "simp"],
                dependencies=["outer_structure"],
                is_critical=True,
                metadata={"may_spawn_subagent": True, "recursive": True},
            ),
            SubTask(
                id="reassemble",
                description="Combine results",
                goal_hint="Final goal closed",
                suggested_tactics=["exact", "trivial"],
                dependencies=["inner_goals"],
                is_critical=True,
            ),
        ]

        return DecompositionPlan(
            strategy=DecompositionStrategy.HIERARCHICAL,
            subtasks=subtasks,
            synthesis_strategy="nested",
            reasoning="Deep goal structure requires hierarchical decomposition.",
            estimated_complexity=complexity_score.overall_score if complexity_score else 0.7,
        )

    def _detect_induction_variable(self, goal_state: str) -> str:
        """Detect the likely induction variable from goal."""
        # Look for bound variables in forall
        match = re.search(r"∀\s*(\w+)\s*:", goal_state)
        if match:
            return match.group(1)

        match = re.search(r"forall\s+(\w+)", goal_state, re.IGNORECASE)
        if match:
            return match.group(1)

        # Default
        return "n"

    def _detect_case_target(self, goal_state: str) -> Optional[str]:
        """Detect what to split cases on."""
        # Look for hypothesis with sum/or type
        match = re.search(r"(\w+)\s*:\s*(?:Or|Sum|Option|Bool)", goal_state)
        if match:
            return match.group(1)

        # Look for match expression
        match = re.search(r"match\s+(\w+)", goal_state)
        if match:
            return match.group(1)

        return None

    def _estimate_case_count(self, goal_state: str, target: Optional[str]) -> int:
        """Estimate number of cases for a case split."""
        if "Bool" in goal_state:
            return 2
        if "Option" in goal_state:
            return 2  # Some/None
        if "∨" in goal_state or "Or" in goal_state:
            return goal_state.count("∨") + goal_state.count("Or") + 1
        return 2  # Default

    def _detect_definitions(self, goal_state: str, context: str) -> list[str]:
        """Detect definitions that might need unfolding."""
        defs = []

        # Look for capitalized identifiers that might be definitions
        for match in re.finditer(r"\b([A-Z]\w+)\b", goal_state):
            name = match.group(1)
            if name not in ["Nat", "List", "Bool", "True", "False", "Type", "Prop"]:
                defs.append(name)

        # Look for definitions in context
        for match in re.finditer(r"def\s+(\w+)", context):
            defs.append(match.group(1))

        return list(set(defs))[:5]  # Limit to 5

    def _enrich_with_rag(
        self,
        plan: DecompositionPlan,
        rag_results: list[dict],
    ) -> None:
        """Enrich plan with tactics from RAG results."""
        for result in rag_results:
            if "tactic" in result:
                # Add successful tactics from similar proofs
                for subtask in plan.subtasks:
                    if result["tactic"] not in subtask.suggested_tactics:
                        subtask.suggested_tactics.append(result["tactic"])

    async def decompose_with_llm(
        self,
        goal_state: str,
        context: str = "",
        complexity_score: Optional[ComplexityScore] = None,
    ) -> DecompositionPlan:
        """
        LLM-enhanced decomposition planning.

        Uses the LLM to create semantically aware decomposition plans.
        Falls back to rule-based planning if no LLM available.
        """
        # Start with rule-based plan
        plan = await self.decompose(
            goal_state=goal_state,
            context=context,
            complexity_score=complexity_score,
        )

        if not self.llm_provider:
            return plan

        # TODO: LLM enhancement would:
        # 1. Analyze the mathematical structure
        # 2. Suggest specific tactics based on domain
        # 3. Reorder subtasks for efficiency

        logger.debug("LLM-enhanced planning not yet implemented")
        return plan
