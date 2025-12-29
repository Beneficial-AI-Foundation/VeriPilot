"""
ROMA Aggregator - synthesizes sub-proofs into the final proof.

The Aggregator takes completed subtask proofs and combines them
according to the decomposition plan's synthesis strategy.

Synthesis Strategies:
- SEQUENTIAL: Chain tactics in order (tactic1 <;> tactic2 <;> ...)
- PARALLEL: Independent proofs joined with semicolons or combinators
- NESTED: Hierarchical proof structure with focus/unfocus
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .planner import DecompositionPlan, SubTask, DecompositionStrategy

logger = logging.getLogger(__name__)


class SynthesisResult(str, Enum):
    """Result of synthesis attempt."""

    SUCCESS = "success"
    """All sub-proofs combined successfully."""

    PARTIAL = "partial"
    """Some sub-proofs combined, but gaps remain."""

    FAILED = "failed"
    """Could not synthesize sub-proofs."""


@dataclass
class SubProof:
    """A completed sub-proof for a subtask."""

    subtask_id: str
    """ID of the subtask this proof completes."""

    tactic_sequence: list[str]
    """Sequence of tactics that proved the subtask."""

    success: bool
    """Whether the subtask was proven."""

    error_message: Optional[str] = None
    """Error message if proof failed."""

    attempts: int = 1
    """Number of attempts to prove this subtask."""

    goal_before: str = ""
    """Goal state before tactics applied."""

    goal_after: str = ""
    """Goal state after tactics (should be 'no goals' if success)."""


@dataclass
class AggregationResult:
    """Result of aggregating sub-proofs."""

    result: SynthesisResult
    """Whether synthesis succeeded."""

    combined_proof: str
    """The combined proof tactic sequence."""

    tactic_list: list[str]
    """Individual tactics in execution order."""

    sub_proofs_used: list[str]
    """IDs of sub-proofs that were used."""

    gaps: list[str] = field(default_factory=list)
    """IDs of subtasks that failed or have gaps."""

    reasoning: str = ""
    """Explanation of the synthesis process."""


class RomaAggregator:
    """
    Aggregates sub-proofs into a complete proof.

    The aggregator takes the results from subtask execution and
    combines them according to the plan's synthesis strategy.

    Example:
        aggregator = RomaAggregator()
        result = await aggregator.synthesize(
            plan=decomposition_plan,
            sub_proofs=[proof1, proof2, proof3],
        )
        if result.result == SynthesisResult.SUCCESS:
            # Apply combined_proof
            ...
    """

    # Lean tactic combinators for different strategies
    SEQUENTIAL_COMBINATOR = " <;> "
    PARALLEL_COMBINATOR = "; "
    FOCUS_PREFIX = "· "

    def __init__(self, llm_provider: Optional[object] = None):
        """
        Initialize the aggregator.

        Args:
            llm_provider: Optional LLM for enhanced synthesis.
        """
        self.llm_provider = llm_provider

    async def synthesize(
        self,
        plan: DecompositionPlan,
        sub_proofs: list[SubProof],
    ) -> AggregationResult:
        """
        Synthesize sub-proofs into a combined proof.

        Args:
            plan: The decomposition plan that was executed.
            sub_proofs: Completed sub-proofs from subtask execution.

        Returns:
            AggregationResult with combined proof or gaps.
        """
        # Index sub-proofs by subtask ID
        proof_map = {sp.subtask_id: sp for sp in sub_proofs}

        # Check for missing or failed proofs
        gaps = []
        for subtask in plan.subtasks:
            if subtask.id not in proof_map:
                gaps.append(subtask.id)
            elif not proof_map[subtask.id].success:
                if subtask.is_critical:
                    gaps.append(subtask.id)

        if gaps:
            # Check if we can still make progress
            successful_ids = [
                sp.subtask_id for sp in sub_proofs if sp.success
            ]
            if not successful_ids:
                return AggregationResult(
                    result=SynthesisResult.FAILED,
                    combined_proof="",
                    tactic_list=[],
                    sub_proofs_used=[],
                    gaps=gaps,
                    reasoning=f"Critical subtasks failed: {', '.join(gaps)}",
                )

        # Synthesize based on strategy
        if plan.synthesis_strategy == "parallel":
            result = self._synthesize_parallel(plan, sub_proofs, proof_map)
        elif plan.synthesis_strategy == "nested":
            result = self._synthesize_nested(plan, sub_proofs, proof_map)
        else:  # sequential (default)
            result = self._synthesize_sequential(plan, sub_proofs, proof_map)

        # Add entry/exit tactics if specified
        result = self._wrap_with_entry_exit(plan, result)

        logger.info(
            f"Synthesis {result.result.value}: "
            f"{len(result.sub_proofs_used)} proofs, {len(result.gaps)} gaps"
        )

        return result

    def _synthesize_sequential(
        self,
        plan: DecompositionPlan,
        sub_proofs: list[SubProof],
        proof_map: dict[str, SubProof],
    ) -> AggregationResult:
        """Synthesize proofs in sequential order."""
        tactics = []
        used = []
        gaps = []

        # Process subtasks in dependency order
        for subtask in self._topological_sort(plan.subtasks):
            if subtask.id in proof_map and proof_map[subtask.id].success:
                sp = proof_map[subtask.id]
                tactics.extend(sp.tactic_sequence)
                used.append(subtask.id)
            elif subtask.is_critical:
                gaps.append(subtask.id)
                # For critical gaps, insert a placeholder
                tactics.append(f"sorry -- TODO: {subtask.description}")

        # Determine result status
        if not gaps:
            result = SynthesisResult.SUCCESS
        elif len(used) > 0:
            result = SynthesisResult.PARTIAL
        else:
            result = SynthesisResult.FAILED

        # Combine tactics
        combined = "\n  ".join(tactics) if tactics else ""

        return AggregationResult(
            result=result,
            combined_proof=combined,
            tactic_list=tactics,
            sub_proofs_used=used,
            gaps=gaps,
            reasoning=f"Sequential synthesis: {len(used)} proofs combined.",
        )

    def _synthesize_parallel(
        self,
        plan: DecompositionPlan,
        sub_proofs: list[SubProof],
        proof_map: dict[str, SubProof],
    ) -> AggregationResult:
        """Synthesize independent proofs in parallel structure."""
        branches = []
        used = []
        gaps = []

        for subtask in plan.subtasks:
            if subtask.id in proof_map and proof_map[subtask.id].success:
                sp = proof_map[subtask.id]
                # Each branch as a focused proof
                branch_tactics = sp.tactic_sequence
                if len(branch_tactics) == 1:
                    branches.append(branch_tactics[0])
                else:
                    branches.append(f"({'; '.join(branch_tactics)})")
                used.append(subtask.id)
            elif subtask.is_critical:
                gaps.append(subtask.id)
                branches.append("sorry")

        # Determine result
        if not gaps:
            result = SynthesisResult.SUCCESS
        elif len(used) > 0:
            result = SynthesisResult.PARTIAL
        else:
            result = SynthesisResult.FAILED

        # Combine with parallel combinator
        # For case splits, each branch handles one case
        if plan.strategy == DecompositionStrategy.CASES:
            # Use focused syntax for cases
            combined = "\n".join(
                f"  {self.FOCUS_PREFIX}{b}" for b in branches
            )
        else:
            combined = self.PARALLEL_COMBINATOR.join(branches)

        return AggregationResult(
            result=result,
            combined_proof=combined,
            tactic_list=branches,
            sub_proofs_used=used,
            gaps=gaps,
            reasoning=f"Parallel synthesis: {len(used)} independent proofs.",
        )

    def _synthesize_nested(
        self,
        plan: DecompositionPlan,
        sub_proofs: list[SubProof],
        proof_map: dict[str, SubProof],
    ) -> AggregationResult:
        """Synthesize hierarchical proof structure."""
        lines = []
        used = []
        gaps = []

        # Process in dependency order with nesting
        for subtask in self._topological_sort(plan.subtasks):
            if subtask.id in proof_map and proof_map[subtask.id].success:
                sp = proof_map[subtask.id]
                # Add indentation based on dependencies
                indent = "  " * len(subtask.dependencies)
                for tactic in sp.tactic_sequence:
                    lines.append(f"{indent}{tactic}")
                used.append(subtask.id)
            elif subtask.is_critical:
                gaps.append(subtask.id)
                indent = "  " * len(subtask.dependencies)
                lines.append(f"{indent}sorry -- {subtask.description}")

        # Determine result
        if not gaps:
            result = SynthesisResult.SUCCESS
        elif len(used) > 0:
            result = SynthesisResult.PARTIAL
        else:
            result = SynthesisResult.FAILED

        combined = "\n".join(lines)

        return AggregationResult(
            result=result,
            combined_proof=combined,
            tactic_list=[l.strip() for l in lines if l.strip()],
            sub_proofs_used=used,
            gaps=gaps,
            reasoning=f"Nested synthesis: hierarchical proof structure.",
        )

    def _wrap_with_entry_exit(
        self,
        plan: DecompositionPlan,
        result: AggregationResult,
    ) -> AggregationResult:
        """Add entry and exit tactics to the combined proof."""
        tactics = result.tactic_list.copy()
        combined_parts = []

        # Add entry tactic
        if plan.entry_tactic:
            tactics.insert(0, plan.entry_tactic)
            combined_parts.append(plan.entry_tactic)

        # Add main proof
        combined_parts.append(result.combined_proof)

        # Add exit tactic
        if plan.exit_tactic:
            tactics.append(plan.exit_tactic)
            combined_parts.append(plan.exit_tactic)

        # Rebuild combined proof
        combined = "\n".join(p for p in combined_parts if p)

        return AggregationResult(
            result=result.result,
            combined_proof=combined,
            tactic_list=tactics,
            sub_proofs_used=result.sub_proofs_used,
            gaps=result.gaps,
            reasoning=result.reasoning,
        )

    def _topological_sort(self, subtasks: list[SubTask]) -> list[SubTask]:
        """Sort subtasks by dependencies (topological order)."""
        # Build dependency graph
        id_to_task = {st.id: st for st in subtasks}
        visited = set()
        result = []

        def visit(task_id: str) -> None:
            if task_id in visited:
                return
            visited.add(task_id)

            task = id_to_task.get(task_id)
            if task:
                for dep_id in task.dependencies:
                    visit(dep_id)
                result.append(task)

        for subtask in subtasks:
            visit(subtask.id)

        return result

    async def synthesize_with_verification(
        self,
        plan: DecompositionPlan,
        sub_proofs: list[SubProof],
        verifier_callback,
    ) -> AggregationResult:
        """
        Synthesize and verify the combined proof.

        Args:
            plan: The decomposition plan.
            sub_proofs: Completed sub-proofs.
            verifier_callback: Async function to verify tactics.

        Returns:
            AggregationResult, verified if possible.
        """
        result = await self.synthesize(plan, sub_proofs)

        if result.result != SynthesisResult.SUCCESS:
            return result

        # Verify the combined proof
        if verifier_callback:
            try:
                verified = await verifier_callback(result.combined_proof)
                if not verified:
                    result.result = SynthesisResult.PARTIAL
                    result.reasoning += " Verification failed on combined proof."
            except Exception as e:
                logger.warning(f"Verification error: {e}")
                result.reasoning += f" Verification error: {e}"

        return result

    def create_partial_proof(
        self,
        plan: DecompositionPlan,
        sub_proofs: list[SubProof],
    ) -> str:
        """
        Create a partial proof with sorries for incomplete parts.

        Useful for showing progress even when not fully complete.

        Args:
            plan: The decomposition plan.
            sub_proofs: Available sub-proofs (may be incomplete).

        Returns:
            Proof string with sorries marking incomplete parts.
        """
        proof_map = {sp.subtask_id: sp for sp in sub_proofs if sp.success}
        lines = []

        # Add entry tactic
        if plan.entry_tactic:
            lines.append(plan.entry_tactic)

        # Add subtask proofs or sorries
        for subtask in self._topological_sort(plan.subtasks):
            if subtask.id in proof_map:
                for tactic in proof_map[subtask.id].tactic_sequence:
                    lines.append(f"  {tactic}")
            else:
                lines.append(f"  sorry -- {subtask.description}")

        # Add exit tactic
        if plan.exit_tactic:
            lines.append(plan.exit_tactic)

        return "\n".join(lines)


def quick_combine(tactics: list[str], strategy: str = "sequential") -> str:
    """
    Quick utility to combine tactics without full aggregation.

    Args:
        tactics: List of tactic strings.
        strategy: 'sequential', 'parallel', or 'all'.

    Returns:
        Combined tactic string.
    """
    if not tactics:
        return ""

    if strategy == "sequential":
        return "\n  ".join(tactics)
    elif strategy == "parallel":
        return "; ".join(tactics)
    elif strategy == "all":
        # Use <;> to apply to all goals
        return " <;> ".join(tactics)
    else:
        return "\n  ".join(tactics)
