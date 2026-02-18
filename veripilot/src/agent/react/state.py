"""
LangGraph state types for ReAct proof verification agent.

Implements the state management for VeriPilot's ReAct agent:
- TypedDict-based state for LangGraph compatibility
- Annotated reducers for append-only trace fields
- Conversion utilities for existing VeriPilot types
"""

from __future__ import annotations

import operator
from dataclasses import asdict
from enum import Enum
from typing import Annotated, Any, Optional, TypedDict

from parser import SorryLocation


class AgentMode(str, Enum):
    """Verification mode selected via CLI menu."""

    JUST_RETRY = "just_retry"      # Simple retry loop (baseline)
    REACT = "react"                 # ReAct reasoning agent
    OM_REACT = "om_react"          # ReAct + OpenManus error recovery
    ROMA = "roma"                   # Full hierarchical decomposition


class ProofStatus(str, Enum):
    """Status of the proof verification attempt."""

    PENDING = "pending"             # Not started
    IN_PROGRESS = "in_progress"     # Currently verifying
    SUCCESS = "success"             # Proof verified
    FAILED = "failed"               # Max attempts or early termination
    BACKTRACKED = "backtracked"     # Restored to checkpoint


class ThoughtRecord(TypedDict):
    """A single reasoning step in the ReAct trace."""

    step: int                       # Step number in the trace
    content: str                    # The reasoning/thought content
    tactic_plan: str               # Planned tactic based on thought
    confidence: float              # 0.0-1.0 confidence in approach


class ActionRecord(TypedDict):
    """A single action taken by the agent."""

    step: int                       # Step number
    action_type: str               # "apply_tactic", "query_rag", "backtrack", etc.
    content: str                   # The action content (tactic code, query, etc.)
    model_used: Optional[str]      # LLM model if applicable
    temperature: Optional[float]   # Temperature if applicable


class ObservationRecord(TypedDict):
    """Observation from executing an action."""

    step: int                       # Step number
    success: bool                  # Did the action succeed?
    content: str                   # Observation content (errors, goal state, etc.)
    error_type: Optional[str]      # "type_mismatch", "unknown_tactic", etc.
    goals_remaining: Optional[int]  # Number of goals still open


class RecoveryRecord(TypedDict):
    """Record of an OpenManus recovery decision."""

    step: int                       # Step number when recovery was triggered
    error_type: str                 # Error type that triggered recovery
    severity: str                   # ErrorSeverity value
    strategy: str                   # RecoveryStrategy value
    original_tactic: str            # Tactic that failed
    modified_tactic: str            # Modified tactic to try
    success: bool                   # Whether recovery succeeded


class SubTaskRecord(TypedDict):
    """Record of a ROMA subtask execution."""

    subtask_id: str                 # Subtask identifier
    description: str                # What this subtask does
    goal_hint: str                  # Expected goal pattern
    tactics_used: list[str]         # Tactics that were applied
    success: bool                   # Whether subtask was solved
    attempts: int                   # Number of attempts
    error: Optional[str]            # Error if failed


class HypothesisInfo(TypedDict):
    """Information about a single hypothesis in the goal state."""

    name: str                       # Hypothesis name (e.g., "h_limbs0")
    type_str: str                   # Type as string (e.g., "limbs0 < 2^51")
    is_equation: bool               # True if hypothesis is an equation (h : a = b)
    is_bound: bool                  # True if looks like a bound (h : x < N)


class GoalAnalysis(TypedDict):
    """
    Parsed analysis of a Lean goal state.

    Provides structured information extracted from raw goal state strings,
    enabling targeted tactic selection instead of wildcard patterns.

    Example:
        goal_analysis = parse_goal_state(goal_state)
        rewrite_candidates = goal_analysis["rewrite_candidates"]
        # Use specific hypotheses: rw [h_limbs0] instead of rw [*]
    """

    hypotheses: list[HypothesisInfo]    # All parsed hypotheses
    goal_type: str                       # "equality" | "implication" | "forall" | "exists" | "other"
    goal_expr: str                       # The goal expression after ⊢
    rewrite_candidates: list[str]        # Hypothesis names usable with rw (h : a = b)
    bound_hypotheses: list[str]          # Hypothesis names that are bounds
    definitions_in_scope: list[str]      # Definitions that might need unfolding
    goal_summary: str                    # One-line summary for prompt
    hypothesis_count: int                # Total number of hypotheses


class ProofStrategy(TypedDict):
    """
    Proof strategy generated before tactic selection.

    Two-phase prompting: Strategy first, then tactics based on strategy.
    """

    approach: str                        # "direct" | "induction" | "case_split" | "lemma" | "rewriting"
    key_hypotheses: list[str]           # Hypotheses critical for this proof
    intermediate_steps: list[str]       # Helper lemmas needed (have statements)
    tactic_plan: list[str]              # Ordered list of planned tactics
    reasoning: str                       # Why this approach was chosen
    confidence: float                    # 0.0-1.0 confidence in strategy


class ProofState(TypedDict):
    """
    LangGraph state for proof verification.

    This is the central state object passed between graph nodes.
    Uses Annotated types with operator.add for append-only fields,
    enabling LangGraph's built-in state merging.

    Example:
        state = create_initial_state(sorry, proof_result)
        graph = create_react_graph()
        final_state = graph.invoke(state)
    """

    # === Core Context ===
    sorry_location: dict                    # SorryLocation as dict (JSON-serializable)
    file_content: str                       # Full file content for context
    goal_state: str                         # Current proof goal from LSP

    # === Proof State ===
    current_proof: str                      # Current proof attempt
    proof_history: Annotated[list[str], operator.add]  # All attempted proofs

    # === ReAct Trace ===
    thoughts: Annotated[list[ThoughtRecord], operator.add]
    actions: Annotated[list[ActionRecord], operator.add]
    observations: Annotated[list[ObservationRecord], operator.add]

    # === Control ===
    step: int                               # Current step number
    attempt_count: int                      # Current attempt (1-indexed)
    max_attempts: int                       # Maximum attempts allowed
    status: str                             # ProofStatus value
    mode: str                               # AgentMode value

    # === Context ===
    rag_results: list[dict]                # RAG retrieval results
    error_history: Annotated[list[str], operator.add]  # All errors encountered
    project_dir: Optional[str]             # Lean project root directory
    import_context: str                    # Formatted import file contents

    # === Goal Analysis (Phase 4.0) ===
    goal_analysis: Optional[dict]          # GoalAnalysis parsed from goal_state
    previous_goal_state: str               # For tracking progress between attempts

    # === Proof Strategy (Phase 4.1) ===
    proof_strategy: Optional[dict]         # ProofStrategy for current proof attempt

    # === Metadata ===
    model_used: str                         # Primary LLM model
    base_temperature: float                 # User-selected base temperature
    start_time: float                       # Unix timestamp when started

    # === Checkpoints (Phase 4) ===
    # Note: "checkpoint_id" is reserved by LangGraph, so we use "cp_id"
    cp_id: Optional[str]                   # Current checkpoint ID if any
    cp_stack: list[str]                    # Stack of checkpoint IDs for backtracking

    # === OpenManus Recovery Fields ===
    recovery_stage: int                     # 0=fresh, 1=primary strategy, 2=fallback
    recovery_attempts: int                  # Total recovery attempts this sorry
    current_error_type: Optional[str]       # Latest classified error type
    current_severity: Optional[str]         # ErrorSeverity value
    active_strategy: Optional[str]          # RecoveryStrategy value
    tried_tactics: Annotated[list[str], operator.add]  # All tactics tried
    successful_tactics: list[str]           # Tactics that worked (for learning)
    definitions_to_unfold: list[str]        # Definitions identified for unfolding
    recovery_records: Annotated[list[RecoveryRecord], operator.add]  # Recovery trace

    # === Iterative Refinement Fields (Phase 4.4) ===
    tactic_sequence: Annotated[list[str], operator.add]  # Tactics applied successfully
    goal_state_history: list[str]           # Goal state after each tactic
    tactic_step: int                        # Current step in tactic loop
    max_tactic_steps: int                   # Max tactics per attempt (default 15)
    consecutive_failures: int               # Counter for adaptive re-analysis trigger

    # === ROMA Hierarchical Decomposition Fields ===
    roma_active: bool                       # Whether ROMA decomposition is active
    roma_complexity: Optional[str]          # GoalComplexity value (atomic/simple/moderate/complex)
    roma_complexity_score: float            # Overall complexity score 0.0-1.0
    roma_strategy: Optional[str]            # DecompositionStrategy value
    roma_plan: Optional[dict]               # Serialized DecompositionPlan
    roma_current_subtask: Optional[str]     # ID of subtask being worked on
    roma_completed_subtasks: list[str]      # IDs of completed subtasks
    roma_subtask_records: Annotated[list[SubTaskRecord], operator.add]  # Subtask trace
    roma_sub_proofs: list[dict]             # Collected sub-proofs (serialized SubProof)
    roma_aggregated_proof: Optional[str]    # Final aggregated proof if available


def sorry_to_dict(sorry: SorryLocation) -> dict:
    """Convert SorryLocation to JSON-serializable dict."""
    return asdict(sorry)


def dict_to_sorry(d: dict) -> SorryLocation:
    """Convert dict back to SorryLocation."""
    return SorryLocation(**d)


def create_initial_state(
    sorry: SorryLocation,
    proof_code: str,
    file_content: str,
    goal_state: str = "",
    rag_results: Optional[list[dict]] = None,
    model_used: str = "gemini-3-pro-preview",
    temperature: float = 0.2,
    max_attempts: int = 5,
    mode: AgentMode = AgentMode.REACT,
    project_dir: Optional[str] = None,
) -> ProofState:
    """
    Create initial ProofState for a verification run.

    Args:
        sorry: The sorry location to fill
        proof_code: Initial proof from LLM
        file_content: Full content of the Lean file
        goal_state: Initial goal state from LSP (optional)
        rag_results: RAG retrieval results (optional)
        model_used: LLM model name
        temperature: Base temperature for generation
        max_attempts: Maximum verification attempts
        mode: Agent mode (REACT, OM_REACT, ROMA)
        project_dir: Lean project root for import resolution (optional)

    Returns:
        ProofState ready for graph invocation
    """
    import time

    # Load import file contents if project_dir is provided
    import_context = ""
    if project_dir and sorry.imports:
        try:
            from agent.context_formatter import format_import_contents
            import_context = format_import_contents(
                sorry.imports,
                project_dir,
                max_lines_per_file=150,
                max_total_lines=500,
            )
        except Exception:
            # Gracefully handle import loading failures
            pass

    return ProofState(
        # Core context
        sorry_location=sorry_to_dict(sorry),
        file_content=file_content,
        goal_state=goal_state,

        # Proof state
        current_proof=proof_code,
        proof_history=[proof_code],

        # ReAct trace (empty initially)
        thoughts=[],
        actions=[],
        observations=[],

        # Control
        step=0,
        attempt_count=1,
        max_attempts=max_attempts,
        status=ProofStatus.IN_PROGRESS.value,
        mode=mode.value,

        # Context
        rag_results=rag_results or [],
        error_history=[],
        project_dir=project_dir,
        import_context=import_context,

        # Goal Analysis (Phase 4.0)
        goal_analysis=None,
        previous_goal_state="",

        # Proof Strategy (Phase 4.1)
        proof_strategy=None,

        # Metadata
        model_used=model_used,
        base_temperature=temperature,
        start_time=time.time(),

        # Checkpoints
        cp_id=None,
        cp_stack=[],

        # OpenManus Recovery (initialized fresh)
        recovery_stage=0,
        recovery_attempts=0,
        current_error_type=None,
        current_severity=None,
        active_strategy=None,
        tried_tactics=[],
        successful_tactics=[],
        definitions_to_unfold=[],
        recovery_records=[],

        # Iterative Refinement (Phase 4.4)
        tactic_sequence=[],
        goal_state_history=[],
        tactic_step=0,
        max_tactic_steps=15,
        consecutive_failures=0,

        # ROMA Hierarchical Decomposition (initialized fresh)
        roma_active=False,
        roma_complexity=None,
        roma_complexity_score=0.0,
        roma_strategy=None,
        roma_plan=None,
        roma_current_subtask=None,
        roma_completed_subtasks=[],
        roma_subtask_records=[],
        roma_sub_proofs=[],
        roma_aggregated_proof=None,
    )


def add_thought(
    state: ProofState,
    content: str,
    tactic_plan: str = "",
    confidence: float = 0.5,
) -> ThoughtRecord:
    """
    Create a thought record for the current step.

    Note: This creates the record but doesn't add it to state.
    LangGraph nodes should return {"thoughts": [record]} to append.
    """
    return ThoughtRecord(
        step=state["step"],
        content=content,
        tactic_plan=tactic_plan,
        confidence=confidence,
    )


def add_action(
    state: ProofState,
    action_type: str,
    content: str,
    model_used: Optional[str] = None,
    temperature: Optional[float] = None,
) -> ActionRecord:
    """
    Create an action record for the current step.

    Action types:
    - "apply_tactic": Apply a tactic/proof to the goal
    - "query_rag": Query RAG for relevant lemmas
    - "query_lsp": Get goal state from LSP
    - "backtrack": Restore to earlier checkpoint
    - "decompose": ROMA decomposition
    """
    return ActionRecord(
        step=state["step"],
        action_type=action_type,
        content=content,
        model_used=model_used,
        temperature=temperature,
    )


def add_observation(
    state: ProofState,
    success: bool,
    content: str,
    error_type: Optional[str] = None,
    goals_remaining: Optional[int] = None,
) -> ObservationRecord:
    """
    Create an observation record for the current step.

    Error types (from error_normalizer.py):
    - "type_mismatch": Type doesn't match expected
    - "unknown_identifier": Unknown tactic or lemma
    - "unsolved_goals": Goals remain after tactic
    - "syntax_error": Lean syntax error
    - "timeout": Verification timed out
    """
    return ObservationRecord(
        step=state["step"],
        success=success,
        content=content,
        error_type=error_type,
        goals_remaining=goals_remaining,
    )


def get_trace_summary(state: ProofState) -> str:
    """
    Get a human-readable summary of the ReAct trace.

    Useful for debugging and logging.
    """
    lines = [f"=== ReAct Trace (Step {state['step']}) ==="]
    lines.append(f"Status: {state['status']}")
    lines.append(f"Attempts: {state['attempt_count']}/{state['max_attempts']}")
    lines.append("")

    # Interleave thoughts, actions, observations by step
    max_step = max(
        max((t["step"] for t in state["thoughts"]), default=0),
        max((a["step"] for a in state["actions"]), default=0),
        max((o["step"] for o in state["observations"]), default=0),
    )

    for step in range(1, max_step + 1):
        lines.append(f"--- Step {step} ---")

        for t in state["thoughts"]:
            if t["step"] == step:
                lines.append(f"THOUGHT: {t['content'][:100]}...")
                if t["tactic_plan"]:
                    lines.append(f"  Plan: {t['tactic_plan'][:50]}")

        for a in state["actions"]:
            if a["step"] == step:
                lines.append(f"ACTION [{a['action_type']}]: {a['content'][:100]}")

        for o in state["observations"]:
            if o["step"] == step:
                status = "✓" if o["success"] else "✗"
                lines.append(f"OBSERVATION {status}: {o['content'][:100]}")

        lines.append("")

    return "\n".join(lines)


def is_terminal(state: ProofState) -> bool:
    """Check if the state is terminal (success or failed)."""
    return state["status"] in (ProofStatus.SUCCESS.value, ProofStatus.FAILED.value)


def should_backtrack(state: ProofState) -> bool:
    """
    Determine if we should backtrack based on error patterns.

    Backtrack conditions:
    - Same error repeated 2+ times (oscillation)
    - Type mismatch after 3+ attempts
    - No progress in goal count
    """
    errors = state["error_history"]
    if len(errors) < 2:
        return False

    # Check for oscillation (same error twice in a row)
    if len(errors) >= 2 and errors[-1] == errors[-2]:
        return True

    # Check observations for repeated failures
    recent_obs = state["observations"][-3:] if len(state["observations"]) >= 3 else []
    if all(not o["success"] for o in recent_obs):
        error_types = [o.get("error_type") for o in recent_obs]
        if error_types.count("type_mismatch") >= 2:
            return True

    return False
