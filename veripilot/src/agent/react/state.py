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

    # === Metadata ===
    model_used: str                         # Primary LLM model
    base_temperature: float                 # User-selected base temperature
    start_time: float                       # Unix timestamp when started

    # === Checkpoints (Phase 4) ===
    checkpoint_id: Optional[str]           # Current checkpoint ID if any
    checkpoint_stack: list[str]            # Stack of checkpoint IDs for backtracking


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
    max_attempts: int = 4,
    mode: AgentMode = AgentMode.REACT,
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

    Returns:
        ProofState ready for graph invocation
    """
    import time

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

        # Metadata
        model_used=model_used,
        base_temperature=temperature,
        start_time=time.time(),

        # Checkpoints
        checkpoint_id=None,
        checkpoint_stack=[],
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
