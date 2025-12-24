"""
LangGraph nodes for ReAct proof verification agent.

Implements the core nodes of the ReAct loop:
- reasoning_node: Generate thought and plan next action
- execution_node: Execute tactic via LSP verification
- observation_node: Parse verification feedback
- router_node: Decide next step (continue/backtrack/terminate)

Each node takes ProofState as input and returns a partial state update.
LangGraph merges updates using Annotated reducers (operator.add for lists).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, TYPE_CHECKING

from .state import (
    ProofState,
    ProofStatus,
    ThoughtRecord,
    ActionRecord,
    ObservationRecord,
    add_thought,
    add_action,
    add_observation,
    dict_to_sorry,
    should_backtrack,
)

if TYPE_CHECKING:
    from verifier.verifier_service import VerifierService

logger = logging.getLogger(__name__)


# ==============================================================================
# Reasoning Node
# ==============================================================================

async def reasoning_node(state: ProofState) -> dict[str, Any]:
    """
    Generate reasoning about the current proof state and plan next action.

    This node:
    1. Analyzes current goal state and error history
    2. Generates a thought about what to try next
    3. Plans the next tactic to attempt

    Uses LLM to generate ReAct-style reasoning.

    Returns:
        Partial state update with new thought and incremented step
    """
    from agent.llm_client import LLMClient
    from agent.prompts import extract_proof_from_response

    step = state["step"] + 1
    attempt = state["attempt_count"]

    # Build reasoning prompt
    prompt = _build_reasoning_prompt(state)

    # Generate thought and action plan
    try:
        client = LLMClient()

        # Use slightly lower temperature for reasoning
        temperature = max(0.1, state["base_temperature"] - 0.1)

        response = await client.generate(
            prompt,
            model=state["model_used"],
            temperature=temperature,
        )

        # Parse thought and action from response
        thought_content, tactic_plan, confidence = _parse_reasoning_response(response)

    except Exception as e:
        logger.warning(f"Reasoning failed: {e}")
        thought_content = f"Reasoning failed: {e}. Will retry with previous approach."
        tactic_plan = state["current_proof"]
        confidence = 0.3

    # Create thought record
    thought = ThoughtRecord(
        step=step,
        content=thought_content,
        tactic_plan=tactic_plan,
        confidence=confidence,
    )

    logger.debug(f"Step {step} thought: {thought_content[:100]}...")

    return {
        "step": step,
        "thoughts": [thought],
        "current_proof": tactic_plan if tactic_plan else state["current_proof"],
    }


def _build_reasoning_prompt(state: ProofState) -> str:
    """Build prompt for reasoning about next action."""
    sorry = dict_to_sorry(state["sorry_location"])

    lines = [
        "# ReAct Proof Reasoning",
        "",
        f"## Task",
        f"Prove the theorem `{sorry.theorem_name}` at line {sorry.line}.",
        "",
        f"## Current State",
        f"- Attempt: {state['attempt_count']}/{state['max_attempts']}",
        f"- Step: {state['step']}",
        "",
    ]

    # Add goal state if available
    if state["goal_state"]:
        lines.extend([
            "## Goal State",
            "```lean",
            state["goal_state"],
            "```",
            "",
        ])

    # Add error history (last 3)
    if state["error_history"]:
        lines.extend([
            "## Recent Errors",
        ])
        for err in state["error_history"][-3:]:
            lines.append(f"- {err[:150]}")
        lines.append("")

    # Add previous attempts
    if state["proof_history"]:
        lines.extend([
            "## Previous Attempts",
        ])
        for i, proof in enumerate(state["proof_history"][-3:], 1):
            lines.append(f"{i}. `{proof[:80]}`")
        lines.append("")

    # Add RAG context if available
    if state["rag_results"]:
        lines.extend([
            "## Relevant Lemmas (from RAG)",
        ])
        for r in state["rag_results"][:5]:
            if isinstance(r, dict):
                name = r.get("name", "")
                sig = r.get("signature", "")[:80]
                lines.append(f"- `{name}`: {sig}")
        lines.append("")

    # Instructions
    lines.extend([
        "## Instructions",
        "",
        "Think step-by-step about what tactic to try next.",
        "Consider:",
        "1. What went wrong in previous attempts?",
        "2. What automation tactics might help? (grind, simp, omega, scalar_tac)",
        "3. Do we need to unfold definitions?",
        "4. Should we use progress* for Aeneas code?",
        "",
        "## Output Format",
        "",
        "THOUGHT: <your reasoning about what to try>",
        "TACTIC: <the exact tactic code to try>",
        "CONFIDENCE: <0.0-1.0 how confident you are>",
    ])

    return "\n".join(lines)


def _parse_reasoning_response(response: str) -> tuple[str, str, float]:
    """
    Parse reasoning response into thought, tactic, and confidence.

    Returns:
        (thought_content, tactic_plan, confidence)
    """
    thought = ""
    tactic = ""
    confidence = 0.5

    lines = response.strip().split("\n")
    current_section = None

    for line in lines:
        line_upper = line.upper().strip()
        if line_upper.startswith("THOUGHT:"):
            thought = line.split(":", 1)[1].strip()
            current_section = "thought"
        elif line_upper.startswith("TACTIC:"):
            tactic = line.split(":", 1)[1].strip()
            current_section = "tactic"
        elif line_upper.startswith("CONFIDENCE:"):
            try:
                conf_str = line.split(":", 1)[1].strip()
                confidence = float(conf_str)
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, IndexError):
                pass
            current_section = None
        elif current_section == "thought" and line.strip():
            thought += " " + line.strip()
        elif current_section == "tactic" and line.strip():
            tactic += "\n" + line.strip()

    # Clean up tactic (remove markdown fences if present)
    tactic = tactic.strip()
    if tactic.startswith("```"):
        tactic = tactic.split("```")[1] if "```" in tactic[3:] else tactic[3:]
        if tactic.startswith("lean"):
            tactic = tactic[4:]
        tactic = tactic.strip()

    return thought.strip(), tactic.strip(), confidence


# ==============================================================================
# Execution Node
# ==============================================================================

async def execution_node(
    state: ProofState,
    verifier_service: Optional["VerifierService"] = None,
) -> dict[str, Any]:
    """
    Execute the planned tactic via LSP verification.

    This node:
    1. Takes the current_proof from state
    2. Creates a verification copy of the file
    3. Runs LSP verification
    4. Returns action record with execution details

    Args:
        state: Current proof state
        verifier_service: VerifierService for LSP verification (optional)

    Returns:
        Partial state update with action record
    """
    sorry = dict_to_sorry(state["sorry_location"])
    proof_code = state["current_proof"]
    step = state["step"]

    # Create action record
    action = ActionRecord(
        step=step,
        action_type="apply_tactic",
        content=proof_code,
        model_used=state["model_used"],
        temperature=state["base_temperature"],
    )

    logger.debug(f"Step {step} executing: {proof_code[:80]}...")

    # Store execution metadata for observation node
    # (actual verification happens in observation_node for cleaner separation)

    return {
        "actions": [action],
        "proof_history": [proof_code],
    }


# ==============================================================================
# Observation Node
# ==============================================================================

async def observation_node(
    state: ProofState,
    verifier_service: Optional["VerifierService"] = None,
) -> dict[str, Any]:
    """
    Get observation from executing the last action.

    This node:
    1. Runs LSP verification on the current proof
    2. Parses the result (success/errors/goals)
    3. Creates observation record
    4. Updates status if proof succeeded

    Args:
        state: Current proof state
        verifier_service: VerifierService for LSP verification

    Returns:
        Partial state update with observation and status
    """
    sorry = dict_to_sorry(state["sorry_location"])
    proof_code = state["current_proof"]
    step = state["step"]
    attempt = state["attempt_count"]

    # Run verification
    success = False
    errors: list[str] = []
    goal_state = ""
    error_type = None

    if verifier_service:
        try:
            success, errors, copy_path = await verifier_service.verify_proof_on_copy(
                sorry=sorry,
                proof_code=proof_code,
                attempt=attempt,
                model_used=state["model_used"],
                temperature=state["base_temperature"],
            )

            # Get goal state if failed
            if not success and hasattr(verifier_service, 'get_goal_state'):
                goal_state = await verifier_service.get_goal_state(
                    sorry.file_path, sorry.line
                )

            # Classify error type
            if errors:
                error_type = _classify_error(errors[0])

        except Exception as e:
            logger.warning(f"Verification error: {e}")
            errors = [str(e)]
            error_type = "verification_error"
    else:
        # No verifier - simulate for testing
        errors = ["No verifier service available"]
        error_type = "no_verifier"

    # Create observation
    observation = ObservationRecord(
        step=step,
        success=success,
        content=errors[0] if errors else "Proof verified successfully",
        error_type=error_type,
        goals_remaining=None,  # TODO: parse from goal state
    )

    logger.debug(f"Step {step} observation: success={success}, errors={len(errors)}")

    # Prepare state updates
    updates: dict[str, Any] = {
        "observations": [observation],
        "goal_state": goal_state if goal_state else state["goal_state"],
    }

    # Update status on success
    if success:
        updates["status"] = ProofStatus.SUCCESS.value
    elif errors:
        updates["error_history"] = errors[:1]  # Add first error to history

    return updates


def _classify_error(error: str) -> str:
    """Classify error type from error message."""
    error_lower = error.lower()

    if "type mismatch" in error_lower:
        return "type_mismatch"
    elif "unknown identifier" in error_lower or "unknown constant" in error_lower:
        return "unknown_identifier"
    elif "unsolved goals" in error_lower:
        return "unsolved_goals"
    elif "expected" in error_lower and "got" in error_lower:
        return "type_mismatch"
    elif "syntax" in error_lower or "unexpected" in error_lower:
        return "syntax_error"
    elif "timeout" in error_lower:
        return "timeout"
    else:
        return "unknown"


# ==============================================================================
# Router Node
# ==============================================================================

def router_node(state: ProofState) -> str:
    """
    Decide the next step based on current state.

    Returns one of:
    - "continue": Go back to reasoning for another attempt
    - "backtrack": Restore to checkpoint (Phase 4)
    - "success": Proof verified, terminate
    - "failed": Max attempts or early termination

    This is a routing function used by LangGraph's conditional edges.
    """
    # Check terminal conditions first
    if state["status"] == ProofStatus.SUCCESS.value:
        return "success"

    if state["status"] == ProofStatus.FAILED.value:
        return "failed"

    # Check max attempts
    if state["attempt_count"] >= state["max_attempts"]:
        logger.info(f"Max attempts ({state['max_attempts']}) reached")
        return "failed"

    # Check for backtrack conditions
    if should_backtrack(state):
        logger.info("Backtrack condition detected")
        # For now, treat backtrack as continue with new attempt
        # Phase 4 will implement actual checkpoint restoration
        return "continue"

    # Check observations for early termination patterns
    if _should_terminate_early(state):
        return "failed"

    # Continue with next attempt
    return "continue"


def _should_terminate_early(state: ProofState) -> bool:
    """
    Check if we should terminate early based on patterns.

    Implements Poetiq self-auditing patterns:
    - Divergence: complexity increasing
    - Oscillation: same error repeating
    """
    # Check oscillation (same error 3 times)
    errors = state["error_history"]
    if len(errors) >= 3:
        if errors[-1] == errors[-2] == errors[-3]:
            logger.info("Oscillation detected: same error 3 times")
            return True

    # Check all observations failed with same error type
    observations = state["observations"]
    if len(observations) >= 3:
        recent = observations[-3:]
        if all(not o["success"] for o in recent):
            error_types = [o.get("error_type") for o in recent]
            if len(set(error_types)) == 1 and error_types[0] is not None:
                logger.info(f"Same error type ({error_types[0]}) 3 times")
                return True

    return False


# ==============================================================================
# Increment Attempt Node
# ==============================================================================

def increment_attempt_node(state: ProofState) -> dict[str, Any]:
    """
    Increment attempt counter when continuing to next iteration.

    This node is called when router returns "continue".
    """
    return {
        "attempt_count": state["attempt_count"] + 1,
    }


# ==============================================================================
# Termination Nodes
# ==============================================================================

def success_node(state: ProofState) -> dict[str, Any]:
    """Mark proof as successful."""
    return {
        "status": ProofStatus.SUCCESS.value,
    }


def failed_node(state: ProofState) -> dict[str, Any]:
    """Mark proof as failed."""
    return {
        "status": ProofStatus.FAILED.value,
    }
