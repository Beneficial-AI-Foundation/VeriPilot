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
    RecoveryRecord,
    add_thought,
    add_action,
    add_observation,
    dict_to_sorry,
    should_backtrack,
)
from .error_recovery import (
    ErrorRecoveryController,
    TacticModifier,
    RecoveryContext,
    ErrorSeverity,
    RecoveryStrategy,
)
from .roma import (
    GoalComplexity,
    GoalComplexityAnalyzer,
    Atomizer,
    RomaPlanner,
    RomaAggregator,
    SubProof,
)
from .state import SubTaskRecord

if TYPE_CHECKING:
    from verifier.verifier_service import VerifierService

logger = logging.getLogger(__name__)


# ==============================================================================
# Prompt Loading
# ==============================================================================

_REACT_SYSTEM_PROMPT_CACHE: str | None = None


def _load_react_system_prompt() -> str:
    """
    Load the ReAct system prompt from prompts/verifier/react_system_v1.md.

    Uses caching for performance. Falls back to minimal prompt if file not found.
    """
    global _REACT_SYSTEM_PROMPT_CACHE

    if _REACT_SYSTEM_PROMPT_CACHE is not None:
        return _REACT_SYSTEM_PROMPT_CACHE

    try:
        from agent.prompt_loader import load_latest_prompt
        _REACT_SYSTEM_PROMPT_CACHE = load_latest_prompt("react_system")
        logger.debug("Loaded ReAct system prompt from file")
    except (ImportError, FileNotFoundError) as e:
        logger.warning(f"Could not load react_system prompt: {e}, using fallback")
        _REACT_SYSTEM_PROMPT_CACHE = """You are a Lean 4 theorem prover using ReAct (Reasoning + Acting).
Output in this exact format:
THOUGHT: <reasoning about what to try>
TACTIC: <Lean 4 tactic code>
CONFIDENCE: <0.0-1.0>

Use `try grind`, `try omega`, `try ring` for safety. Use `rw [h]` to rewrite with hypotheses."""

    return _REACT_SYSTEM_PROMPT_CACHE


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

    # Build reasoning prompt (user message)
    prompt = _build_reasoning_prompt(state)

    # Load ReAct system prompt
    system_prompt = _load_react_system_prompt()

    # Generate thought and action plan
    try:
        client = LLMClient()

        # Use slightly lower temperature for reasoning
        temperature = max(0.1, state["base_temperature"] - 0.1)

        response = await client.generate(
            prompt,
            model=state["model_used"],
            system_prompt=system_prompt,
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
# Recovery Node (OpenManus Error Recovery)
# ==============================================================================

async def recovery_node(state: ProofState) -> dict[str, Any]:
    """
    OpenManus recovery decision node.

    Analyzes the latest error, determines recovery strategy,
    and modifies the tactic for the next attempt.

    Called when om_router_node routes to "recover" instead of "continue".

    This node:
    1. Gets latest error from observations
    2. Classifies error severity and selects recovery strategy
    3. Modifies tactic using TacticModifier
    4. Returns updated state with modified tactic and recovery record

    Returns:
        Partial state update with modified tactic and recovery metadata
    """
    controller = ErrorRecoveryController()
    modifier = TacticModifier()

    step = state["step"]
    current_tactic = state["current_proof"]
    recovery_stage = state["recovery_stage"]

    # Get latest error from observations
    latest_obs = state["observations"][-1] if state["observations"] else None

    if not latest_obs or latest_obs.get("success"):
        # No error to recover from - shouldn't happen but handle gracefully
        logger.debug("recovery_node called with no error - resetting recovery stage")
        return {
            "recovery_stage": 0,
        }

    error_type = latest_obs.get("error_type", "unknown")
    error_content = latest_obs.get("content", "")

    # Build recovery context from state
    context = RecoveryContext(
        tried_tactics=list(state["tried_tactics"]),
        definitions_to_unfold=state.get("definitions_to_unfold", []),
        successful_tactics=state.get("successful_tactics", []),
        error_content=error_content,
        goal_state=state.get("goal_state", ""),
        attempt_count=state["attempt_count"],
    )

    # Classify error and get strategy
    severity, primary_strategy = controller.classify_error(error_type)

    # Get actual strategy based on recovery stage
    strategy = controller.get_recovery_strategy(
        error_type=error_type,
        recovery_stage=recovery_stage,
        context=context,
    )

    # Check for fatal errors or abort
    if strategy == RecoveryStrategy.ABORT:
        logger.info(f"Recovery aborted for {error_type} (fatal)")
        return {
            "status": ProofStatus.FAILED.value,
            "current_error_type": error_type,
            "current_severity": severity.value,
            "active_strategy": strategy.value,
        }

    # Check for escalation (backtrack or human review)
    if strategy == RecoveryStrategy.BACKTRACK:
        logger.info(f"Recovery escalated to backtrack for {error_type}")
        # For now, just continue with a safe default tactic
        # Phase 4 will implement actual checkpoint restoration
        modified_tactic = "try grind <;> try simp"
    elif strategy == RecoveryStrategy.ESCALATE:
        logger.info(f"Recovery escalated for {error_type} - human review needed")
        modified_tactic = current_tactic  # Keep current, let router decide
    else:
        # Apply recovery strategy to modify tactic
        modified_tactic = modifier.apply_strategy(current_tactic, strategy, context)

    # Extract definitions from error for UNFOLD_MORE strategy
    definitions = modifier.extract_definitions_from_error(error_content)

    # Create recovery record
    recovery_record = RecoveryRecord(
        step=step,
        error_type=error_type,
        severity=severity.value,
        strategy=strategy.value,
        original_tactic=current_tactic,
        modified_tactic=modified_tactic,
        success=False,  # Will be updated by next observation
    )

    logger.info(
        f"Recovery: {error_type} ({severity.value}) -> {strategy.value}: "
        f"{modified_tactic[:50]}..."
    )

    return {
        "current_proof": modified_tactic,
        "current_error_type": error_type,
        "current_severity": severity.value,
        "active_strategy": strategy.value,
        "recovery_stage": recovery_stage + 1,
        "recovery_attempts": state["recovery_attempts"] + 1,
        "tried_tactics": [current_tactic],  # Appended via reducer
        "definitions_to_unfold": definitions if definitions else state.get("definitions_to_unfold", []),
        "recovery_records": [recovery_record],  # Appended via reducer
    }


def reset_recovery_stage(state: ProofState) -> dict[str, Any]:
    """
    Reset recovery stage after a successful verification or new attempt.

    Called when transitioning from recovery back to normal flow.
    """
    return {
        "recovery_stage": 0,
        "current_error_type": None,
        "active_strategy": None,
    }


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
# OpenManus Router Node
# ==============================================================================

def om_router_node(state: ProofState) -> str:
    """
    OpenManus-aware router for OM_REACT mode.

    Routes to "recover" when:
    - Error is RECOVERABLE or TRANSIENT
    - Recovery stage < 2 (haven't exhausted recovery options)
    - Haven't exceeded max recovery attempts

    Returns one of:
    - "continue": Go back to reasoning for another attempt
    - "recover": Route to recovery_node for error analysis
    - "success": Proof verified, terminate
    - "failed": Max attempts or unrecoverable error
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

    # Get latest observation
    latest_obs = state["observations"][-1] if state["observations"] else None

    # If no observation or success, continue normally
    if not latest_obs:
        return "continue"

    if latest_obs.get("success"):
        # Reset recovery stage on success
        return "success"

    # Error occurred - decide whether to recover or continue
    error_type = latest_obs.get("error_type", "unknown")
    recovery_stage = state["recovery_stage"]

    # Check if we should attempt recovery
    if recovery_stage < 2:  # Still have recovery stages available
        # Classify error to check severity
        controller = ErrorRecoveryController()
        severity, _ = controller.classify_error(error_type)

        # Recoverable and transient errors can be recovered
        if severity in (ErrorSeverity.RECOVERABLE, ErrorSeverity.TRANSIENT):
            logger.debug(f"Routing to recovery for {error_type} (stage {recovery_stage})")
            return "recover"

        # Fatal errors skip recovery
        if severity == ErrorSeverity.FATAL:
            logger.info(f"Fatal error {error_type} - skipping recovery")
            return "failed"

    # Recovery exhausted - check for early termination
    if _should_terminate_early(state):
        return "failed"

    # Check for backtrack conditions
    if should_backtrack(state):
        logger.info("Backtrack condition detected")
        # For now, continue (Phase 4 will implement checkpoints)
        return "continue"

    # Continue with next attempt
    return "continue"


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


# ==============================================================================
# ROMA Nodes (Hierarchical Decomposition)
# ==============================================================================

async def complexity_analysis_node(state: ProofState) -> dict[str, Any]:
    """
    Analyze proof goal complexity for ROMA mode.

    This is the first node in the ROMA graph. It:
    1. Analyzes the goal state complexity
    2. Scores nesting, quantifiers, type complexity
    3. Estimates automation likelihood

    Returns:
        Partial state update with complexity analysis results
    """
    analyzer = GoalComplexityAnalyzer()

    goal_state = state["goal_state"]
    context = state["file_content"]
    attempt_count = state["attempt_count"]
    errors = list(state["error_history"])

    # Run complexity analysis
    score = analyzer.analyze(
        goal_state=goal_state,
        context=context,
        previous_attempts=attempt_count,
        previous_errors=errors,
    )

    logger.info(
        f"ROMA complexity: {score.complexity.value} "
        f"(score={score.overall_score:.2f}, auto={score.automation_likelihood:.2f})"
    )

    return {
        "roma_active": True,
        "roma_complexity": score.complexity.value,
        "roma_complexity_score": score.overall_score,
        "step": state["step"] + 1,
    }


async def atomizer_node(state: ProofState) -> dict[str, Any]:
    """
    Decide whether to solve goal directly or decompose.

    This node uses the Atomizer to make the atomic/decompose decision
    based on complexity analysis and attempt history.

    Returns:
        Partial state update with atomizer decision
    """
    atomizer = Atomizer()

    decision = await atomizer.should_decompose(
        goal_state=state["goal_state"],
        context=state["file_content"],
        previous_attempts=state["attempt_count"],
        previous_errors=list(state["error_history"]),
        tried_tactics=list(state["tried_tactics"]),
    )

    logger.info(
        f"ROMA atomizer: {'DECOMPOSE' if decision.should_decompose else 'DIRECT'} "
        f"(confidence={decision.confidence:.2f})"
    )

    return {
        "roma_strategy": decision.suggested_strategy if decision.should_decompose else None,
    }


def roma_router_node(state: ProofState) -> str:
    """
    Route based on atomizer decision.

    Returns one of:
    - "direct": Solve goal directly (atomic)
    - "decompose": Decompose into subtasks
    - "success": Already succeeded
    - "failed": Already failed
    """
    if state["status"] == ProofStatus.SUCCESS.value:
        return "success"

    if state["status"] == ProofStatus.FAILED.value:
        return "failed"

    # Check if decomposition was suggested
    if state.get("roma_strategy"):
        return "decompose"

    return "direct"


async def planner_node(state: ProofState) -> dict[str, Any]:
    """
    Create decomposition plan for complex goal.

    Uses RomaPlanner to break the goal into subtasks with dependencies.

    Returns:
        Partial state update with decomposition plan
    """
    planner = RomaPlanner()

    # Build complexity score from state (for enhanced planning)
    from .roma.complexity import ComplexityScore

    complexity_score = ComplexityScore(
        overall_score=state.get("roma_complexity_score", 0.5),
        complexity=GoalComplexity(state.get("roma_complexity", "simple")),
    )

    plan = await planner.decompose(
        goal_state=state["goal_state"],
        context=state["file_content"],
        complexity_score=complexity_score,
        suggested_strategy=state.get("roma_strategy"),
        rag_results=state.get("rag_results", []),
    )

    logger.info(
        f"ROMA planner: {plan.strategy.value} with {len(plan.subtasks)} subtasks"
    )

    # Serialize plan for state storage
    plan_dict = {
        "strategy": plan.strategy.value,
        "subtasks": [
            {
                "id": st.id,
                "description": st.description,
                "goal_hint": st.goal_hint,
                "suggested_tactics": st.suggested_tactics,
                "dependencies": st.dependencies,
                "is_critical": st.is_critical,
                "max_attempts": st.max_attempts,
            }
            for st in plan.subtasks
        ],
        "synthesis_strategy": plan.synthesis_strategy,
        "entry_tactic": plan.entry_tactic,
        "exit_tactic": plan.exit_tactic,
    }

    # Set first ready subtask
    first_subtask = plan.subtasks[0].id if plan.subtasks else None

    return {
        "roma_plan": plan_dict,
        "roma_current_subtask": first_subtask,
        "current_proof": plan.entry_tactic or state["current_proof"],
    }


async def subtask_executor_node(
    state: ProofState,
    verifier_service: Optional["VerifierService"] = None,
) -> dict[str, Any]:
    """
    Execute the current subtask.

    This node:
    1. Gets the current subtask from the plan
    2. Tries suggested tactics in order
    3. Records results and updates state

    Returns:
        Partial state update with subtask results
    """
    plan_dict = state.get("roma_plan")
    if not plan_dict:
        logger.warning("subtask_executor called without plan")
        return {"status": ProofStatus.FAILED.value}

    current_id = state.get("roma_current_subtask")
    if not current_id:
        logger.warning("No current subtask to execute")
        return {}

    # Find current subtask
    subtask_data = None
    for st in plan_dict.get("subtasks", []):
        if st["id"] == current_id:
            subtask_data = st
            break

    if not subtask_data:
        logger.warning(f"Subtask {current_id} not found in plan")
        return {}

    sorry = dict_to_sorry(state["sorry_location"])
    success = False
    tactics_used = []
    error = None

    # Try suggested tactics
    for tactic in subtask_data.get("suggested_tactics", [])[:5]:  # Max 5 tactics
        tactics_used.append(tactic)

        if verifier_service:
            try:
                success, errors, _ = await verifier_service.verify_proof_on_copy(
                    sorry=sorry,
                    proof_code=tactic,
                    attempt=state["attempt_count"],
                    model_used=state["model_used"],
                    temperature=state["base_temperature"],
                )
                if success:
                    break
                if errors:
                    error = errors[0]
            except Exception as e:
                error = str(e)
        else:
            # Simulate for testing
            success = False
            error = "No verifier service"

    # Record subtask result
    subtask_record = SubTaskRecord(
        subtask_id=current_id,
        description=subtask_data.get("description", ""),
        goal_hint=subtask_data.get("goal_hint", ""),
        tactics_used=tactics_used,
        success=success,
        attempts=len(tactics_used),
        error=error,
    )

    # Create sub-proof if successful
    sub_proofs = list(state.get("roma_sub_proofs", []))
    if success:
        sub_proofs.append({
            "subtask_id": current_id,
            "tactic_sequence": tactics_used,
            "success": True,
        })

    # Update completed subtasks
    completed = list(state.get("roma_completed_subtasks", []))
    if success:
        completed.append(current_id)

    # Find next subtask
    next_subtask = None
    completed_set = set(completed)
    for st in plan_dict.get("subtasks", []):
        st_id = st["id"]
        if st_id in completed_set:
            continue
        deps = st.get("dependencies", [])
        if all(d in completed_set for d in deps):
            next_subtask = st_id
            break

    logger.info(
        f"ROMA subtask {current_id}: {'SUCCESS' if success else 'FAILED'}, "
        f"next={next_subtask}"
    )

    return {
        "roma_subtask_records": [subtask_record],
        "roma_sub_proofs": sub_proofs,
        "roma_completed_subtasks": completed,
        "roma_current_subtask": next_subtask,
        "tried_tactics": tactics_used,
    }


def subtask_router_node(state: ProofState) -> str:
    """
    Route after subtask execution.

    Returns one of:
    - "next_subtask": More subtasks to execute
    - "aggregate": All subtasks done, synthesize proof
    - "failed": Critical subtask failed
    """
    plan_dict = state.get("roma_plan")
    if not plan_dict:
        return "failed"

    completed = set(state.get("roma_completed_subtasks", []))
    all_subtasks = plan_dict.get("subtasks", [])

    # Check if all subtasks completed
    all_done = all(st["id"] in completed for st in all_subtasks)
    if all_done:
        return "aggregate"

    # Check if there's a next subtask to execute
    if state.get("roma_current_subtask"):
        return "next_subtask"

    # Check for critical failures
    records = state.get("roma_subtask_records", [])
    for st in all_subtasks:
        if st.get("is_critical", True):
            st_id = st["id"]
            # Find if this subtask failed
            for rec in records:
                if rec["subtask_id"] == st_id and not rec["success"]:
                    logger.info(f"Critical subtask {st_id} failed")
                    return "failed"

    # If no next subtask but not all done, try to aggregate partial
    return "aggregate"


async def aggregator_node(state: ProofState) -> dict[str, Any]:
    """
    Synthesize sub-proofs into final proof.

    Uses RomaAggregator to combine completed sub-proofs according
    to the plan's synthesis strategy.

    Returns:
        Partial state update with aggregated proof
    """
    aggregator = RomaAggregator()

    plan_dict = state.get("roma_plan")
    if not plan_dict:
        return {"status": ProofStatus.FAILED.value}

    # Reconstruct plan and sub-proofs
    from .roma.planner import DecompositionPlan, SubTask, DecompositionStrategy

    subtasks = [
        SubTask(
            id=st["id"],
            description=st.get("description", ""),
            goal_hint=st.get("goal_hint", ""),
            suggested_tactics=st.get("suggested_tactics", []),
            dependencies=st.get("dependencies", []),
            is_critical=st.get("is_critical", True),
        )
        for st in plan_dict.get("subtasks", [])
    ]

    plan = DecompositionPlan(
        strategy=DecompositionStrategy(plan_dict.get("strategy", "sequential")),
        subtasks=subtasks,
        synthesis_strategy=plan_dict.get("synthesis_strategy", "sequential"),
        entry_tactic=plan_dict.get("entry_tactic"),
        exit_tactic=plan_dict.get("exit_tactic"),
    )

    # Build sub-proofs
    sub_proofs = [
        SubProof(
            subtask_id=sp["subtask_id"],
            tactic_sequence=sp.get("tactic_sequence", []),
            success=sp.get("success", False),
        )
        for sp in state.get("roma_sub_proofs", [])
    ]

    # Synthesize
    result = await aggregator.synthesize(plan, sub_proofs)

    logger.info(
        f"ROMA aggregation: {result.result.value}, "
        f"proofs_used={len(result.sub_proofs_used)}, gaps={len(result.gaps)}"
    )

    from .roma.aggregator import SynthesisResult

    if result.result == SynthesisResult.SUCCESS:
        return {
            "roma_aggregated_proof": result.combined_proof,
            "current_proof": result.combined_proof,
            "status": ProofStatus.SUCCESS.value,
        }
    elif result.result == SynthesisResult.PARTIAL:
        # Partial success - might need more work
        return {
            "roma_aggregated_proof": result.combined_proof,
            "current_proof": result.combined_proof,
        }
    else:
        # Failed synthesis
        return {
            "status": ProofStatus.FAILED.value,
        }
