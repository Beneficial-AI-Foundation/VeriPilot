"""
ReAct Agent main entry point for VeriPilot proof verification.

Provides a high-level API for running proof verification with different
agent modes (JUST_RETRY, REACT, OM_REACT, ROMA).

This module is the primary interface between VeriPilot's CLI/core and
the LangGraph-based ReAct agent.

Usage:
    from agent.react import ReActAgent, AgentMode

    # Create agent with desired mode
    agent = ReActAgent(mode=AgentMode.REACT)

    # Run verification
    result = await agent.verify(
        sorry=sorry_location,
        initial_proof=proof_code,
        file_content=file_content,
        verifier_service=verifier,
        rag=rag_instance,
    )

    if result.success:
        print(f"Proof verified: {result.proof_code}")

References:
- CLI menu design: Just Retry → ReAct → OM ReAct → ROMA
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Any

from .state import (
    AgentMode,
    ProofState,
    ProofStatus,
    create_initial_state,
    get_trace_summary,
    dict_to_sorry,
)
from .graph import (
    create_react_graph,
    run_react_verification,
    create_om_react_graph,
    run_om_react_verification,
    create_roma_graph,
    run_roma_verification,
)
from verifier.file_modifier import AttemptLog, write_attempt_log

if TYPE_CHECKING:
    from parser import SorryLocation
    from agent import ProofResult
    from verifier import VerificationResult
    from verifier.verifier_service import VerifierService

logger = logging.getLogger(__name__)


@dataclass
class ReActResult:
    """Result of a ReAct verification run."""

    success: bool
    proof_code: str
    attempts: int
    elapsed_time: float

    # ReAct-specific fields
    thoughts: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)

    # Error information
    errors: list[str] = field(default_factory=list)
    final_status: str = ""

    # Metadata
    mode: str = ""
    model_used: str = ""
    steps: int = 0

    # Output files (for compatibility with VerificationResult)
    output_file: Optional[str] = None
    log_file: Optional[str] = None

    def to_verification_result(self) -> "VerificationResult":
        """Convert to VerificationResult for compatibility."""
        from verifier import VerificationResult

        return VerificationResult(
            success=self.success,
            proof_code=self.proof_code,
            attempts=self.attempts,
            build_output=self.get_trace_summary(),
            errors=self.errors,
            elapsed_time=self.elapsed_time,
            output_file=self.output_file,
            log_file=self.log_file,
        )

    def get_trace_summary(self) -> str:
        """Get human-readable trace summary."""
        lines = [f"=== ReAct Verification ({self.mode}) ==="]
        lines.append(f"Status: {self.final_status}")
        lines.append(f"Attempts: {self.attempts}, Steps: {self.steps}")
        lines.append(f"Time: {self.elapsed_time:.2f}s")
        lines.append("")

        for i, (t, a, o) in enumerate(
            zip(self.thoughts, self.actions, self.observations), 1
        ):
            lines.append(f"--- Step {i} ---")
            if t:
                lines.append(f"THOUGHT: {t.get('content', '')[:100]}")
            if a:
                lines.append(f"ACTION: {a.get('content', '')[:80]}")
            if o:
                status = "✓" if o.get("success") else "✗"
                lines.append(f"OBSERVE {status}: {o.get('content', '')[:80]}")
            lines.append("")

        return "\n".join(lines)


class ReActAgent:
    """
    High-level ReAct agent for proof verification.

    Provides a clean API for running verification with different modes.
    Handles mode dispatch, state management, and result conversion.
    """

    def __init__(
        self,
        mode: AgentMode = AgentMode.REACT,
        max_attempts: int = 4,
        model: str = "gemini-3-pro-preview",
        temperature: float = 0.2,
        project_dir: Optional[str] = None,
    ):
        """
        Initialize ReAct agent.

        Args:
            mode: Verification mode (JUST_RETRY, REACT, OM_REACT, ROMA)
            max_attempts: Maximum verification attempts
            model: LLM model for proof generation
            temperature: Base temperature for LLM calls
            project_dir: Lean project root for import resolution
        """
        self.mode = mode
        self.max_attempts = max_attempts
        self.model = model
        self.temperature = temperature
        self.project_dir = project_dir

    async def verify(
        self,
        sorry: "SorryLocation",
        initial_proof: str,
        file_content: str,
        verifier_service: Optional["VerifierService"] = None,
        rag: Optional[Any] = None,
        goal_state: str = "",
        on_step: Optional[callable] = None,
        project_dir: Optional[str] = None,
    ) -> ReActResult:
        """
        Run proof verification with the configured mode.

        Args:
            sorry: The sorry location to fill
            initial_proof: Initial proof from LLM
            file_content: Full content of the Lean file
            verifier_service: VerifierService for LSP verification
            rag: Optional RAG instance for context retrieval
            goal_state: Initial goal state from LSP
            on_step: Optional callback for step updates
            project_dir: Lean project root (overrides instance default)

        Returns:
            ReActResult with verification outcome and trace
        """
        start_time = time.time()

        # Resolve project_dir (parameter takes precedence over instance attribute)
        resolved_project_dir = project_dir or self.project_dir

        # Dispatch based on mode
        if self.mode == AgentMode.JUST_RETRY:
            return await self._run_just_retry(
                sorry, initial_proof, file_content,
                verifier_service, rag, start_time
            )
        elif self.mode == AgentMode.REACT:
            return await self._run_react(
                sorry, initial_proof, file_content, goal_state,
                verifier_service, rag, on_step, start_time, resolved_project_dir
            )
        elif self.mode == AgentMode.OM_REACT:
            # OpenManus error recovery mode
            return await self._run_om_react(
                sorry, initial_proof, file_content, goal_state,
                verifier_service, rag, on_step, start_time, resolved_project_dir
            )
        elif self.mode == AgentMode.ROMA:
            # ROMA hierarchical decomposition mode
            return await self._run_roma(
                sorry, initial_proof, file_content, goal_state,
                verifier_service, rag, on_step, start_time, resolved_project_dir
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    async def _run_just_retry(
        self,
        sorry: "SorryLocation",
        initial_proof: str,
        file_content: str,
        verifier_service: Optional["VerifierService"],
        rag: Optional[Any],
        start_time: float,
    ) -> ReActResult:
        """
        Run simple retry loop (baseline mode).

        This delegates to the existing verify_proof_lsp function
        for backwards compatibility.
        """
        from agent import ProofResult
        from verifier.retry_handler import verify_proof_lsp

        # Create ProofResult for compatibility
        proof_result = ProofResult(
            success=True,
            proof_code=initial_proof,
            model_used=self.model,
            temperature=self.temperature,
        )

        if verifier_service:
            result = await verify_proof_lsp(
                sorry=sorry,
                proof_result=proof_result,
                verifier_service=verifier_service,
                rag=rag,
                max_attempts=self.max_attempts,
            )

            return ReActResult(
                success=result.success,
                proof_code=result.proof_code,
                attempts=result.attempts,
                elapsed_time=result.elapsed_time,
                errors=result.errors,
                final_status="success" if result.success else "failed",
                mode=AgentMode.JUST_RETRY.value,
                model_used=self.model,
                output_file=result.output_file,
                log_file=result.log_file,
            )
        else:
            # No verifier - return failure
            return ReActResult(
                success=False,
                proof_code=initial_proof,
                attempts=0,
                elapsed_time=time.time() - start_time,
                errors=["No verifier service available"],
                final_status="failed",
                mode=AgentMode.JUST_RETRY.value,
                model_used=self.model,
            )

    async def _run_react(
        self,
        sorry: "SorryLocation",
        initial_proof: str,
        file_content: str,
        goal_state: str,
        verifier_service: Optional["VerifierService"],
        rag: Optional[Any],
        on_step: Optional[callable],
        start_time: float,
        project_dir: Optional[str] = None,
    ) -> ReActResult:
        """Run ReAct verification loop."""
        # Get RAG results if available
        rag_results = []
        if rag:
            try:
                from agent.rag_query import retrieve_context
                rag_results = await retrieve_context(sorry, rag)
                # Convert to list of dicts if needed
                if rag_results and hasattr(rag_results[0], '__dict__'):
                    rag_results = [vars(r) for r in rag_results]
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")

        # Create initial state
        state = create_initial_state(
            sorry=sorry,
            proof_code=initial_proof,
            file_content=file_content,
            goal_state=goal_state,
            rag_results=rag_results,
            model_used=self.model,
            temperature=self.temperature,
            max_attempts=self.max_attempts,
            mode=AgentMode.REACT,
            project_dir=project_dir,
        )

        # Run the graph
        try:
            final_state = await run_react_verification(
                initial_state=state,
                verifier_service=verifier_service,
                on_step=on_step,
            )
        except Exception as e:
            logger.error(f"ReAct graph error: {e}")
            return ReActResult(
                success=False,
                proof_code=initial_proof,
                attempts=state.get("attempt_count", 1),
                elapsed_time=time.time() - start_time,
                errors=[str(e)],
                final_status="error",
                mode=AgentMode.REACT.value,
                model_used=self.model,
            )

        # Build attempt logs from observations
        file_path = str(sorry.file_path)
        attempt_logs = self._build_attempt_logs(final_state, start_time)
        log_file = None
        if attempt_logs:
            try:
                log_file = write_attempt_log(file_path, attempt_logs, format="json")
                logger.info(f"Wrote ReAct log to {log_file}")
            except Exception as e:
                logger.warning(f"Failed to write attempt log: {e}")

        # Convert final state to result
        return self._state_to_result(final_state, start_time, log_file)

    async def _run_om_react(
        self,
        sorry: "SorryLocation",
        initial_proof: str,
        file_content: str,
        goal_state: str,
        verifier_service: Optional["VerifierService"],
        rag: Optional[Any],
        on_step: Optional[callable],
        start_time: float,
        project_dir: Optional[str] = None,
    ) -> ReActResult:
        """Run OpenManus-enhanced ReAct verification loop."""
        # Get RAG results if available
        rag_results = []
        if rag:
            try:
                from agent.rag_query import retrieve_context
                rag_results = await retrieve_context(sorry, rag)
                if rag_results and hasattr(rag_results[0], '__dict__'):
                    rag_results = [vars(r) for r in rag_results]
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")

        # Create initial state
        state = create_initial_state(
            sorry=sorry,
            proof_code=initial_proof,
            file_content=file_content,
            goal_state=goal_state,
            rag_results=rag_results,
            model_used=self.model,
            temperature=self.temperature,
            max_attempts=self.max_attempts,
            mode=AgentMode.OM_REACT,
            project_dir=project_dir,
        )

        # Run the OpenManus-enhanced graph
        try:
            final_state = await run_om_react_verification(
                initial_state=state,
                verifier_service=verifier_service,
                on_step=on_step,
            )
        except Exception as e:
            logger.error(f"OM_REACT graph error: {e}")
            return ReActResult(
                success=False,
                proof_code=initial_proof,
                attempts=state.get("attempt_count", 1),
                elapsed_time=time.time() - start_time,
                errors=[str(e)],
                final_status="error",
                mode=AgentMode.OM_REACT.value,
                model_used=self.model,
            )

        # Build attempt logs including recovery info
        file_path = str(sorry.file_path)
        attempt_logs = self._build_attempt_logs(final_state, start_time)
        log_file = None
        if attempt_logs:
            try:
                log_file = write_attempt_log(file_path, attempt_logs, format="json")
                logger.info(f"Wrote OM_REACT log to {log_file}")
            except Exception as e:
                logger.warning(f"Failed to write attempt log: {e}")

        # Convert final state to result with recovery info
        return self._state_to_result_om(final_state, start_time, log_file)

    async def _run_roma(
        self,
        sorry: "SorryLocation",
        initial_proof: str,
        file_content: str,
        goal_state: str,
        verifier_service: Optional["VerifierService"],
        rag: Optional[Any],
        on_step: Optional[callable],
        start_time: float,
        project_dir: Optional[str] = None,
    ) -> ReActResult:
        """Run ROMA hierarchical decomposition verification loop."""
        # Get RAG results if available
        rag_results = []
        if rag:
            try:
                from agent.rag_query import retrieve_context
                rag_results = await retrieve_context(sorry, rag)
                if rag_results and hasattr(rag_results[0], '__dict__'):
                    rag_results = [vars(r) for r in rag_results]
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")

        # Create initial state
        state = create_initial_state(
            sorry=sorry,
            proof_code=initial_proof,
            file_content=file_content,
            goal_state=goal_state,
            rag_results=rag_results,
            model_used=self.model,
            temperature=self.temperature,
            max_attempts=self.max_attempts,
            mode=AgentMode.ROMA,
            project_dir=project_dir,
        )

        # Run the ROMA graph
        try:
            final_state = await run_roma_verification(
                initial_state=state,
                verifier_service=verifier_service,
                on_step=on_step,
            )
        except Exception as e:
            logger.error(f"ROMA graph error: {e}")
            return ReActResult(
                success=False,
                proof_code=initial_proof,
                attempts=state.get("attempt_count", 1),
                elapsed_time=time.time() - start_time,
                errors=[str(e)],
                final_status="error",
                mode=AgentMode.ROMA.value,
                model_used=self.model,
            )

        # Build attempt logs including ROMA decomposition info
        file_path = str(sorry.file_path)
        attempt_logs = self._build_attempt_logs(final_state, start_time)
        log_file = None
        if attempt_logs:
            try:
                log_file = write_attempt_log(file_path, attempt_logs, format="json")
                logger.info(f"Wrote ROMA log to {log_file}")
            except Exception as e:
                logger.warning(f"Failed to write attempt log: {e}")

        # Convert final state to result with ROMA info
        return self._state_to_result_roma(final_state, start_time, log_file)

    def _state_to_result_roma(
        self,
        state: ProofState,
        start_time: float,
        log_file: Optional[str] = None,
    ) -> ReActResult:
        """Convert ProofState to ReActResult with ROMA decomposition info."""
        result = ReActResult(
            success=state["status"] == ProofStatus.SUCCESS.value,
            proof_code=state.get("roma_aggregated_proof") or state["current_proof"],
            attempts=state["attempt_count"],
            elapsed_time=time.time() - start_time,
            thoughts=list(state["thoughts"]),
            actions=list(state["actions"]),
            observations=list(state["observations"]),
            errors=list(state["error_history"]),
            final_status=state["status"],
            mode=state["mode"],
            model_used=state["model_used"],
            steps=state["step"],
            log_file=log_file,
        )

        # Log ROMA-specific info
        subtask_records = state.get("roma_subtask_records", [])
        if subtask_records:
            completed = [r for r in subtask_records if r.get("success")]
            logger.info(
                f"ROMA completed with {len(completed)}/{len(subtask_records)} subtasks, "
                f"complexity={state.get('roma_complexity', 'unknown')}"
            )

        return result

    def _state_to_result_om(
        self,
        state: ProofState,
        start_time: float,
        log_file: Optional[str] = None,
    ) -> ReActResult:
        """Convert ProofState to ReActResult with OpenManus recovery info."""
        result = ReActResult(
            success=state["status"] == ProofStatus.SUCCESS.value,
            proof_code=state["current_proof"],
            attempts=state["attempt_count"],
            elapsed_time=time.time() - start_time,
            thoughts=list(state["thoughts"]),
            actions=list(state["actions"]),
            observations=list(state["observations"]),
            errors=list(state["error_history"]),
            final_status=state["status"],
            mode=state["mode"],
            model_used=state["model_used"],
            steps=state["step"],
            log_file=log_file,
        )

        # Add recovery-specific info to trace summary
        recovery_records = state.get("recovery_records", [])
        if recovery_records:
            logger.info(
                f"OM_REACT completed with {len(recovery_records)} recovery attempts, "
                f"{state.get('recovery_attempts', 0)} total recoveries"
            )

        return result

    def _state_to_result(
        self,
        state: ProofState,
        start_time: float,
        log_file: Optional[str] = None,
    ) -> ReActResult:
        """Convert ProofState to ReActResult."""
        return ReActResult(
            success=state["status"] == ProofStatus.SUCCESS.value,
            proof_code=state["current_proof"],
            attempts=state["attempt_count"],
            elapsed_time=time.time() - start_time,
            thoughts=list(state["thoughts"]),
            actions=list(state["actions"]),
            observations=list(state["observations"]),
            errors=list(state["error_history"]),
            final_status=state["status"],
            mode=state["mode"],
            model_used=state["model_used"],
            steps=state["step"],
            log_file=log_file,
        )

    def _build_attempt_logs(
        self,
        state: ProofState,
        start_time: float,
    ) -> list[AttemptLog]:
        """Build AttemptLog entries from ReAct state observations."""
        attempt_logs = []
        observations = state.get("observations", [])
        actions = state.get("actions", [])

        for i, obs in enumerate(observations):
            # Get corresponding action if available
            action = actions[i] if i < len(actions) else {}
            proof_code = action.get("content", state.get("current_proof", ""))

            # Extract errors from observation
            errors = []
            if not obs.get("success", False):
                error_content = obs.get("content", "")
                if error_content:
                    errors = [error_content[:500]]  # Truncate long errors

            attempt_logs.append(AttemptLog.create(
                attempt=i + 1,
                proof_code=proof_code,
                build_success=obs.get("success", False),
                errors=errors,
                elapsed_time=time.time() - start_time,
                model_used=state.get("model_used", self.model),
                temperature=state.get("temperature", self.temperature),
            ))

        return attempt_logs


# ==============================================================================
# Convenience Functions
# ==============================================================================

async def verify_with_react(
    sorry: "SorryLocation",
    proof_result: "ProofResult",
    verifier_service: Optional["VerifierService"] = None,
    rag: Optional[Any] = None,
    mode: AgentMode = AgentMode.REACT,
    max_attempts: int = 4,
    project_dir: Optional[str] = None,
) -> "VerificationResult":
    """
    Verify a proof using ReAct agent (convenience function).

    This is a drop-in replacement for verify_proof_lsp that uses
    the ReAct agent instead of simple retry.

    Args:
        sorry: The sorry location
        proof_result: Initial proof from LLM
        verifier_service: VerifierService for LSP verification
        rag: Optional RAG instance
        mode: Agent mode (default: REACT)
        max_attempts: Maximum attempts
        project_dir: Lean project root for import resolution

    Returns:
        VerificationResult for compatibility
    """
    # Read file content
    try:
        with open(sorry.file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
    except Exception:
        file_content = ""

    agent = ReActAgent(
        mode=mode,
        max_attempts=max_attempts,
        model=proof_result.model_used,
        temperature=proof_result.temperature,
        project_dir=project_dir,
    )

    result = await agent.verify(
        sorry=sorry,
        initial_proof=proof_result.proof_code,
        file_content=file_content,
        verifier_service=verifier_service,
        rag=rag,
        project_dir=project_dir,
    )

    return result.to_verification_result()


def get_available_modes() -> list[tuple[str, str, str]]:
    """
    Get list of available verification modes for CLI menu.

    Returns:
        List of (mode_value, display_name, description) tuples
    """
    return [
        (
            AgentMode.JUST_RETRY.value,
            "Just Retry",
            "Simple retry loop (baseline)",
        ),
        (
            AgentMode.REACT.value,
            "ReAct",
            "Reasoning agent (Thought → Action → Observe)",
        ),
        (
            AgentMode.OM_REACT.value,
            "OM ReAct",
            "ReAct + OpenManus error recovery",
        ),
        (
            AgentMode.ROMA.value,
            "ROMA",
            "Hierarchical goal decomposition",
        ),
    ]
