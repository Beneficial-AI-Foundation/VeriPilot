"""
Retry handler for VeriPilot verification loop.

Orchestrates the proof verification cycle:
1. Replace sorry with generated proof
2. Run lake build
3. Parse errors
4. Retry with error feedback (up to max_attempts)

Implements Poetiq patterns (POETIQ_deep_dive.md):
- Section 2.1: Iterative refinement through feedback loops
- Section 2.3: Self-auditing and termination logic
- Section 4.4: Error message normalization for LLM-friendly feedback
"""

import logging
import time
from typing import Optional, TYPE_CHECKING

from . import VerificationResult, BuildResult
from .file_modifier import (
    backup_file,
    restore_file,
    cleanup_backup,
    replace_sorry,
    file_contains_sorry,
    create_attempt_copy,
    write_attempt_log,
    cleanup_intermediate_attempts,
    AttemptLog,
)
from .lake_runner import run_lake_build, get_module_from_file
from .error_parser import (
    parse_lean_errors,
    extract_error_summary,
    filter_errors_for_file,
)
from .self_audit import (
    SelfAuditingController,
    AuditConfig,
    estimate_goal_complexity,
    estimate_tokens,
)
from .error_normalizer import ErrorMessageNormalizer, format_error_for_prompt

if TYPE_CHECKING:
    from parser import SorryLocation
    from agent import ProofResult
    from .verifier_service import VerifierService

logger = logging.getLogger(__name__)


# Default project directory for dalek benchmark
DEFAULT_PROJECT_DIR = "/workspace/projects/VeriPilot/lean-projects/dalek-verify-lean"


async def verify_proof(
    sorry: "SorryLocation",
    proof_result: "ProofResult",
    rag=None,  # LeanRAG instance for regeneration
    max_attempts: int = 4,
    project_dir: str = DEFAULT_PROJECT_DIR,
    timeout: int = 300,
    audit_config: Optional[AuditConfig] = None,
) -> VerificationResult:
    """
    Verify a proof with retry loop and Poetiq self-auditing.

    This is the main entry point for proof verification. It:
    1. Backs up the original file
    2. Replaces sorry with the generated proof
    3. Runs lake build to verify
    4. If errors, checks self-audit for early termination
    5. Regenerates proof with normalized error feedback
    6. Accumulates tactic history across iterations
    7. Repeats up to max_attempts times or until self-audit stops
    8. Restores original on failure

    Poetiq patterns implemented:
    - Iterative refinement (Section 2.1)
    - Self-auditing termination (Section 2.3)
    - Error message normalization (Section 4.4)
    - Context accumulation (Section 1.2)

    Args:
        sorry: The sorry location being filled
        proof_result: Initial proof from agent
        rag: Optional LeanRAG instance for regeneration
        max_attempts: Maximum verification attempts
        project_dir: Lean project directory for lake build
        timeout: Lake build timeout in seconds
        audit_config: Optional self-auditing configuration

    Returns:
        VerificationResult with success status and details
    """
    start_time = time.time()
    file_path = str(sorry.file_path)
    current_proof = proof_result.proof_code
    all_errors: list[str] = []
    attempt_logs: list[AttemptLog] = []  # Track all attempts for logging

    # Initialize Poetiq self-auditing controller
    audit_controller = SelfAuditingController(
        audit_config or AuditConfig(max_iterations=max_attempts)
    )
    error_normalizer = ErrorMessageNormalizer()

    # Backup original file before any modifications
    backup_path = backup_file(file_path)

    try:
        for attempt in range(1, max_attempts + 1):
            # Poetiq: Check self-audit before each attempt (after first)
            if attempt > 1:
                should_continue, stop_reason = audit_controller.should_continue()
                if not should_continue:
                    logger.info(f"Early termination at attempt {attempt}: {stop_reason}")

                    # Create output file from last attempted proof
                    output_file = create_attempt_copy(file_path, attempt - 1)
                    log_file = write_attempt_log(file_path, attempt_logs, format="json")

                    restore_file(file_path, backup_path)
                    cleanup_backup(backup_path)
                    return VerificationResult(
                        success=False,
                        proof_code=current_proof,
                        attempts=attempt - 1,
                        build_output="",
                        errors=all_errors + [f"Early termination: {stop_reason}"],
                        elapsed_time=time.time() - start_time,
                        output_file=output_file,
                        log_file=log_file,
                    )

            # Replace sorry with current proof
            success = replace_sorry(file_path, sorry, current_proof)
            if not success:
                # Restore and fail
                restore_file(file_path, backup_path)
                return VerificationResult(
                    success=False,
                    proof_code=current_proof,
                    attempts=attempt,
                    build_output="",
                    errors=["Failed to replace sorry in file"],
                    elapsed_time=time.time() - start_time,
                )

            # Run lake build
            build_result = await run_lake_build(project_dir, timeout=timeout)

            if build_result.success:
                # Check that sorry is actually gone
                if not file_contains_sorry(file_path):
                    # Success! Record successful tactic and clean up
                    audit_controller.record_attempt(
                        error=None,
                        goal_complexity=0,
                        tactic=current_proof,
                        success=True,
                    )

                    # Log this successful attempt
                    attempt_logs.append(AttemptLog.create(
                        attempt=attempt,
                        proof_code=current_proof,
                        build_success=True,
                        errors=[],
                        elapsed_time=time.time() - start_time,
                        model_used=proof_result.model_used,
                        temperature=proof_result.temperature,
                    ))

                    # Create final attempt copy (e.g., Invert_VP2.lean)
                    output_file = create_attempt_copy(file_path, attempt)

                    # Write attempt log
                    log_file = write_attempt_log(file_path, attempt_logs, format="json")

                    # Cleanup intermediate attempts (keep only successful one)
                    if attempt > 1:
                        cleanup_intermediate_attempts(file_path, attempt)

                    cleanup_backup(backup_path)
                    logger.info(f"Proof verified on attempt {attempt}")
                    return VerificationResult(
                        success=True,
                        proof_code=current_proof,
                        attempts=attempt,
                        build_output=build_result.stdout,
                        elapsed_time=time.time() - start_time,
                        output_file=output_file,
                        log_file=log_file,
                    )
                # Build succeeded but sorry still present (shouldn't happen)
                # Continue to retry

            # Parse errors from build output
            combined_output = f"{build_result.stdout}\n{build_result.stderr}"
            errors = parse_lean_errors(combined_output)

            # Filter to errors from our file
            file_errors = filter_errors_for_file(errors, file_path)
            error_summary = extract_error_summary(file_errors or errors)

            # Poetiq: Normalize error for LLM and record attempt
            normalized_error = error_normalizer.normalize(error_summary)
            goal_complexity = estimate_goal_complexity(error_summary)
            prompt_tokens = estimate_tokens(current_proof + error_summary)

            audit_controller.record_attempt(
                error=normalized_error.normalized,
                goal_complexity=goal_complexity,
                tokens=prompt_tokens,
                tactic=current_proof,
                success=False,
            )

            # Log this failed attempt
            attempt_logs.append(AttemptLog.create(
                attempt=attempt,
                proof_code=current_proof,
                build_success=False,
                errors=[normalized_error.normalized],
                elapsed_time=time.time() - start_time,
                model_used=proof_result.model_used,
                temperature=proof_result.temperature,
            ))

            all_errors.append(f"Attempt {attempt}: {normalized_error.normalized}")

            # Check if we've exhausted attempts
            if attempt >= max_attempts:
                # Create final attempt copy (e.g., Invert_VP4.lean)
                output_file = create_attempt_copy(file_path, attempt)

                # Write attempt log
                log_file = write_attempt_log(file_path, attempt_logs, format="json")

                # Restore original and return failure
                restore_file(file_path, backup_path)
                cleanup_backup(backup_path)
                audit_summary = audit_controller.get_summary()
                logger.info(f"Max attempts reached. Audit: {audit_summary}")
                return VerificationResult(
                    success=False,
                    proof_code=current_proof,
                    attempts=attempt,
                    build_output=combined_output,
                    errors=all_errors,
                    elapsed_time=time.time() - start_time,
                    output_file=output_file,
                    log_file=log_file,
                )

            # Restore file before regenerating (need original sorry for context)
            restore_file(file_path, backup_path)

            # Regenerate proof with error feedback and tactic history
            new_proof = await _regenerate_with_feedback(
                sorry=sorry,
                file_path=file_path,
                prev_proof=current_proof,
                error=normalized_error.normalized,
                attempt=attempt + 1,
                model=proof_result.model_used,
                base_temperature=proof_result.temperature,
                rag=rag,
                suggestion=normalized_error.suggestion,
                successful_tactics=audit_controller.state.successful_tactics,
                failed_tactics=audit_controller.state.failed_tactics,
                project_dir=project_dir,
            )

            if new_proof:
                current_proof = new_proof
            # else: retry with same proof (maybe transient error)

        # Should not reach here, but handle gracefully
        output_file = create_attempt_copy(file_path, max_attempts) if attempt_logs else None
        log_file = write_attempt_log(file_path, attempt_logs, format="json") if attempt_logs else None

        restore_file(file_path, backup_path)
        cleanup_backup(backup_path)
        return VerificationResult(
            success=False,
            proof_code=current_proof,
            attempts=max_attempts,
            build_output="",
            errors=all_errors or ["Max attempts reached"],
            elapsed_time=time.time() - start_time,
            output_file=output_file,
            log_file=log_file,
        )

    except Exception as e:
        # Ensure we restore on any exception
        output_file = None
        log_file = None
        try:
            if attempt_logs:
                output_file = create_attempt_copy(file_path, len(attempt_logs))
                log_file = write_attempt_log(file_path, attempt_logs, format="json")
            restore_file(file_path, backup_path)
            cleanup_backup(backup_path)
        except Exception:
            pass
        return VerificationResult(
            success=False,
            proof_code=current_proof,
            attempts=len(attempt_logs) if attempt_logs else 1,
            build_output="",
            errors=[f"Exception during verification: {e}"],
            elapsed_time=time.time() - start_time,
            output_file=output_file,
            log_file=log_file,
        )


async def _regenerate_with_feedback(
    sorry: "SorryLocation",
    file_path: str,
    prev_proof: str,
    error: str,
    attempt: int,
    model: str,
    base_temperature: float = 0.2,
    rag=None,
    suggestion: str = "",
    successful_tactics: Optional[list[str]] = None,
    failed_tactics: Optional[list[str]] = None,
    project_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Regenerate proof using error feedback and tactic history.

    Implements Poetiq context accumulation (Section 1.2):
    - Successful tactics from prior iterations become positive examples
    - Failed tactics become negative examples ("this didn't work")
    - Normalized error with suggestion guides the next attempt

    Args:
        sorry: The sorry location
        file_path: Path to the Lean file
        prev_proof: Previous proof that failed
        error: Normalized error message from failed build
        attempt: Current attempt number (2-4)
        model: LLM model to use
        base_temperature: User's selected base temperature
        rag: Optional RAG instance
        suggestion: LLM-friendly suggestion for fixing the error
        successful_tactics: Tactics that worked in similar contexts
        failed_tactics: Tactics that have already failed

    Returns:
        New proof code, or None if regeneration fails
    """
    try:
        # Import agent module
        from agent.llm_client import LLMClient
        from agent.prompts import build_retry_prompt, extract_proof_from_response
        from agent.context_formatter import (
            format_context,
            format_error_context,
            format_tactic_history,
        )
        from agent.rag_query import retrieve_context

        # Read current file content
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()

        # Get RAG context if available
        rag_results = []
        if rag is not None:
            try:
                rag_results = await retrieve_context(sorry, rag)
            except Exception:
                pass

        # Build context with error information (include imports if project_dir available)
        context = format_context(sorry, file_content, rag_results, project_dir=project_dir)
        error_context = format_error_context(error, prev_proof)

        # Poetiq: Add tactic history for context accumulation
        tactic_history = format_tactic_history(
            successful_tactics or [],
            failed_tactics or [],
        )

        # Combine all context sections
        context_parts = [context, error_context]
        if tactic_history:
            context_parts.append(tactic_history)
        if suggestion:
            context_parts.append(f"## Suggestion\n\n{suggestion}")

        full_context = "\n\n".join(context_parts)

        # Build retry prompt with attempt-specific guidance
        prompt = build_retry_prompt(sorry, full_context, prev_proof, error, attempt)

        # Generate new proof
        client = LLMClient()

        # Adjust temperature based on attempt and user's base temperature
        # Later attempts use slightly higher temperature for diversity (Poetiq pattern)
        # e.g., if base=0.2: attempt 2→0.275, attempt 3→0.35, attempt 4→0.425
        temperature = base_temperature + (attempt - 1) * 0.075

        response = await client.generate(
            prompt,
            model=model,
            temperature=min(temperature, 0.8),
        )

        return extract_proof_from_response(response)

    except Exception as e:
        logger.warning(f"Failed to regenerate proof: {e}")
        return None


async def verify_proof_lsp(
    sorry: "SorryLocation",
    proof_result: "ProofResult",
    verifier_service: "VerifierService",
    rag=None,
    max_attempts: int = 4,
    audit_config: Optional[AuditConfig] = None,
) -> VerificationResult:
    """
    Verify a proof using LSP (instant feedback, no file overwriting).

    CRITICAL: This function NEVER modifies the original file.
    All work is done on _VPN.lean copies.

    Uses VerifierService which:
    - Provides instant feedback via MCP lean-lsp (~10ms when warm)
    - Falls back to lake build if MCP not ready

    Args:
        sorry: The sorry location being filled
        proof_result: Initial proof from agent
        verifier_service: VerifierService instance (should be started)
        rag: Optional LeanRAG instance for regeneration
        max_attempts: Maximum verification attempts
        audit_config: Optional self-auditing configuration

    Returns:
        VerificationResult with success status and details
    """
    from .verifier_service import VerifierService

    start_time = time.time()
    file_path = str(sorry.file_path)
    current_proof = proof_result.proof_code
    all_errors: list[str] = []
    attempt_logs: list[AttemptLog] = []

    # Initialize Poetiq self-auditing controller
    audit_controller = SelfAuditingController(
        audit_config or AuditConfig(max_iterations=max_attempts)
    )
    error_normalizer = ErrorMessageNormalizer()

    try:
        for attempt in range(1, max_attempts + 1):
            # Poetiq: Check self-audit before each attempt (after first)
            if attempt > 1:
                should_continue, stop_reason = audit_controller.should_continue()
                if not should_continue:
                    logger.info(f"Early termination at attempt {attempt}: {stop_reason}")

                    # Keep the last attempt file
                    last_copy = create_attempt_copy(file_path, attempt - 1)
                    log_file = write_attempt_log(file_path, attempt_logs, format="json")

                    return VerificationResult(
                        success=False,
                        proof_code=current_proof,
                        attempts=attempt - 1,
                        build_output="",
                        errors=all_errors + [f"Early termination: {stop_reason}"],
                        elapsed_time=time.time() - start_time,
                        output_file=last_copy,
                        log_file=log_file,
                    )

            # Verify on copy (NEVER touches original!)
            success, errors, copy_path = await verifier_service.verify_proof_on_copy(
                sorry=sorry,
                proof_code=current_proof,
                attempt=attempt,
                model_used=proof_result.model_used,
                temperature=proof_result.temperature,
            )

            if success:
                # Record successful tactic
                audit_controller.record_attempt(
                    error=None,
                    goal_complexity=0,
                    tactic=current_proof,
                    success=True,
                )

                # Log this successful attempt
                attempt_logs.append(AttemptLog.create(
                    attempt=attempt,
                    proof_code=current_proof,
                    build_success=True,
                    errors=[],
                    elapsed_time=time.time() - start_time,
                    model_used=proof_result.model_used,
                    temperature=proof_result.temperature,
                ))

                # Write attempt log
                log_file = write_attempt_log(file_path, attempt_logs, format="json")

                # Cleanup intermediate attempts (keep only successful one)
                if attempt > 1:
                    cleanup_intermediate_attempts(file_path, attempt)

                method = verifier_service.status.last_verification_method
                logger.info(f"Proof verified on attempt {attempt} via {method}")
                return VerificationResult(
                    success=True,
                    proof_code=current_proof,
                    attempts=attempt,
                    build_output=f"Verified via {method}",
                    elapsed_time=time.time() - start_time,
                    output_file=copy_path,
                    log_file=log_file,
                )

            # Failed - normalize errors
            error_summary = "\n".join(errors[:3]) if errors else "Unknown error"
            normalized_error = error_normalizer.normalize(error_summary)
            goal_complexity = estimate_goal_complexity(error_summary)
            prompt_tokens = estimate_tokens(current_proof + error_summary)

            audit_controller.record_attempt(
                error=normalized_error.normalized,
                goal_complexity=goal_complexity,
                tokens=prompt_tokens,
                tactic=current_proof,
                success=False,
            )

            # Log this failed attempt
            attempt_logs.append(AttemptLog.create(
                attempt=attempt,
                proof_code=current_proof,
                build_success=False,
                errors=[normalized_error.normalized],
                elapsed_time=time.time() - start_time,
                model_used=proof_result.model_used,
                temperature=proof_result.temperature,
            ))

            all_errors.append(f"Attempt {attempt}: {normalized_error.normalized}")

            # Check if we've exhausted attempts
            if attempt >= max_attempts:
                log_file = write_attempt_log(file_path, attempt_logs, format="json")
                audit_summary = audit_controller.get_summary()
                logger.info(f"Max attempts reached. Audit: {audit_summary}")
                return VerificationResult(
                    success=False,
                    proof_code=current_proof,
                    attempts=attempt,
                    build_output="",
                    errors=all_errors,
                    elapsed_time=time.time() - start_time,
                    output_file=copy_path,
                    log_file=log_file,
                )

            # Regenerate proof with error feedback (use original file for context)
            new_proof = await _regenerate_with_feedback(
                sorry=sorry,
                file_path=file_path,  # Read original for context
                prev_proof=current_proof,
                error=normalized_error.normalized,
                attempt=attempt + 1,
                model=proof_result.model_used,
                base_temperature=proof_result.temperature,
                rag=rag,
                suggestion=normalized_error.suggestion,
                successful_tactics=audit_controller.state.successful_tactics,
                failed_tactics=audit_controller.state.failed_tactics,
                project_dir=verifier_service.project_dir,
            )

            if new_proof:
                current_proof = new_proof

        # Should not reach here
        log_file = write_attempt_log(file_path, attempt_logs, format="json") if attempt_logs else None
        return VerificationResult(
            success=False,
            proof_code=current_proof,
            attempts=max_attempts,
            build_output="",
            errors=all_errors or ["Max attempts reached"],
            elapsed_time=time.time() - start_time,
            log_file=log_file,
        )

    except Exception as e:
        log_file = None
        try:
            if attempt_logs:
                log_file = write_attempt_log(file_path, attempt_logs, format="json")
        except Exception:
            pass
        return VerificationResult(
            success=False,
            proof_code=current_proof,
            attempts=len(attempt_logs) if attempt_logs else 1,
            build_output="",
            errors=[f"Exception during verification: {e}"],
            elapsed_time=time.time() - start_time,
            log_file=log_file,
        )


async def verify_single_sorry(
    file_path: str,
    line: int,
    proof: str,
    project_dir: str = DEFAULT_PROJECT_DIR,
    timeout: int = 300,
) -> VerificationResult:
    """
    Simple verification of a single sorry replacement.

    Does not retry or regenerate - just tests if the proof compiles.

    Args:
        file_path: Path to the Lean file
        line: Line number of sorry (1-indexed)
        proof: Proof code to insert
        project_dir: Lean project directory
        timeout: Build timeout

    Returns:
        VerificationResult
    """
    from parser import SorryLocation

    # Create minimal SorryLocation
    sorry = SorryLocation(
        file_path=file_path,
        line=line,
        column=1,
        theorem_name="",
        theorem_signature="",
        proof_prefix="",
        namespace="",
        imports=[],
    )

    start_time = time.time()
    backup_path = backup_file(file_path)

    try:
        # Replace sorry
        success = replace_sorry(file_path, sorry, proof)
        if not success:
            restore_file(file_path, backup_path)
            return VerificationResult(
                success=False,
                proof_code=proof,
                attempts=1,
                build_output="",
                errors=["Failed to replace sorry"],
                elapsed_time=time.time() - start_time,
            )

        # Build
        build_result = await run_lake_build(project_dir, timeout=timeout)

        # Always restore for single verification
        restore_file(file_path, backup_path)
        cleanup_backup(backup_path)

        combined_output = f"{build_result.stdout}\n{build_result.stderr}"

        if build_result.success and not file_contains_sorry(file_path):
            return VerificationResult(
                success=True,
                proof_code=proof,
                attempts=1,
                build_output=build_result.stdout,
                elapsed_time=time.time() - start_time,
            )

        # Parse errors for feedback
        errors = parse_lean_errors(combined_output)
        error_messages = [e.message for e in errors[:5]]

        return VerificationResult(
            success=False,
            proof_code=proof,
            attempts=1,
            build_output=combined_output,
            errors=error_messages or ["Build failed"],
            elapsed_time=time.time() - start_time,
        )

    except Exception as e:
        restore_file(file_path, backup_path)
        cleanup_backup(backup_path)
        return VerificationResult(
            success=False,
            proof_code=proof,
            attempts=1,
            build_output="",
            errors=[str(e)],
            elapsed_time=time.time() - start_time,
        )
