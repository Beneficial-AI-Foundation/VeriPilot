"""
Retry handler for VeriPilot verification loop.

Orchestrates the proof verification cycle:
1. Replace sorry with generated proof
2. Run lake build
3. Parse errors
4. Retry with error feedback (up to max_attempts)
"""

import time
from typing import Optional, TYPE_CHECKING

from . import VerificationResult, BuildResult
from .file_modifier import (
    backup_file,
    restore_file,
    cleanup_backup,
    replace_sorry,
    file_contains_sorry,
)
from .lake_runner import run_lake_build, get_module_from_file
from .error_parser import (
    parse_lean_errors,
    extract_error_summary,
    filter_errors_for_file,
)

if TYPE_CHECKING:
    from parser import SorryLocation
    from agent import ProofResult


# Default project directory for dalek benchmark
DEFAULT_PROJECT_DIR = "/workspace/projects/VeriPilot/lean-projects/dalek-verify-lean"


async def verify_proof(
    sorry: "SorryLocation",
    proof_result: "ProofResult",
    rag=None,  # LeanRAG instance for regeneration
    max_attempts: int = 4,
    project_dir: str = DEFAULT_PROJECT_DIR,
    timeout: int = 300,
) -> VerificationResult:
    """
    Verify a proof with retry loop.

    This is the main entry point for proof verification. It:
    1. Backs up the original file
    2. Replaces sorry with the generated proof
    3. Runs lake build to verify
    4. If errors, regenerates proof with error feedback
    5. Repeats up to max_attempts times
    6. Restores original on failure

    Args:
        sorry: The sorry location being filled
        proof_result: Initial proof from agent
        rag: Optional LeanRAG instance for regeneration
        max_attempts: Maximum verification attempts
        project_dir: Lean project directory for lake build
        timeout: Lake build timeout in seconds

    Returns:
        VerificationResult with success status and details
    """
    start_time = time.time()
    file_path = str(sorry.file_path)
    current_proof = proof_result.proof_code
    all_errors: list[str] = []

    # Backup original file before any modifications
    backup_path = backup_file(file_path)

    try:
        for attempt in range(1, max_attempts + 1):
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
                    # Success! Clean up backup and return
                    cleanup_backup(backup_path)
                    return VerificationResult(
                        success=True,
                        proof_code=current_proof,
                        attempts=attempt,
                        build_output=build_result.stdout,
                        elapsed_time=time.time() - start_time,
                    )
                # Build succeeded but sorry still present (shouldn't happen)
                # Continue to retry

            # Parse errors from build output
            combined_output = f"{build_result.stdout}\n{build_result.stderr}"
            errors = parse_lean_errors(combined_output)

            # Filter to errors from our file
            file_errors = filter_errors_for_file(errors, file_path)
            error_summary = extract_error_summary(file_errors or errors)

            all_errors.append(f"Attempt {attempt}: {error_summary}")

            # Check if we've exhausted attempts
            if attempt >= max_attempts:
                # Restore original and return failure
                restore_file(file_path, backup_path)
                cleanup_backup(backup_path)
                return VerificationResult(
                    success=False,
                    proof_code=current_proof,
                    attempts=attempt,
                    build_output=combined_output,
                    errors=all_errors,
                    elapsed_time=time.time() - start_time,
                )

            # Restore file before regenerating (need original sorry for context)
            restore_file(file_path, backup_path)

            # Regenerate proof with error feedback
            new_proof = await _regenerate_with_feedback(
                sorry=sorry,
                file_path=file_path,
                prev_proof=current_proof,
                error=error_summary,
                attempt=attempt + 1,
                model=proof_result.model_used,
                rag=rag,
            )

            if new_proof:
                current_proof = new_proof
            # else: retry with same proof (maybe transient error)

        # Should not reach here, but handle gracefully
        restore_file(file_path, backup_path)
        cleanup_backup(backup_path)
        return VerificationResult(
            success=False,
            proof_code=current_proof,
            attempts=max_attempts,
            build_output="",
            errors=all_errors or ["Max attempts reached"],
            elapsed_time=time.time() - start_time,
        )

    except Exception as e:
        # Ensure we restore on any exception
        try:
            restore_file(file_path, backup_path)
            cleanup_backup(backup_path)
        except Exception:
            pass
        return VerificationResult(
            success=False,
            proof_code=current_proof,
            attempts=1,
            build_output="",
            errors=[f"Exception during verification: {e}"],
            elapsed_time=time.time() - start_time,
        )


async def _regenerate_with_feedback(
    sorry: "SorryLocation",
    file_path: str,
    prev_proof: str,
    error: str,
    attempt: int,
    model: str,
    rag=None,
) -> Optional[str]:
    """
    Regenerate proof using error feedback.

    Args:
        sorry: The sorry location
        file_path: Path to the Lean file
        prev_proof: Previous proof that failed
        error: Error summary from failed build
        attempt: Current attempt number (2-4)
        model: LLM model to use
        rag: Optional RAG instance

    Returns:
        New proof code, or None if regeneration fails
    """
    try:
        # Import agent module
        from agent.llm_client import LLMClient
        from agent.prompts import build_retry_prompt, extract_proof_from_response
        from agent.context_formatter import format_context, format_error_context
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

        # Build context with error information
        context = format_context(sorry, file_content, rag_results)
        error_context = format_error_context(prev_proof, error)
        full_context = f"{context}\n\n{error_context}"

        # Build retry prompt with attempt-specific guidance
        prompt = build_retry_prompt(sorry, full_context, prev_proof, error, attempt)

        # Generate new proof
        client = LLMClient()

        # Adjust temperature based on attempt
        # Later attempts use higher temperature for diversity
        temperature = 0.3 + (attempt - 1) * 0.15  # 0.3, 0.45, 0.6, 0.75

        response = await client.generate(
            prompt,
            model=model,
            temperature=min(temperature, 0.8),
        )

        return extract_proof_from_response(response)

    except Exception:
        return None


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
