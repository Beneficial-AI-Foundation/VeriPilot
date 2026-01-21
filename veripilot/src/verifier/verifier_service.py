"""
Verifier service for VeriPilot.

Provides a unified interface for proof verification with:
- MCP lean-lsp for instant verification (when warmed up)
- lake build as fallback (when MCP not ready)
- Copy-based workflow (never modifies original during verification)
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from . import VerificationResult, BuildResult
from .file_modifier import (
    AttemptLog,
    create_attempt_copy,
    write_attempt_log,
    cleanup_intermediate_attempts,
    replace_sorry,
    file_contains_sorry,
)
from .lake_runner import run_lake_build
from .error_parser import parse_lean_errors, extract_error_summary, filter_errors_for_file
from .error_normalizer import ErrorMessageNormalizer

if TYPE_CHECKING:
    from parser import SorryLocation
    from agent import ProofResult

logger = logging.getLogger(__name__)


@dataclass
class VerifierStatus:
    """Status of the verifier service."""

    mcp_available: bool = False
    mcp_warming_up: bool = False
    mcp_warm_up_time: float = 0.0
    lake_available: bool = True
    last_verification_method: str = ""


class VerifierService:
    """
    Unified verifier service with MCP warm-up and fallback.

    Usage:
        service = VerifierService(project_dir)
        await service.start()  # Start warm-up in background

        # Later...
        result = await service.verify_proof_on_copy(sorry, proof_result)

        await service.stop()  # Clean shutdown
    """

    def __init__(
        self,
        project_dir: str,
        timeout: int = 300,
        mcp_warmup_timeout: int = 120,
    ):
        """
        Initialize verifier service.

        Args:
            project_dir: Lean project directory (with lakefile)
            timeout: Build timeout in seconds
            mcp_warmup_timeout: Max time to wait for MCP warm-up (default 120s for large projects)
        """
        self.project_dir = project_dir
        self.timeout = timeout
        self.mcp_warmup_timeout = mcp_warmup_timeout

        self._status = VerifierStatus()
        self._mcp_client = None
        self._mcp_connection = None
        self._warmup_task: Optional[asyncio.Task] = None

    @property
    def status(self) -> VerifierStatus:
        """Get current verifier status."""
        return self._status

    async def start(self, wait_for_warmup: bool = True) -> None:
        """
        Start the verifier service.

        Args:
            wait_for_warmup: If True (default), wait for MCP to be ready.
                           If False, warmup runs in background.
        """
        # Start MCP warm-up
        self._warmup_task = asyncio.create_task(self._warmup_mcp())
        logger.info("Verifier service started, MCP warming up...")

        if wait_for_warmup:
            # Wait for warmup to complete before returning
            await self._warmup_task
            if self._status.mcp_available:
                logger.info("MCP ready for instant verification")
            else:
                logger.warning("MCP warmup failed - falling back to lake build")

    async def stop(self) -> None:
        """Stop the verifier service and clean up."""
        # Cancel warm-up if still running
        if self._warmup_task and not self._warmup_task.done():
            self._warmup_task.cancel()
            try:
                await self._warmup_task
            except asyncio.CancelledError:
                pass

        # Close MCP connection
        if self._mcp_connection:
            try:
                await self._mcp_connection.__aexit__(None, None, None)
            except Exception:
                pass
            self._mcp_connection = None
            self._mcp_client = None

        self._status.mcp_available = False
        logger.info("Verifier service stopped")

    async def _warmup_mcp(self) -> None:
        """
        Warm up MCP connection in background.

        Based on LeanDojo's warmup strategy:
        - First request takes ~350ms (cold start)
        - Subsequent requests take ~10ms (warm)
        - Key: Keep the connection alive and reuse it

        The lean-lsp-mcp server initializes the Lean LSP on first request.
        We pay this cost once, then all verifications are fast.
        """
        self._status.mcp_warming_up = True
        start_time = time.time()

        try:
            from .mcp_client import LeanMCPClient

            self._mcp_client = LeanMCPClient()
            self._mcp_connection = self._mcp_client.connect()

            logger.info("Starting MCP connection (this may take a moment on first use)...")

            # Start connection - this spawns the lean-lsp-mcp server
            await asyncio.wait_for(
                self._mcp_connection.__aenter__(),
                timeout=self.mcp_warmup_timeout,
            )

            # Warmup query: Get diagnostics on the main project file
            # This forces the Lean LSP to initialize and load project dependencies
            # LeanDojo shows this takes ~350ms on first call, ~10ms after
            test_file = Path(self.project_dir) / "Curve25519Dalek.lean"
            if test_file.exists():
                logger.info(f"Warming up Lean LSP with {test_file.name}...")
                await asyncio.wait_for(
                    self._mcp_client.get_diagnostics(str(test_file)),
                    timeout=self.mcp_warmup_timeout,
                )

            self._status.mcp_available = True
            self._status.mcp_warm_up_time = time.time() - start_time
            logger.info(f"MCP warmed up in {self._status.mcp_warm_up_time:.1f}s - subsequent verifications will be instant")

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.warning(f"MCP warm-up timed out after {elapsed:.1f}s (limit: {self.mcp_warmup_timeout}s)")
            logger.warning("Falling back to lake build for verification (slower but reliable)")
            self._status.mcp_available = False
        except Exception as e:
            logger.warning(f"MCP warm-up failed: {e}")
            logger.warning("Falling back to lake build for verification")
            self._status.mcp_available = False
        finally:
            self._status.mcp_warming_up = False

    async def verify_proof_on_copy(
        self,
        sorry: "SorryLocation",
        proof_code: str,
        attempt: int = 1,
        model_used: str = "",
        temperature: float = 0.0,
    ) -> tuple[bool, list[str], str]:
        """
        Verify a proof by creating a copy and checking it.

        This method:
        1. Creates a copy file (_VP{N}.lean)
        2. Replaces sorry with proof in the copy
        3. Verifies using MCP (instant) or lake build (fallback)
        4. Returns result - ORIGINAL FILE IS NEVER MODIFIED

        Args:
            sorry: The sorry location from the original file
            proof_code: Proof code to verify
            attempt: Attempt number for file naming
            model_used: Model name for logging
            temperature: Temperature used for logging

        Returns:
            Tuple of (success, errors, copy_file_path)
        """
        file_path = str(sorry.file_path)

        # Create copy file
        copy_path = create_attempt_copy(file_path, attempt)
        logger.debug(f"Created verification copy: {copy_path}")

        # Replace sorry in copy
        success = replace_sorry(copy_path, sorry, proof_code)
        if not success:
            return False, ["Failed to replace sorry in copy file"], copy_path

        # Verify using best available method
        if self._status.mcp_available and self._mcp_client:
            return await self._verify_with_mcp(copy_path, sorry.line)
        else:
            return await self._verify_with_lake(copy_path)

    async def _verify_with_mcp(
        self,
        file_path: str,
        sorry_line: int,
    ) -> tuple[bool, list[str], str]:
        """Verify using MCP lean-lsp (instant)."""
        self._status.last_verification_method = "mcp"

        try:
            # Get diagnostics
            diagnostics = await self._mcp_client.get_diagnostics(file_path)
            errors = [d for d in diagnostics if d.severity == "error"]

            if errors:
                error_msgs = [f"[{d.severity}] line {d.line}: {d.message}" for d in errors]
                return False, error_msgs, file_path

            # Check if proof is complete (no remaining goals)
            goal_state = await self._mcp_client.get_goal(file_path, sorry_line, 5)
            if goal_state and not goal_state.is_complete:
                return False, [f"Remaining goals: {goal_state.goals_after}"], file_path

            # Check file no longer contains sorry
            if file_contains_sorry(file_path):
                return False, ["File still contains sorry"], file_path

            return True, [], file_path

        except Exception as e:
            logger.warning(f"MCP verification failed, falling back to lake: {e}")
            return await self._verify_with_lake(file_path)

    async def _verify_with_lake(
        self,
        file_path: str,
    ) -> tuple[bool, list[str], str]:
        """Verify using lake build (slower but reliable)."""
        self._status.last_verification_method = "lake"

        # For lake build to work on the copy, we need to temporarily
        # make it part of the build. This is tricky because lake builds
        # the module tree, not individual files.
        #
        # Workaround: We actually need to modify the original for lake build.
        # But we'll be careful to track the copy for logging purposes.

        # For now, return an error indicating lake build on copies isn't supported
        # The user should use the regular verify_proof() function
        return (
            False,
            ["Lake build requires working on original file. Use verify_proof() instead."],
            file_path,
        )


# Convenience function for one-off verification
async def verify_proof_instant(
    sorry: "SorryLocation",
    proof_code: str,
    project_dir: str,
    timeout: int = 60,
) -> tuple[bool, list[str]]:
    """
    Verify a proof instantly using MCP (if available).

    This is a convenience function for quick verification.
    For multiple verifications, use VerifierService.

    Args:
        sorry: The sorry location
        proof_code: Proof code to verify
        project_dir: Lean project directory
        timeout: MCP timeout in seconds

    Returns:
        Tuple of (success, errors)
    """
    from .mcp_client import LeanMCPClient
    from .lsp_verifier import create_verification_copy

    # Create copy
    copy_path = create_verification_copy(
        str(sorry.file_path),
        1,
        proof_code,
        sorry.line,
        sorry.column,
    )

    try:
        client = LeanMCPClient()
        async with client.connect():
            # Get diagnostics
            diagnostics = await asyncio.wait_for(
                client.get_diagnostics(copy_path),
                timeout=timeout,
            )
            errors = [d for d in diagnostics if d.severity == "error"]

            if errors:
                return False, [f"[{d.severity}] line {d.line}: {d.message}" for d in errors]

            # Check goal state
            goal = await asyncio.wait_for(
                client.get_goal(copy_path, sorry.line, 5),
                timeout=timeout,
            )
            if goal and not goal.is_complete:
                return False, [f"Remaining goals: {goal.goals_after}"]

            return True, []

    except asyncio.TimeoutError:
        return False, ["MCP verification timed out"]
    except Exception as e:
        return False, [f"MCP verification failed: {e}"]
    finally:
        # Clean up copy
        try:
            Path(copy_path).unlink()
        except OSError:
            pass
