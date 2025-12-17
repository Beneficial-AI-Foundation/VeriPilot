"""
LSP-based verification using lean-lsp MCP server.

This module provides instant proof verification without running lake build.
Key principles:
1. NEVER modify original files - work on copies only
2. Use MCP lean-lsp for instant diagnostics (< 1 second)
3. Clean up intermediate attempts, keep only final result
"""

import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .mcp_client import LeanMCPClient, get_mcp_client, DiagnosticItem

if TYPE_CHECKING:
    from parser import SorryLocation

logger = logging.getLogger(__name__)


@dataclass
class LSPCheckResult:
    """Result of checking a file via LSP."""

    success: bool
    errors: list[str]
    file_path: str
    has_remaining_goals: bool
    goals: Optional[str] = None
    elapsed_time: float = 0.0


class LeanLSPVerifier:
    """
    Verifies Lean proofs via LSP without running lake build.

    This verifier:
    - Works on COPY files only (original never touched)
    - Gets instant feedback via MCP lean-lsp
    - Provides detailed error messages for retry feedback
    """

    def __init__(self, project_root: str, mcp_client: Optional[LeanMCPClient] = None):
        """
        Initialize the LSP verifier.

        Args:
            project_root: Root directory of the Lean project
            mcp_client: Optional MCP client (uses singleton if not provided)
        """
        self.project_root = project_root
        self._client = mcp_client or get_mcp_client()
        self._connected = False

    async def __aenter__(self):
        """Async context manager entry - connect to MCP server."""
        self._connection = self._client.connect()
        await self._connection.__aenter__()
        self._connected = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - disconnect from MCP server."""
        self._connected = False
        await self._connection.__aexit__(exc_type, exc_val, exc_tb)

    async def check_file(
        self,
        file_path: str,
        sorry_line: Optional[int] = None,
        sorry_column: Optional[int] = None,
    ) -> LSPCheckResult:
        """
        Check a Lean file for errors via LSP.

        Args:
            file_path: Absolute path to the Lean file to check
            sorry_line: Line of the sorry being replaced (for goal checking)
            sorry_column: Column of the sorry being replaced

        Returns:
            LSPCheckResult with diagnostics and goal state
        """
        if not self._connected:
            raise RuntimeError(
                "Not connected to MCP server. Use 'async with verifier:'"
            )

        start_time = time.time()

        # Get diagnostics
        diagnostics = await self._client.get_diagnostics(file_path)

        # Filter to errors only
        errors = [d for d in diagnostics if d.severity == "error"]
        error_messages = [self._format_diagnostic(d) for d in errors]

        # Check goal state if sorry location provided
        has_remaining_goals = True
        goals = None

        if sorry_line and not errors:
            # Check if proof is complete
            goal_state = await self._client.get_goal(
                file_path, sorry_line, sorry_column or 1
            )
            if goal_state:
                has_remaining_goals = not goal_state.is_complete
                goals = goal_state.goals_after

        # Success = no errors and no remaining goals
        success = len(errors) == 0 and not has_remaining_goals

        return LSPCheckResult(
            success=success,
            errors=error_messages,
            file_path=file_path,
            has_remaining_goals=has_remaining_goals,
            goals=goals,
            elapsed_time=time.time() - start_time,
        )

    def _format_diagnostic(self, d: DiagnosticItem) -> str:
        """Format a diagnostic for display/feedback."""
        location = f"line {d.line}"
        if d.column:
            location += f", col {d.column}"
        return f"[{d.severity}] {location}: {d.message}"


def create_verification_copy(
    original_path: str,
    attempt: int,
    proof_code: str,
    sorry_line: int,
    sorry_column: int = 1,
) -> str:
    """
    Create a verification copy with the proof inserted.

    This function:
    1. Copies the original file to {base}_VP{N}.lean
    2. Replaces the sorry at the given location with the proof
    3. Returns the path to the copy

    The original file is NEVER modified.

    Args:
        original_path: Path to the original Lean file
        attempt: Attempt number (1, 2, 3, ...)
        proof_code: Proof code to insert
        sorry_line: Line number of sorry (1-indexed)
        sorry_column: Column number of sorry (1-indexed)

    Returns:
        Path to the created copy file
    """
    original = Path(original_path)
    base_name = original.stem

    # Remove any existing _VP suffix to get true base name
    if "_VP" in base_name:
        base_name = base_name.split("_VP")[0]

    copy_name = f"{base_name}_VP{attempt}.lean"
    copy_path = original.parent / copy_name

    # Read original content
    with open(original_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find and replace sorry
    if 0 < sorry_line <= len(lines):
        line_content = lines[sorry_line - 1]

        # Find sorry in the line
        sorry_start = line_content.find("sorry")
        if sorry_start >= 0:
            # Replace sorry with proof
            # Handle multi-line proofs by using 'by' block format
            if "\n" in proof_code:
                # Multi-line: wrap in 'by' if not already
                if not proof_code.strip().startswith("by"):
                    proof_code = f"by\n{proof_code}"
                indent = " " * sorry_start
                proof_lines = proof_code.split("\n")
                formatted_proof = proof_lines[0]
                for pl in proof_lines[1:]:
                    formatted_proof += f"\n{indent}  {pl}"
                proof_code = formatted_proof

            # Replace sorry with proof
            new_line = line_content[:sorry_start] + proof_code + line_content[sorry_start + 5:]
            lines[sorry_line - 1] = new_line

    # Write copy
    with open(copy_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    logger.debug(f"Created verification copy: {copy_path}")
    return str(copy_path)


def cleanup_verification_copies(
    original_path: str,
    keep_attempt: Optional[int] = None,
) -> None:
    """
    Clean up verification copies, optionally keeping one.

    Args:
        original_path: Path to the original Lean file
        keep_attempt: Attempt number to keep (None = delete all)
    """
    original = Path(original_path)
    base_name = original.stem

    # Remove any existing _VP suffix to get true base name
    if "_VP" in base_name:
        base_name = base_name.split("_VP")[0]

    # Find all VP copies
    pattern = f"{base_name}_VP*.lean"
    for copy_file in original.parent.glob(pattern):
        if keep_attempt:
            keep_name = f"{base_name}_VP{keep_attempt}.lean"
            if copy_file.name == keep_name:
                continue
        try:
            copy_file.unlink()
            logger.debug(f"Deleted verification copy: {copy_file}")
        except OSError:
            pass


async def verify_proof_lsp(
    file_path: str,
    proof_code: str,
    sorry_line: int,
    sorry_column: int = 1,
    project_root: Optional[str] = None,
) -> LSPCheckResult:
    """
    Verify a proof using LSP (convenience function).

    This is a standalone function for quick verification.
    For multiple verifications, use LeanLSPVerifier as context manager.

    Args:
        file_path: Path to the original Lean file
        proof_code: Proof code to verify
        sorry_line: Line number of sorry
        sorry_column: Column number of sorry
        project_root: Lean project root (inferred if not provided)

    Returns:
        LSPCheckResult
    """
    if project_root is None:
        # Try to infer from file path
        project_root = str(Path(file_path).parent.parent.parent.parent)

    # Create copy
    copy_path = create_verification_copy(
        file_path, 1, proof_code, sorry_line, sorry_column
    )

    try:
        verifier = LeanLSPVerifier(project_root)
        async with verifier:
            result = await verifier.check_file(copy_path, sorry_line, sorry_column)
            return result
    finally:
        # Clean up
        try:
            Path(copy_path).unlink()
        except OSError:
            pass
