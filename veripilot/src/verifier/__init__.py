"""
Verifier module for VeriPilot MVP.

Provides proof verification via lake build with retry logic.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LeanError:
    """Parsed Lean compiler error."""

    file_path: str
    line: int
    column: int
    error_type: str  # "type", "tactic", "sorry", "unknown"
    message: str
    context: str = ""


@dataclass
class BuildResult:
    """Result of running lake build."""

    success: bool  # return_code == 0
    stdout: str
    stderr: str
    return_code: int
    elapsed_time: float


@dataclass
class VerificationResult:
    """Result of verifying a proof."""

    success: bool
    proof_code: str
    attempts: int
    build_output: str
    errors: list[str] = field(default_factory=list)
    elapsed_time: float = 0.0
    output_file: Optional[str] = None  # Path to final _VPN.lean file
    log_file: Optional[str] = None  # Path to VP_log_filename.json


# Lazy imports to avoid circular dependencies
def _get_verify_proof():
    from .retry_handler import verify_proof

    return verify_proof


def _get_run_lake_build():
    from .lake_runner import run_lake_build

    return run_lake_build


def _get_parse_lean_errors():
    from .error_parser import parse_lean_errors

    return parse_lean_errors


__all__ = [
    "VerificationResult",
    "LeanError",
    "BuildResult",
    "AttemptLog",
    "verify_proof",
    "verify_proof_lsp",
    "run_lake_build",
    "parse_lean_errors",
    "create_attempt_copy",
    "cleanup_intermediate_attempts",
    "write_attempt_log",
    "read_attempt_log",
    # LSP verification (Phase 7.3)
    "LeanMCPClient",
    "LeanLSPVerifier",
    "VerifierService",
    "verify_proof_instant",
]


# Import AttemptLog from file_modifier
from .file_modifier import (
    AttemptLog,
    create_attempt_copy,
    cleanup_intermediate_attempts,
    cleanup_all_attempt_files,
    write_attempt_log,
    read_attempt_log,
    cleanup_log_file,
)


# Re-export functions (imported lazily when accessed)
def __getattr__(name: str):
    """Lazy loading of submodule functions."""
    if name == "verify_proof":
        return _get_verify_proof()
    elif name == "verify_proof_lsp":
        from .retry_handler import verify_proof_lsp
        return verify_proof_lsp
    elif name == "run_lake_build":
        return _get_run_lake_build()
    elif name == "parse_lean_errors":
        return _get_parse_lean_errors()
    # LSP verification (Phase 7.3)
    elif name == "LeanMCPClient":
        from .mcp_client import LeanMCPClient
        return LeanMCPClient
    elif name == "LeanLSPVerifier":
        from .lsp_verifier import LeanLSPVerifier
        return LeanLSPVerifier
    elif name == "VerifierService":
        from .verifier_service import VerifierService
        return VerifierService
    elif name == "verify_proof_instant":
        from .verifier_service import verify_proof_instant
        return verify_proof_instant
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
