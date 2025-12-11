"""
File modification utilities for VeriPilot.

Handles replacing sorry placeholders with generated proofs,
with backup/restore support for safe rollback.

Includes attempt-numbered file copies and cumulative logging
for debugging and observability.
"""

import json
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from parser import SorryLocation


@dataclass
class AttemptLog:
    """Log entry for a single proof attempt."""

    attempt: int
    proof_code: str
    build_success: bool
    errors: list[str]
    timestamp: str
    elapsed_time: float
    model_used: str = ""
    temperature: float = 0.0

    @classmethod
    def create(
        cls,
        attempt: int,
        proof_code: str,
        build_success: bool,
        errors: list[str],
        elapsed_time: float,
        model_used: str = "",
        temperature: float = 0.0,
    ) -> "AttemptLog":
        """Create a new AttemptLog with current timestamp."""
        return cls(
            attempt=attempt,
            proof_code=proof_code,
            build_success=build_success,
            errors=errors,
            timestamp=datetime.now().isoformat(),
            elapsed_time=elapsed_time,
            model_used=model_used,
            temperature=temperature,
        )


def backup_file(file_path: str) -> str:
    """
    Create a backup of the file before modification.

    Args:
        file_path: Path to the file to backup

    Returns:
        Path to the backup file (.bak extension)
    """
    backup_path = f"{file_path}.bak"
    shutil.copy2(file_path, backup_path)
    return backup_path


def restore_file(file_path: str, backup_path: str) -> None:
    """
    Restore a file from its backup.

    Args:
        file_path: Path to the file to restore
        backup_path: Path to the backup file
    """
    shutil.copy2(backup_path, file_path)


def cleanup_backup(backup_path: str) -> None:
    """
    Remove a backup file after successful verification.

    Args:
        backup_path: Path to the backup file to remove
    """
    path = Path(backup_path)
    if path.exists():
        path.unlink()


def format_proof_block(proof: str, indentation: int) -> str:
    """
    Format a proof with proper indentation.

    Args:
        proof: The proof code (possibly multi-line)
        indentation: Number of spaces for indentation

    Returns:
        Formatted proof with each line properly indented
    """
    indent_str = " " * indentation
    lines = proof.strip().split("\n")

    # First line replaces 'sorry' inline, subsequent lines get full indent
    if len(lines) == 1:
        return lines[0].strip()

    # Multi-line: first line inline, rest indented
    formatted_lines = [lines[0].strip()]
    for line in lines[1:]:
        stripped = line.strip()
        if stripped:
            formatted_lines.append(indent_str + stripped)
        else:
            formatted_lines.append("")

    return "\n".join(formatted_lines)


def replace_sorry(
    file_path: str,
    sorry: "SorryLocation",
    proof: str,
) -> bool:
    """
    Replace a sorry placeholder with generated proof code.

    Args:
        file_path: Path to the Lean file
        sorry: SorryLocation containing line/column info
        proof: The proof code to insert

    Returns:
        True if replacement succeeded, False otherwise
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Sorry line is 1-indexed
        sorry_line_idx = sorry.line - 1

        if sorry_line_idx < 0 or sorry_line_idx >= len(lines):
            return False

        line = lines[sorry_line_idx]

        # Verify this line contains 'sorry'
        if "sorry" not in line:
            return False

        # Extract indentation from current line
        indentation = len(line) - len(line.lstrip())

        # Format proof with proper indentation
        formatted_proof = format_proof_block(proof, indentation)

        # Replace 'sorry' with formatted proof
        # Handle case where sorry might appear multiple times (take first)
        new_line = line.replace("sorry", formatted_proof, 1)
        lines[sorry_line_idx] = new_line

        # Write back
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return True

    except (OSError, IOError):
        return False


def replace_sorry_at_position(
    file_path: str,
    line: int,
    column: int,
    proof: str,
) -> bool:
    """
    Replace sorry at specific position (more precise than replace_sorry).

    Args:
        file_path: Path to the Lean file
        line: Line number (1-indexed)
        column: Column number (1-indexed)
        proof: The proof code to insert

    Returns:
        True if replacement succeeded, False otherwise
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        line_idx = line - 1
        col_idx = column - 1

        if line_idx < 0 or line_idx >= len(lines):
            return False

        current_line = lines[line_idx]

        # Check if 'sorry' starts at this column
        if not current_line[col_idx:].startswith("sorry"):
            # Fallback: find sorry anywhere in line
            sorry_pos = current_line.find("sorry")
            if sorry_pos == -1:
                return False
            col_idx = sorry_pos

        # Extract indentation
        indentation = len(current_line) - len(current_line.lstrip())
        formatted_proof = format_proof_block(proof, indentation)

        # Replace at exact position
        new_line = (
            current_line[:col_idx]
            + formatted_proof
            + current_line[col_idx + len("sorry") :]
        )
        lines[line_idx] = new_line

        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return True

    except (OSError, IOError):
        return False


def get_file_content(file_path: str) -> str:
    """
    Read file content.

    Args:
        file_path: Path to the file

    Returns:
        File content as string
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def file_contains_sorry(file_path: str) -> bool:
    """
    Check if file still contains any sorry placeholders.

    Args:
        file_path: Path to the Lean file

    Returns:
        True if file contains 'sorry', False otherwise
    """
    content = get_file_content(file_path)
    return "sorry" in content


def create_vp_copy(original_path: str, prefix: str = "VP_") -> str:
    """
    Create a VeriPilot working copy of a file.

    Args:
        original_path: Path to original .lean file
        prefix: Prefix for copy (default: "VP_")

    Returns:
        Path to VP_<filename>.lean copy
    """
    original = Path(original_path)
    vp_name = f"{prefix}{original.name}"
    vp_path = original.parent / vp_name

    shutil.copy2(original, vp_path)
    return str(vp_path)


def cleanup_vp_files(vp_path: str):
    """
    Clean up VP_ working copy and any backup files.

    Args:
        vp_path: Path to VP_ file
    """
    vp_file = Path(vp_path)
    if vp_file.exists():
        vp_file.unlink()

    # Also remove backup
    bak_file = Path(str(vp_path) + ".bak")
    if bak_file.exists():
        bak_file.unlink()


# ============================================================================
# Attempt-numbered file copies and cumulative logging
# ============================================================================


def create_attempt_copy(original_path: str, attempt: int) -> str:
    """
    Create VP_N_originalfile.lean for attempt N.

    Args:
        original_path: Path to original .lean file
        attempt: Attempt number (1, 2, 3, ...)

    Returns:
        Path to VP_N_<filename>.lean
    """
    original = Path(original_path)
    vp_name = f"VP_{attempt}_{original.name}"
    vp_path = original.parent / vp_name

    shutil.copy2(original, vp_path)
    return str(vp_path)


def cleanup_intermediate_attempts(original_path: str, final_attempt: int) -> list[str]:
    """
    Delete VP_1 through VP_{N-1}, keeping only VP_N.

    Args:
        original_path: Original file path
        final_attempt: The final attempt number to keep

    Returns:
        List of deleted file paths
    """
    original = Path(original_path)
    deleted = []

    for i in range(1, final_attempt):
        attempt_path = original.parent / f"VP_{i}_{original.name}"
        if attempt_path.exists():
            attempt_path.unlink()
            deleted.append(str(attempt_path))

    return deleted


def cleanup_all_attempt_files(original_path: str, max_attempts: int = 10) -> list[str]:
    """
    Delete all VP_N_ files for a given original file.

    Args:
        original_path: Original file path
        max_attempts: Maximum attempt number to check

    Returns:
        List of deleted file paths
    """
    original = Path(original_path)
    deleted = []

    for i in range(1, max_attempts + 1):
        attempt_path = original.parent / f"VP_{i}_{original.name}"
        if attempt_path.exists():
            attempt_path.unlink()
            deleted.append(str(attempt_path))

    # Also clean up basic VP_ file
    basic_vp = original.parent / f"VP_{original.name}"
    if basic_vp.exists():
        basic_vp.unlink()
        deleted.append(str(basic_vp))

    return deleted


def write_attempt_log(
    original_path: str,
    logs: list[AttemptLog],
    format: str = "json",
) -> str:
    """
    Write cumulative log to VP_log_originalfile.{format}.

    Args:
        original_path: Original file path (used for naming)
        logs: List of AttemptLog entries
        format: Output format - "json" (default), "md", "txt"

    Returns:
        Path to created log file
    """
    original = Path(original_path)
    log_name = f"VP_log_{original.stem}.{format}"
    log_path = original.parent / log_name

    if format == "json":
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump([asdict(log) for log in logs], f, indent=2)

    elif format == "md":
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"# Verification Log: {original.name}\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(f"Total Attempts: {len(logs)}\n\n")
            f.write("---\n\n")

            for log in logs:
                status = "SUCCESS" if log.build_success else "FAILED"
                f.write(f"## Attempt {log.attempt} [{status}]\n\n")
                f.write(f"- **Timestamp:** {log.timestamp}\n")
                f.write(f"- **Elapsed:** {log.elapsed_time:.2f}s\n")
                if log.model_used:
                    f.write(f"- **Model:** {log.model_used}\n")
                if log.temperature > 0:
                    f.write(f"- **Temperature:** {log.temperature}\n")
                f.write("\n**Proof:**\n```lean\n")
                f.write(log.proof_code)
                f.write("\n```\n\n")
                if log.errors:
                    f.write("**Errors:**\n")
                    for error in log.errors:
                        f.write(f"- {error}\n")
                f.write("\n---\n\n")

    elif format == "txt":
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Verification Log: {original.name}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Total Attempts: {len(logs)}\n")
            f.write("=" * 60 + "\n\n")

            for log in logs:
                status = "SUCCESS" if log.build_success else "FAILED"
                f.write(f"Attempt {log.attempt} [{status}]\n")
                f.write("-" * 40 + "\n")
                f.write(f"Time: {log.timestamp}\n")
                f.write(f"Elapsed: {log.elapsed_time:.2f}s\n")
                if log.model_used:
                    f.write(f"Model: {log.model_used}\n")
                f.write(f"Proof:\n{log.proof_code}\n")
                if log.errors:
                    f.write("Errors:\n")
                    for error in log.errors:
                        f.write(f"  - {error}\n")
                f.write("\n")

    else:
        raise ValueError(f"Unsupported log format: {format}")

    return str(log_path)


def read_attempt_log(log_path: str) -> list[AttemptLog]:
    """
    Read attempt logs from a JSON log file.

    Args:
        log_path: Path to the log file

    Returns:
        List of AttemptLog entries
    """
    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [AttemptLog(**entry) for entry in data]


def cleanup_log_file(original_path: str, format: str = "json") -> bool:
    """
    Remove the log file for a given original file.

    Args:
        original_path: Original file path
        format: Log format extension

    Returns:
        True if file was deleted, False if not found
    """
    original = Path(original_path)
    log_name = f"VP_log_{original.stem}.{format}"
    log_path = original.parent / log_name

    if log_path.exists():
        log_path.unlink()
        return True
    return False

