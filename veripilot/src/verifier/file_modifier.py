"""
File modification utilities for VeriPilot.

Handles replacing sorry placeholders with generated proofs,
with backup/restore support for safe rollback.
"""

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from parser import SorryLocation


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

