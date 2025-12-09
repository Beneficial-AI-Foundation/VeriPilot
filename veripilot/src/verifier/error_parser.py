"""
Lean error parser for VeriPilot.

Parses compiler errors and warnings from lake build output
to provide structured feedback for retry prompts.
"""

import re
from typing import Optional

from . import LeanError


# Regex patterns for Lean error messages
# Pattern: /path/to/file.lean:line:col: error|warning: message
ERROR_PATTERN = re.compile(
    r"([^\s:]+\.lean):(\d+):(\d+):\s*(error|warning):\s*(.+?)(?=\n\S|\Z)",
    re.DOTALL,
)

# Alternative pattern for errors without column
ERROR_PATTERN_NO_COL = re.compile(
    r"([^\s:]+\.lean):(\d+):\s*(error|warning):\s*(.+?)(?=\n\S|\Z)",
    re.DOTALL,
)

# Pattern for sorry warnings
SORRY_PATTERN = re.compile(
    r"warning:.*declaration uses 'sorry'",
    re.IGNORECASE,
)

# Pattern for tactic failures
TACTIC_FAILED_PATTERN = re.compile(
    r"tactic '(\w+)' failed",
    re.IGNORECASE,
)

# Pattern for type mismatches
TYPE_MISMATCH_PATTERN = re.compile(
    r"type mismatch",
    re.IGNORECASE,
)

# Pattern for unknown identifier
UNKNOWN_ID_PATTERN = re.compile(
    r"unknown identifier '([^']+)'",
    re.IGNORECASE,
)


def classify_error(message: str) -> str:
    """
    Classify error type from message content.

    Args:
        message: Error message text

    Returns:
        Error type: "type", "tactic", "sorry", "identifier", or "unknown"
    """
    message_lower = message.lower()

    if "type mismatch" in message_lower:
        return "type"
    elif "tactic" in message_lower and "failed" in message_lower:
        return "tactic"
    elif "sorry" in message_lower:
        return "sorry"
    elif "unknown identifier" in message_lower:
        return "identifier"
    elif "expected" in message_lower and "got" in message_lower:
        return "type"
    elif "unsolved goals" in message_lower:
        return "tactic"
    else:
        return "unknown"


def parse_lean_errors(build_output: str) -> list[LeanError]:
    """
    Parse Lean compiler errors from build output.

    Args:
        build_output: Combined stdout/stderr from lake build

    Returns:
        List of parsed LeanError objects
    """
    errors = []

    # Try primary pattern (with column)
    for match in ERROR_PATTERN.finditer(build_output):
        file_path = match.group(1)
        line = int(match.group(2))
        column = int(match.group(3))
        severity = match.group(4)
        message = match.group(5).strip()

        # Skip warnings unless they're sorry-related
        if severity == "warning" and "sorry" not in message.lower():
            continue

        error_type = classify_error(message)

        errors.append(
            LeanError(
                file_path=file_path,
                line=line,
                column=column,
                error_type=error_type,
                message=message,
            )
        )

    # Try secondary pattern if no errors found
    if not errors:
        for match in ERROR_PATTERN_NO_COL.finditer(build_output):
            file_path = match.group(1)
            line = int(match.group(2))
            severity = match.group(3)
            message = match.group(4).strip()

            if severity == "warning" and "sorry" not in message.lower():
                continue

            error_type = classify_error(message)

            errors.append(
                LeanError(
                    file_path=file_path,
                    line=line,
                    column=1,  # Default column
                    error_type=error_type,
                    message=message,
                )
            )

    return errors


def filter_errors_for_file(
    errors: list[LeanError],
    file_path: str,
) -> list[LeanError]:
    """
    Filter errors to only those from a specific file.

    Args:
        errors: List of LeanError objects
        file_path: Path to filter by (matches end of path)

    Returns:
        Filtered list of errors
    """
    # Normalize path for comparison
    file_path = file_path.replace("\\", "/")
    file_name = file_path.split("/")[-1]

    filtered = []
    for error in errors:
        error_path = error.file_path.replace("\\", "/")
        # Match full path or just filename
        if error_path.endswith(file_path) or error_path.endswith(file_name):
            filtered.append(error)

    return filtered


def filter_errors_for_line(
    errors: list[LeanError],
    line: int,
    tolerance: int = 5,
) -> list[LeanError]:
    """
    Filter errors to those near a specific line.

    Args:
        errors: List of LeanError objects
        line: Target line number
        tolerance: Lines above/below to include

    Returns:
        Filtered list of errors
    """
    return [e for e in errors if abs(e.line - line) <= tolerance]


def extract_error_summary(errors: list[LeanError]) -> str:
    """
    Create a concise error summary for retry prompts.

    Args:
        errors: List of LeanError objects

    Returns:
        Human-readable error summary
    """
    if not errors:
        return "No specific errors found in build output."

    # Group by error type
    by_type: dict[str, list[LeanError]] = {}
    for error in errors:
        by_type.setdefault(error.error_type, []).append(error)

    summary_parts = []

    # Prioritize type errors and tactic failures
    priority_order = ["type", "tactic", "identifier", "sorry", "unknown"]

    for error_type in priority_order:
        if error_type not in by_type:
            continue

        type_errors = by_type[error_type]

        if error_type == "type":
            summary_parts.append(f"Type errors ({len(type_errors)}):")
            for e in type_errors[:3]:  # Show top 3
                summary_parts.append(f"  - Line {e.line}: {_truncate(e.message, 100)}")

        elif error_type == "tactic":
            summary_parts.append(f"Tactic failures ({len(type_errors)}):")
            for e in type_errors[:3]:
                summary_parts.append(f"  - Line {e.line}: {_truncate(e.message, 100)}")

        elif error_type == "identifier":
            summary_parts.append(f"Unknown identifiers ({len(type_errors)}):")
            for e in type_errors[:3]:
                # Extract identifier name
                match = UNKNOWN_ID_PATTERN.search(e.message)
                if match:
                    summary_parts.append(f"  - '{match.group(1)}' at line {e.line}")
                else:
                    summary_parts.append(f"  - Line {e.line}: {_truncate(e.message, 80)}")

        elif error_type == "sorry":
            summary_parts.append(f"Sorry warnings ({len(type_errors)}): Proof incomplete")

        else:
            if len(type_errors) > 0:
                summary_parts.append(f"Other errors ({len(type_errors)}):")
                for e in type_errors[:2]:
                    summary_parts.append(f"  - Line {e.line}: {_truncate(e.message, 80)}")

    return "\n".join(summary_parts)


def extract_primary_error(errors: list[LeanError]) -> Optional[LeanError]:
    """
    Extract the most important error for debugging.

    Args:
        errors: List of LeanError objects

    Returns:
        Most relevant error, or None if no errors
    """
    if not errors:
        return None

    # Priority: type > tactic > identifier > sorry > unknown
    priority = {"type": 0, "tactic": 1, "identifier": 2, "sorry": 3, "unknown": 4}

    return min(errors, key=lambda e: (priority.get(e.error_type, 5), e.line))


def format_error_for_prompt(error: LeanError) -> str:
    """
    Format a single error for inclusion in retry prompt.

    Args:
        error: LeanError object

    Returns:
        Formatted error string
    """
    return f"Error at line {error.line}, column {error.column}:\n{error.message}"


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max length with ellipsis."""
    # Clean up whitespace
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
