"""Find sorry placeholders in Lean files with their context."""

import re
from pathlib import Path

from . import SorryLocation


def find_sorries(
    file_path: str, line_range: tuple[int, int] | None = None
) -> list[SorryLocation]:
    """
    Find all sorry placeholders in a Lean file.

    Args:
        file_path: Path to the Lean file
        line_range: Optional (start, end) line range to search (1-indexed, inclusive)

    Returns:
        List of SorryLocation objects with context
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    content = path.read_text()
    lines = content.splitlines()

    imports = _extract_imports(lines)
    namespace = _extract_namespace(lines)
    sorries = []

    # Find all sorry occurrences
    sorry_pattern = re.compile(r"\bsorry\b")

    for line_num, line in enumerate(lines, start=1):
        # Skip if outside line range
        if line_range:
            start, end = line_range
            if line_num < start or line_num > end:
                continue

        for match in sorry_pattern.finditer(line):
            column = match.start() + 1  # 1-indexed

            # Find the enclosing theorem/lemma/def
            theorem_name, theorem_sig, proof_prefix = _find_theorem_context(
                lines, line_num - 1  # 0-indexed for internal use
            )

            sorries.append(
                SorryLocation(
                    file_path=str(path.absolute()),
                    line=line_num,
                    column=column,
                    theorem_name=theorem_name,
                    theorem_signature=theorem_sig,
                    proof_prefix=proof_prefix,
                    namespace=namespace,
                    imports=imports,
                )
            )

    return sorries


def _extract_imports(lines: list[str]) -> list[str]:
    """Extract import statements from the file."""
    imports = []
    import_pattern = re.compile(r"^\s*import\s+(.+)$")

    for line in lines:
        match = import_pattern.match(line)
        if match:
            imports.append(match.group(1).strip())

    return imports


def _extract_namespace(lines: list[str]) -> str:
    """Extract the active namespace (last declared)."""
    namespace_pattern = re.compile(r"^\s*namespace\s+(\S+)")
    namespace = ""

    for line in lines:
        match = namespace_pattern.match(line)
        if match:
            namespace = match.group(1)

    return namespace


def _find_theorem_context(
    lines: list[str], sorry_line_idx: int
) -> tuple[str, str, str]:
    """
    Find the enclosing theorem and extract context.

    Args:
        lines: All lines in the file
        sorry_line_idx: 0-indexed line number of the sorry

    Returns:
        (theorem_name, theorem_signature, proof_prefix)
    """
    # Pattern to match theorem/lemma/def declarations
    decl_pattern = re.compile(
        r"^\s*(?:@\[.*?\]\s*)*"  # optional attributes
        r"(theorem|lemma|def)\s+"  # declaration keyword
        r"(\w+)"  # name
        r"(.*?)$",  # rest of line (part of signature)
        re.DOTALL,
    )

    theorem_name = ""
    theorem_sig_lines = []
    decl_start_idx = -1

    # Search backwards for the declaration
    for i in range(sorry_line_idx, -1, -1):
        line = lines[i]
        match = decl_pattern.match(line)
        if match:
            theorem_name = match.group(2)
            decl_start_idx = i
            break

    if not theorem_name:
        return "", "", ""

    # Extract full signature (from declaration to := or :=by)
    sig_end_idx = decl_start_idx
    by_pattern = re.compile(r":=\s*by\b")

    for i in range(decl_start_idx, min(sorry_line_idx + 1, len(lines))):
        line = lines[i]
        if by_pattern.search(line):
            sig_end_idx = i
            break
        # Also check for just `:=` without `by` (term-mode)
        if ":=" in line and "by" not in line:
            sig_end_idx = i
            break

    # Build signature
    for i in range(decl_start_idx, sig_end_idx + 1):
        theorem_sig_lines.append(lines[i])

    theorem_signature = "\n".join(theorem_sig_lines)

    # Extract proof prefix (tactics between := by and sorry)
    proof_lines = []
    in_proof = False

    for i in range(decl_start_idx, sorry_line_idx + 1):
        line = lines[i]

        if by_pattern.search(line):
            in_proof = True
            # Get the part after `:= by`
            match = by_pattern.search(line)
            if match:
                after_by = line[match.end() :].strip()
                if after_by and "sorry" not in after_by:
                    proof_lines.append(after_by)
            continue

        if in_proof and i < sorry_line_idx:
            # Don't include the sorry line itself
            stripped = line.strip()
            if stripped and "sorry" not in stripped:
                proof_lines.append(stripped)

    proof_prefix = "\n".join(proof_lines)

    return theorem_name, theorem_signature, proof_prefix
