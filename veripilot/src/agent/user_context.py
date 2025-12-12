"""
User context loader for VeriPilot.

Allows users to provide additional context via a markdown/text file containing:
- Custom prompt text
- Paths to additional files to include (Lean, Rust, docs)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import re


@dataclass
class UserContext:
    """Parsed user context from a context file."""
    prompt_text: str = ""
    file_paths: list[str] = field(default_factory=list)
    raw_content: str = ""


def parse_context_file(path: str) -> UserContext:
    """
    Parse a user context file (markdown or text).

    The file can contain:
    - Free text that becomes additional prompt context
    - File paths prefixed with "file:" or in a "## Files" section
    - Code blocks are preserved as-is

    Example context file:
    ```
    ## Custom Instructions
    Focus on using `grind` tactic for this proof.

    ## Files
    file: /path/to/helper.lean
    file: /path/to/spec.md

    ## Additional Notes
    The function uses a specific encoding...
    ```

    Args:
        path: Path to the context file

    Returns:
        UserContext with parsed content
    """
    context_path = Path(path)
    if not context_path.exists():
        raise FileNotFoundError(f"Context file not found: {path}")

    content = context_path.read_text(encoding="utf-8")

    # Extract file paths
    file_paths = []

    # Pattern 1: "file: /path/to/file"
    file_pattern = re.compile(r"^file:\s*(.+)$", re.MULTILINE)
    for match in file_pattern.finditer(content):
        file_path = match.group(1).strip()
        if file_path:
            file_paths.append(file_path)

    # Pattern 2: Lines in a "## Files" section that look like paths
    files_section = re.search(
        r"##\s*Files?\s*\n((?:.*\n)*?)(?=\n##|\Z)",
        content,
        re.IGNORECASE
    )
    if files_section:
        for line in files_section.group(1).split("\n"):
            line = line.strip()
            # Skip empty lines and file: prefixed (already handled)
            if not line or line.startswith("file:"):
                continue
            # Check if it looks like a path
            if line.startswith("/") or line.startswith("./") or line.startswith("~"):
                file_paths.append(line)
            # Also handle list items: "- /path/to/file"
            elif line.startswith("- ") or line.startswith("* "):
                potential_path = line[2:].strip()
                if potential_path.startswith(("/", "./", "~")):
                    file_paths.append(potential_path)

    # Remove file: lines and ## Files section from content for prompt text
    prompt_text = file_pattern.sub("", content)
    prompt_text = re.sub(
        r"##\s*Files?\s*\n(?:.*\n)*?(?=\n##|\Z)",
        "",
        prompt_text,
        flags=re.IGNORECASE
    )
    prompt_text = prompt_text.strip()

    return UserContext(
        prompt_text=prompt_text,
        file_paths=file_paths,
        raw_content=content,
    )


def load_additional_files(
    paths: list[str],
    max_lines_per_file: int = 200,
    max_total_lines: int = 1000,
) -> str:
    """
    Load content from additional user-specified files.

    Args:
        paths: List of file paths to load
        max_lines_per_file: Max lines to include per file
        max_total_lines: Max total lines across all files

    Returns:
        Formatted content from all files
    """
    if not paths:
        return ""

    sections = []
    total_lines = 0

    for path_str in paths:
        if total_lines >= max_total_lines:
            sections.append("-- Skipped remaining files (token budget)")
            break

        path = Path(path_str).expanduser()
        if not path.exists():
            sections.append(f"-- File not found: {path_str}")
            continue

        try:
            content = path.read_text(encoding="utf-8")
            lines = content.split("\n")
            line_count = len(lines)

            # Truncate if needed
            if line_count > max_lines_per_file:
                lines = lines[:max_lines_per_file]
                lines.append(f"-- ... truncated ({line_count - max_lines_per_file} more lines)")

            if total_lines + len(lines) > max_total_lines:
                remaining = max_total_lines - total_lines
                lines = lines[:remaining]
                lines.append("-- ... truncated (budget)")

            # Determine file type for syntax highlighting
            suffix = path.suffix.lower()
            lang = {
                ".lean": "lean",
                ".rs": "rust",
                ".py": "python",
                ".md": "markdown",
                ".txt": "",
            }.get(suffix, "")

            sections.append(f"### {path.name}")
            sections.append(f"```{lang}")
            sections.append("\n".join(lines))
            sections.append("```")
            sections.append("")

            total_lines += len(lines)

        except Exception as e:
            sections.append(f"-- Error reading {path_str}: {e}")

    if not sections:
        return ""

    return "\n".join(sections)


def load_user_context(context_path: str) -> Optional[str]:
    """
    Load and format complete user context from a context file.

    This is the main entry point for user context loading.

    Args:
        context_path: Path to the context file

    Returns:
        Formatted context string ready for LLM prompt, or None if failed
    """
    try:
        ctx = parse_context_file(context_path)
    except FileNotFoundError:
        return None
    except Exception:
        return None

    parts = []

    # Add prompt text
    if ctx.prompt_text:
        parts.append(ctx.prompt_text)

    # Add file contents
    if ctx.file_paths:
        file_content = load_additional_files(ctx.file_paths)
        if file_content:
            parts.append("\n### Referenced Files\n")
            parts.append(file_content)

    return "\n\n".join(parts) if parts else None
