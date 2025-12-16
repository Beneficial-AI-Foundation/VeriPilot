"""
Context formatter for Prover Agent prompts.

Formats RAG results and file context into structured
prompt sections for LLM consumption.
"""

from pathlib import Path
from typing import Optional

from interfaces.rag_provider import RetrievalResult
from parser import SorryLocation


# =============================================================================
# Import Resolution Functions
# =============================================================================


def resolve_import_to_file(import_path: str, project_dir: str) -> Optional[Path]:
    """
    Resolve a Lean import path to an actual file path.

    Examples:
        "Dalek.Basic" → project_dir/Dalek/Basic.lean
        "Aeneas.Primitives" → project_dir/Aeneas/Primitives.lean

    Args:
        import_path: Lean import path (e.g., "Dalek.Basic")
        project_dir: Root directory of the Lean project

    Returns:
        Path to the file if it exists, None otherwise
    """
    # Convert dot notation to path
    # e.g., "Dalek.Basic" → "Dalek/Basic.lean"
    parts = import_path.split(".")
    relative_path = "/".join(parts) + ".lean"

    project_path = Path(project_dir)

    # Try direct resolution
    candidate = project_path / relative_path
    if candidate.exists():
        return candidate

    # Try in common subdirectories (lake-packages, etc.)
    for subdir in ["", "lake-packages", ".lake/packages"]:
        for pkg_dir in (project_path / subdir).glob("*") if subdir else [project_path]:
            candidate = pkg_dir / relative_path
            if candidate.exists():
                return candidate

    return None


def load_import_content(
    import_path: str,
    project_dir: str,
    max_lines: int = 200,
) -> str:
    """
    Load content from a Lean import file.

    Args:
        import_path: Lean import path
        project_dir: Root directory of the Lean project
        max_lines: Maximum lines to include (truncated if longer)

    Returns:
        File content (truncated if necessary), or empty string if not found
    """
    file_path = resolve_import_to_file(import_path, project_dir)
    if file_path is None:
        return ""

    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        if len(lines) > max_lines:
            truncated = lines[:max_lines]
            truncated.append(f"-- ... truncated ({len(lines) - max_lines} more lines)")
            return "\n".join(truncated)

        return content
    except Exception:
        return ""


def format_import_contents(
    imports: list[str],
    project_dir: str,
    max_lines_per_file: int = 150,
    max_total_lines: int = 500,
) -> str:
    """
    Format import file contents for LLM context.

    Args:
        imports: List of import statements (e.g., ["import Dalek.Basic"])
        project_dir: Root directory of the Lean project
        max_lines_per_file: Max lines to include per import
        max_total_lines: Max total lines across all imports

    Returns:
        Markdown-formatted import contents
    """
    if not imports or not project_dir:
        return ""

    sections = []
    total_lines = 0

    for imp in imports:
        if total_lines >= max_total_lines:
            sections.append(f"-- Skipped remaining imports (token budget)")
            break

        # Extract module path from import statement
        # "import Dalek.Basic" → "Dalek.Basic"
        module_path = imp.replace("import ", "").strip()

        # Skip Mathlib and standard library imports (too large)
        if module_path.startswith(("Mathlib", "Init", "Lean", "Std")):
            continue

        content = load_import_content(module_path, project_dir, max_lines_per_file)
        if not content:
            continue

        line_count = content.count("\n") + 1
        if total_lines + line_count > max_total_lines:
            # Truncate to fit budget
            remaining = max_total_lines - total_lines
            content_lines = content.split("\n")[:remaining]
            content = "\n".join(content_lines)
            content += f"\n-- ... truncated (budget)"

        sections.append(f"### {module_path}")
        sections.append("```lean")
        sections.append(content)
        sections.append("```")
        sections.append("")

        total_lines += line_count

    if not sections:
        return ""

    return "## Imported File Contents\n\n" + "\n".join(sections)


def format_context(
    sorry: SorryLocation,
    file_content: str,
    rag_results: list[RetrievalResult],
    project_dir: Optional[str] = None,
    user_context: Optional[str] = None,
) -> str:
    """
    Format all context for an LLM prompt.

    Combines:
    - RAG results (lemmas, tactics, proofs)
    - File context (imports, theorem, proof prefix)
    - Import file contents (if project_dir provided)
    - User-provided context (if provided)
    - Proof strategy hints

    Args:
        sorry: The sorry location
        file_content: Full file content
        rag_results: Retrieved RAG results
        project_dir: Optional project directory for import resolution
        user_context: Optional user-provided context string

    Returns:
        Formatted context string for prompt
    """
    sections = []

    # 1. RAG results
    rag_section = format_rag_results(rag_results)
    if rag_section:
        sections.append(rag_section)

    # 2. File context
    file_section = format_file_context(sorry, file_content)
    sections.append(file_section)

    # 3. Import file contents (if project_dir provided)
    if project_dir and sorry.imports:
        import_section = format_import_contents(sorry.imports, project_dir)
        if import_section:
            sections.append(import_section)

    # 4. User-provided context
    if user_context:
        sections.append("## Additional Context\n\n" + user_context)

    # 5. Proof hints
    hints_section = format_proof_hints(sorry)
    sections.append(hints_section)

    return "\n\n".join(sections)


def format_rag_results(results: list[RetrievalResult]) -> str:
    """
    Format RAG results as a markdown section.

    Args:
        results: List of RAG retrieval results

    Returns:
        Markdown-formatted string of available lemmas
    """
    if not results:
        return ""

    lines = ["## Available Lemmas and Tactics", ""]

    for r in results:
        # Format as: `name : type` - docstring
        entry = f"- `{r.full_name} : {r.type_signature}`"
        if r.doc_string:
            # Truncate long docstrings
            doc = r.doc_string[:100] + "..." if len(r.doc_string) > 100 else r.doc_string
            entry += f"\n  {doc}"

        lines.append(entry)

        # Include proof preview if available (helps LLM see patterns)
        if r.proof and len(r.proof) < 200:
            lines.append(f"  ```lean")
            lines.append(f"  {r.proof.strip()}")
            lines.append(f"  ```")

    return "\n".join(lines)


def format_file_context(sorry: SorryLocation, file_content: str) -> str:
    """
    Format file context around the sorry.

    Includes:
    - Imports
    - Namespace
    - Full theorem signature
    - Proof prefix (tactics before sorry)

    Args:
        sorry: The sorry location
        file_content: Full file content

    Returns:
        Markdown-formatted file context
    """
    lines = ["## File Context", ""]

    # Imports (first 15 import lines)
    if sorry.imports:
        lines.append("### Imports")
        lines.append("```lean")
        for imp in sorry.imports[:15]:
            lines.append(imp)
        if len(sorry.imports) > 15:
            lines.append(f"-- ... and {len(sorry.imports) - 15} more imports")
        lines.append("```")
        lines.append("")

    # Namespace
    if sorry.namespace:
        lines.append(f"**Namespace**: `{sorry.namespace}`")
        lines.append("")

    # Theorem with context
    lines.append("### Theorem to Prove")
    lines.append("```lean")
    lines.append(sorry.theorem_signature)

    if sorry.proof_prefix.strip():
        # Show existing tactics
        lines.append(sorry.proof_prefix.rstrip())
        lines.append("  sorry  -- FILL THIS")
    else:
        lines.append("  sorry  -- FILL THIS")

    lines.append("```")

    # Location info
    lines.append("")
    lines.append(f"**Location**: Line {sorry.line}, Column {sorry.column}")

    return "\n".join(lines)


def format_proof_hints(sorry: SorryLocation) -> str:
    """
    Format Aeneas-specific proof hints.

    Based on dalek reference prompt patterns.

    Args:
        sorry: The sorry location (used for context-specific hints)

    Returns:
        Markdown-formatted proof strategy hints
    """
    lines = [
        "## Proof Strategy",
        "",
        "Follow this general approach for Aeneas-generated code:",
        "",
        "1. **Unfold** the function being verified:",
        "   ```lean",
        "   unfold function_name",
        "   ```",
        "",
        "2. **Apply progress** repeatedly to step through monadic code:",
        "   ```lean",
        "   progress",
        "   progress",
        "   -- or use: progress*",
        "   ```",
        "",
        "3. **Handle side goals** with automation tactics:",
        "   - `grind` - General-purpose automation",
        "   - `omega` - Linear arithmetic",
        "   - `scalar_tac` - Scalar/integer reasoning",
        "   - `simp [lemma1, lemma2]` - Simplification with specific lemmas",
        "   - `decide` - For decidable propositions",
        "",
        "4. **Use available lemmas** from the context above",
        "",
        "**Important**:",
        "- Prefer automation (`grind`, `simp`, `omega`) over manual proofs",
        "- For constant equality, `unfold` + `decide` usually works",
        "- If stuck, try `progress*` to auto-apply progress repeatedly",
    ]

    # Add theorem-specific hints based on name patterns
    theorem_name = sorry.theorem_name.lower()

    if "_spec" in theorem_name:
        lines.append("")
        lines.append("**This is a spec theorem** - likely needs `unfold` + `progress*` pattern")

    if "loop" in theorem_name:
        lines.append("")
        lines.append("**This involves a loop** - may need induction or loop-specific lemmas")

    return "\n".join(lines)


def format_error_context(error: str, prev_proof: str) -> str:
    """
    Format error context for retry prompts.

    Args:
        error: The Lean compiler error message
        prev_proof: The previous proof attempt that failed

    Returns:
        Markdown-formatted error context
    """
    lines = [
        "## Previous Attempt Failed",
        "",
        "### Your Previous Proof",
        "```lean",
        prev_proof,
        "```",
        "",
        "### Error Message",
        "```",
        error,
        "```",
        "",
        "### Suggestions",
        "- Check that all lemma names are correct",
        "- Ensure types match (use explicit type annotations if needed)",
        "- Try a different tactic approach",
        "- If `progress` fails, check that spec theorems are tagged with `@[progress]`",
    ]

    return "\n".join(lines)


def format_tactic_history(
    successful_tactics: list[str],
    failed_tactics: list[str],
    max_entries: int = 5,
) -> str:
    """
    Format tactic history for context accumulation in prompts.

    Implements Poetiq pattern (Section 1.2): Prompts as state machines.
    Accumulated successful tactic patterns from prior iterations become
    in-context examples for subsequent attempts.

    Args:
        successful_tactics: Tactics that worked (can be used as positive examples)
        failed_tactics: Tactics that failed (negative examples)
        max_entries: Maximum entries to show per category (sliding window)

    Returns:
        Markdown-formatted tactic history for prompt inclusion
    """
    lines = []

    # Include successful tactics as positive examples
    if successful_tactics:
        recent_success = successful_tactics[-max_entries:]
        lines.append("## What Has Worked")
        lines.append("")
        lines.append("The following tactic patterns were successful in similar contexts:")
        lines.append("")
        for i, tactic in enumerate(recent_success, 1):
            # Truncate long tactics
            tactic_preview = tactic[:200] + "..." if len(tactic) > 200 else tactic
            lines.append(f"{i}. ```lean")
            lines.append(f"   {tactic_preview}")
            lines.append("   ```")
        lines.append("")

    # Include failed tactics as negative examples
    if failed_tactics:
        recent_failed = failed_tactics[-max_entries:]
        lines.append("## What Did NOT Work")
        lines.append("")
        lines.append("Avoid these approaches - they have already been tried and failed:")
        lines.append("")
        for i, tactic in enumerate(recent_failed, 1):
            # Truncate long tactics
            tactic_preview = tactic[:150] + "..." if len(tactic) > 150 else tactic
            lines.append(f"{i}. ~~`{tactic_preview}`~~")
        lines.append("")
        lines.append("**Try a different approach.**")

    if not lines:
        return ""

    return "\n".join(lines)
