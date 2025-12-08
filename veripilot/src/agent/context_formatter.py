"""
Context formatter for Prover Agent prompts.

Formats RAG results and file context into structured
prompt sections for LLM consumption.
"""

from interfaces.rag_provider import RetrievalResult
from parser import SorryLocation


def format_context(
    sorry: SorryLocation,
    file_content: str,
    rag_results: list[RetrievalResult],
) -> str:
    """
    Format all context for an LLM prompt.

    Combines:
    - RAG results (lemmas, tactics, proofs)
    - File context (imports, theorem, proof prefix)
    - Proof strategy hints

    Args:
        sorry: The sorry location
        file_content: Full file content
        rag_results: Retrieved RAG results

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

    # 3. Proof hints
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
