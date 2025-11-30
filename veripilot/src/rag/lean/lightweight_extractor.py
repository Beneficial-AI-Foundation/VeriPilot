"""
Lightweight Lean parser using regex patterns.

This module provides fast extraction of Lean declarations without requiring
LeanDojo or building the project. It extracts:
- Theorem/lemma/def declarations
- Type signatures
- Docstrings
- Namespaces

Trade-off: No proof states, tactics, or premise data (use LeanDojo for that).
Speed: ~1000 files/minute vs ~10-50 files/minute for LeanDojo.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


@dataclass
class LightweightDeclaration:
    """A lightweight extracted declaration (no proof states)."""
    name: str
    full_name: str
    decl_type: str  # theorem, lemma, def, instance, class, structure, inductive
    type_signature: str
    file_path: str
    line_start: int
    line_end: int
    namespace: str = ""
    doc_string: Optional[str] = None
    proof_preview: str = ""  # First line of proof (if available)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "full_name": self.full_name,
            "decl_type": self.decl_type,
            "type_signature": self.type_signature,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "namespace": self.namespace,
            "doc_string": self.doc_string,
            "proof_preview": self.proof_preview,
        }


class LightweightLeanParser:
    """
    Fast regex-based Lean parser.

    Extracts declarations without building the project or using LeanDojo.
    Suitable for large repos like mathlib4 where speed is critical.
    """

    # Regex patterns for Lean 4 syntax
    THEOREM_PATTERN = re.compile(
        r'^(theorem|lemma)\s+(\w+(?:\.\w+)*)\s*'
        r'(?:\{[^}]*\}\s*)*'  # Implicit args
        r'(?:\([^)]*\)\s*)*'  # Explicit args
        r':\s*(.+?)(?::=|where)',
        re.MULTILINE
    )

    DEF_PATTERN = re.compile(
        r'^(def|abbrev)\s+(\w+(?:\.\w+)*)\s*'
        r'(?:\{[^}]*\}\s*)*'
        r'(?:\([^)]*\)\s*)*'
        r'(?::\s*(.+?))?(?::=|where)',
        re.MULTILINE
    )

    CLASS_PATTERN = re.compile(
        r'^(class|structure|inductive)\s+(\w+)\s*'
        r'(?:\{[^}]*\}\s*)*'
        r'(?:\([^)]*\)\s*)*'
        r'(?::\s*(.+?))?(?:where|:=)',
        re.MULTILINE
    )

    INSTANCE_PATTERN = re.compile(
        r'^instance\s+(?:(\w+)\s+)?:\s*(.+?)(?:where|:=)',
        re.MULTILINE
    )

    NAMESPACE_PATTERN = re.compile(r'^namespace\s+(\w+(?:\.\w+)*)', re.MULTILINE)
    END_NAMESPACE_PATTERN = re.compile(r'^end\s+(\w+(?:\.\w+)*)?', re.MULTILINE)

    DOCSTRING_PATTERN = re.compile(r'/--\s*(.*?)\s*-/', re.DOTALL)

    def __init__(self):
        """Initialize the parser."""
        self.current_namespace = ""

    def parse_file(self, file_path: Path) -> list[LightweightDeclaration]:
        """
        Parse a single .lean file and extract declarations.

        Args:
            file_path: Path to the .lean file

        Returns:
            List of extracted declarations
        """
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            console.print(f"[yellow]Warning: Could not read {file_path}: {e}[/yellow]")
            return []

        declarations = []
        lines = content.split('\n')

        # Track namespace changes
        self.current_namespace = ""
        namespace_stack = []

        for i, line in enumerate(lines):
            line_num = i + 1

            # Track namespace
            ns_match = self.NAMESPACE_PATTERN.match(line.strip())
            if ns_match:
                ns_name = ns_match.group(1)
                namespace_stack.append(ns_name)
                self.current_namespace = ".".join(namespace_stack)
                continue

            end_match = self.END_NAMESPACE_PATTERN.match(line.strip())
            if end_match and namespace_stack:
                namespace_stack.pop()
                self.current_namespace = ".".join(namespace_stack)
                continue

        # Extract declarations
        declarations.extend(self._extract_theorems(content, lines, str(file_path)))
        declarations.extend(self._extract_defs(content, lines, str(file_path)))
        declarations.extend(self._extract_classes(content, lines, str(file_path)))
        declarations.extend(self._extract_instances(content, lines, str(file_path)))

        return declarations

    def _extract_theorems(self, content: str, lines: list[str], file_path: str) -> list[LightweightDeclaration]:
        """Extract theorem and lemma declarations."""
        declarations = []

        for match in self.THEOREM_PATTERN.finditer(content):
            decl_type = match.group(1)  # theorem or lemma
            name = match.group(2)
            type_sig = match.group(3).strip()

            # Find line numbers
            start_pos = match.start()
            line_start = content[:start_pos].count('\n') + 1

            # Try to find end of declaration (simplified)
            line_end = line_start + 1

            # Extract docstring if available
            doc_string = self._extract_docstring_before(content, start_pos)

            # Extract proof preview (first line after :=)
            proof_preview = self._extract_proof_preview(content, match.end())

            # Build full name
            full_name = f"{self.current_namespace}.{name}" if self.current_namespace else name

            declarations.append(LightweightDeclaration(
                name=name.split('.')[-1],
                full_name=full_name,
                decl_type=decl_type,
                type_signature=type_sig,
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                namespace=self.current_namespace,
                doc_string=doc_string,
                proof_preview=proof_preview,
            ))

        return declarations

    def _extract_defs(self, content: str, lines: list[str], file_path: str) -> list[LightweightDeclaration]:
        """Extract def and abbrev declarations."""
        declarations = []

        for match in self.DEF_PATTERN.finditer(content):
            decl_type = match.group(1)
            name = match.group(2)
            type_sig = match.group(3).strip() if match.group(3) else ""

            start_pos = match.start()
            line_start = content[:start_pos].count('\n') + 1
            line_end = line_start + 1

            doc_string = self._extract_docstring_before(content, start_pos)
            proof_preview = self._extract_proof_preview(content, match.end())

            full_name = f"{self.current_namespace}.{name}" if self.current_namespace else name

            declarations.append(LightweightDeclaration(
                name=name.split('.')[-1],
                full_name=full_name,
                decl_type=decl_type,
                type_signature=type_sig,
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                namespace=self.current_namespace,
                doc_string=doc_string,
                proof_preview=proof_preview,
            ))

        return declarations

    def _extract_classes(self, content: str, lines: list[str], file_path: str) -> list[LightweightDeclaration]:
        """Extract class, structure, and inductive declarations."""
        declarations = []

        for match in self.CLASS_PATTERN.finditer(content):
            decl_type = match.group(1)
            name = match.group(2)
            type_sig = match.group(3).strip() if match.group(3) else ""

            start_pos = match.start()
            line_start = content[:start_pos].count('\n') + 1
            line_end = line_start + 1

            doc_string = self._extract_docstring_before(content, start_pos)

            full_name = f"{self.current_namespace}.{name}" if self.current_namespace else name

            declarations.append(LightweightDeclaration(
                name=name,
                full_name=full_name,
                decl_type=decl_type,
                type_signature=type_sig,
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                namespace=self.current_namespace,
                doc_string=doc_string,
            ))

        return declarations

    def _extract_instances(self, content: str, lines: list[str], file_path: str) -> list[LightweightDeclaration]:
        """Extract instance declarations."""
        declarations = []

        for match in self.INSTANCE_PATTERN.finditer(content):
            name = match.group(1) if match.group(1) else f"instance_{match.start()}"
            type_sig = match.group(2).strip()

            start_pos = match.start()
            line_start = content[:start_pos].count('\n') + 1
            line_end = line_start + 1

            doc_string = self._extract_docstring_before(content, start_pos)

            full_name = f"{self.current_namespace}.{name}" if self.current_namespace else name

            declarations.append(LightweightDeclaration(
                name=name,
                full_name=full_name,
                decl_type="instance",
                type_signature=type_sig,
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                namespace=self.current_namespace,
                doc_string=doc_string,
            ))

        return declarations

    def _extract_docstring_before(self, content: str, pos: int) -> Optional[str]:
        """Extract docstring immediately before a declaration."""
        # Look backwards from pos for /-- ... -/
        before = content[:pos]
        matches = list(self.DOCSTRING_PATTERN.finditer(before))
        if matches:
            last_match = matches[-1]
            # Check if docstring is close to declaration (within 100 chars)
            if pos - last_match.end() < 100:
                return last_match.group(1).strip()
        return None

    def _extract_proof_preview(self, content: str, pos: int) -> str:
        """Extract first line of proof after := or where."""
        # Get text after declaration
        after = content[pos:pos+200]  # Look ahead 200 chars
        lines = after.split('\n')
        if lines:
            first_line = lines[0].strip()
            # Remove := if present
            if first_line.startswith(':='):
                first_line = first_line[2:].strip()
            if first_line.startswith('where'):
                first_line = first_line[5:].strip()
            return first_line[:100]  # Truncate
        return ""

    def parse_directory(self, dir_path: Path, pattern: str = "**/*.lean") -> list[LightweightDeclaration]:
        """
        Parse all .lean files in a directory.

        Args:
            dir_path: Directory to search
            pattern: Glob pattern for files (default: **/*.lean)

        Returns:
            List of all extracted declarations
        """
        lean_files = list(dir_path.glob(pattern))
        console.print(f"[blue]Found {len(lean_files)} .lean files[/blue]")

        all_declarations = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Parsing files...", total=len(lean_files))

            for file_path in lean_files:
                declarations = self.parse_file(file_path)
                all_declarations.extend(declarations)
                progress.update(task, advance=1)

        console.print(f"[green]Extracted {len(all_declarations)} declarations[/green]")
        return all_declarations


def extract_lightweight(
    repo_path: Path,
    pattern: str = "**/*.lean"
) -> list[LightweightDeclaration]:
    """
    Convenience function for lightweight extraction.

    Args:
        repo_path: Path to Lean repository
        pattern: Glob pattern for files

    Returns:
        List of extracted declarations
    """
    parser = LightweightLeanParser()
    return parser.parse_directory(repo_path, pattern)
