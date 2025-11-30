"""
Extract educational content from Lean 4 book repositories.

Handles two formats:
1. Verso MD format (fp-lean, tpil4) - literate .lean files with #doc directives
2. Traditional format (mil, metaprogramming) - standard .lean with comments

Each format has a dedicated parser that extracts code examples with surrounding
context (prose explanations). This is crucial for RAG because the educational
context helps understand when and how to apply certain tactics or patterns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


@dataclass
class BookDeclaration:
    """
    A declaration extracted from a Lean 4 educational book.

    Includes chapter/section hierarchy and surrounding prose context,
    which is valuable for RAG retrieval - understanding *why* a tactic
    is used is as important as knowing *what* it does.
    """
    name: str
    full_name: str
    decl_type: str  # "example", "theorem", "def", "verso_anchor", etc.
    code: str  # The actual Lean code
    type_signature: str
    file_path: str
    line_start: int
    line_end: int
    chapter: str
    section: str
    book_source: str  # "tpil4", "fp-lean", "mil", "metaprogramming"
    prose_before: str = ""  # Surrounding explanation (200 chars)
    prose_after: str = ""  # Surrounding explanation (200 chars)
    anchor: Optional[str] = None  # Verso anchor name if present
    tags: list[str] = field(default_factory=list)  # Verso metadata tags

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "full_name": self.full_name,
            "decl_type": self.decl_type,
            "code": self.code,
            "type_signature": self.type_signature,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "chapter": self.chapter,
            "section": self.section,
            "corpus_source": self.book_source,
            "extraction_mode": "book",
            "doc_string": self.prose_before,
            "proof_preview": self.code[:200] if self.code else "",
            "namespace": f"{self.chapter}.{self.section}".replace(" ", "_"),
            "anchor": self.anchor,
            "tags": self.tags,
        }


class VersoParser:
    """
    Parse Verso MD format .lean files.

    Verso is a literate programming format used by:
    - Functional Programming in Lean (fp-lean)
    - Theorem Proving in Lean 4 (tpil4)

    Key syntax patterns:
    - #doc (Manual) "Title" => declares a document
    - %%% ... %%% contains metadata (tags, htmlSplit, etc.)
    - ```anchor name ... ``` marks code examples
    - # and ## create section headers
    """

    # Regex patterns for Verso syntax
    DOC_PATTERN = re.compile(
        r'#doc\s*\([^)]+\)\s*"([^"]+)"\s*=>',
        re.MULTILINE
    )

    METADATA_PATTERN = re.compile(
        r'%%%\n(.*?)\n%%%',
        re.DOTALL
    )

    # Match various anchor types: anchor, anchorTerm, anchorInfo, anchorEvalStep, anchorEvalSteps
    ANCHOR_PATTERN = re.compile(
        r'```(anchor|anchorTerm|anchorInfo|anchorEvalStep|anchorEvalSteps)\s+(\w+)(?:\s+\d+)?\n(.*?)\n```',
        re.DOTALL
    )

    SECTION_PATTERN = re.compile(
        r'^(#{1,3})\s+(.+)$',
        re.MULTILINE
    )

    def parse_file(self, file_path: Path, book_source: str) -> list[BookDeclaration]:
        """
        Parse a Verso format .lean file.

        Args:
            file_path: Path to the .lean file
            book_source: Name of the book source (e.g., "tpil4")

        Returns:
            List of extracted BookDeclaration objects
        """
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            console.print(f"[yellow]Warning: Could not read {file_path}: {e}[/yellow]")
            return []

        declarations = []

        # Extract chapter title from #doc directive
        chapter = "Unknown"
        doc_match = self.DOC_PATTERN.search(content)
        if doc_match:
            chapter = doc_match.group(1)

        # Extract all sections for context tracking
        current_section = ""
        section_matches = list(self.SECTION_PATTERN.finditer(content))

        # Extract anchored code blocks
        for match in self.ANCHOR_PATTERN.finditer(content):
            anchor_type = match.group(1)
            anchor_name = match.group(2)
            code = match.group(3).strip()

            # Find surrounding context (prose before and after)
            start_pos = match.start()
            end_pos = match.end()
            prose_before = self._clean_prose(content[max(0, start_pos-300):start_pos])
            prose_after = self._clean_prose(content[end_pos:min(len(content), end_pos+300)])

            # Determine current section based on position
            for section_match in section_matches:
                if section_match.start() < start_pos:
                    current_section = section_match.group(2).strip()

            # Calculate line numbers
            line_start = content[:start_pos].count('\n') + 1
            line_end = content[:end_pos].count('\n') + 1

            decl = BookDeclaration(
                name=anchor_name,
                full_name=f"{book_source}.{chapter}.{anchor_name}",
                decl_type=f"verso_{anchor_type}",
                code=code,
                type_signature="",  # Verso anchors don't have explicit signatures
                file_path=str(file_path),
                line_start=line_start,
                line_end=line_end,
                chapter=chapter,
                section=current_section,
                book_source=book_source,
                prose_before=prose_before,
                prose_after=prose_after,
                anchor=anchor_name,
                tags=self._extract_tags(content, start_pos)
            )
            declarations.append(decl)

        return declarations

    def _clean_prose(self, text: str) -> str:
        """Clean prose text by removing Verso-specific markup."""
        # Remove inline markup like {lit}, {kw}, {name}, etc.
        text = re.sub(r'\{[^}]+\}', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Truncate to 200 chars
        return text[-200:] if len(text) > 200 else text

    def _extract_tags(self, content: str, pos: int) -> list[str]:
        """Extract tags from nearby %%% metadata blocks."""
        tags = []
        # Find metadata blocks before position
        for match in self.METADATA_PATTERN.finditer(content):
            if match.end() < pos and pos - match.end() < 500:  # Within 500 chars
                meta_content = match.group(1)
                if 'tag :=' in meta_content:
                    tag_match = re.search(r'tag\s*:=\s*"([^"]+)"', meta_content)
                    if tag_match:
                        tags.append(tag_match.group(1))
        return tags


class TraditionalBookParser:
    """
    Parse traditional Lean comment-based book files.

    Used by:
    - Mathematics in Lean (mil)
    - Lean 4 Metaprogramming Book (metaprogramming)

    These books use standard Lean syntax with:
    - Block comments /- ... -/ for prose sections
    - Line comments -- for inline explanations
    - example/theorem/lemma/def for code demonstrations
    - sorry placeholders for exercises
    """

    # Patterns for comments (prose context)
    BLOCK_COMMENT_PATTERN = re.compile(r'/-(.*?)-/', re.DOTALL)
    LINE_COMMENT_PATTERN = re.compile(r'--\s*(.*)$', re.MULTILINE)

    # Pattern for declarations - capture the full declaration including body
    # This is more permissive than lightweight_extractor to capture examples
    DECLARATION_PATTERN = re.compile(
        r'^(example|theorem|lemma|def)\s+(\w+)?\s*'  # declaration type and optional name
        r'(?:\{[^}]*\}\s*)*'  # implicit args {α : Type}
        r'(?:\([^)]*\)\s*)*'  # explicit args (n : Nat)
        r'(?::\s*([^\n:=]+?))?'  # type signature
        r'\s*:=\s*'  # assignment
        r'((?:by\s+.*?(?=\n(?:example|theorem|lemma|def|#|end\b|$))|.*?(?=\n(?:example|theorem|lemma|def|#|end\b|$))))',
        re.MULTILINE | re.DOTALL
    )

    # Simpler pattern for capturing any example/theorem/lemma/def block
    SIMPLE_DECL_PATTERN = re.compile(
        r'^(example|theorem|lemma|def)\b[^\n]*\n(?:(?!example\b|theorem\b|lemma\b|def\b|#check\b|#eval\b|section\b|end\b)[^\n]*\n)*',
        re.MULTILINE
    )

    def parse_file(self, file_path: Path, book_source: str) -> list[BookDeclaration]:
        """
        Parse a traditional format .lean file.

        Args:
            file_path: Path to the .lean file
            book_source: Name of the book source (e.g., "mil")

        Returns:
            List of extracted BookDeclaration objects
        """
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            console.print(f"[yellow]Warning: Could not read {file_path}: {e}[/yellow]")
            return []

        declarations = []

        # Extract chapter/section from file path
        # e.g., MIL/C02_Basics/S01_Calculating.lean
        chapter, section = self._parse_path_metadata(file_path)

        # Find all declaration blocks
        for match in self.SIMPLE_DECL_PATTERN.finditer(content):
            code = match.group(0).strip()
            if not code:
                continue

            # Parse the declaration line to extract type and name
            first_line = code.split('\n')[0]
            decl_type = first_line.split()[0] if first_line.split() else "example"

            # Try to extract name
            name_match = re.match(r'(?:example|theorem|lemma|def)\s+(\w+)', first_line)
            name = name_match.group(1) if name_match else f"anonymous_{match.start()}"

            start_pos = match.start()
            end_pos = match.end()

            # Get surrounding comments as prose
            prose_before = self._get_preceding_comments(content, start_pos)
            prose_after = self._get_following_comments(content, end_pos)

            line_start = content[:start_pos].count('\n') + 1
            line_end = content[:end_pos].count('\n') + 1

            # Extract type signature if present
            type_signature = self._extract_type_signature(code)

            decl = BookDeclaration(
                name=name,
                full_name=f"{book_source}.{chapter}.{section}.{name}",
                decl_type=decl_type,
                code=code,
                type_signature=type_signature,
                file_path=str(file_path),
                line_start=line_start,
                line_end=line_end,
                chapter=chapter,
                section=section,
                book_source=book_source,
                prose_before=prose_before,
                prose_after=prose_after,
            )
            declarations.append(decl)

        return declarations

    def _parse_path_metadata(self, file_path: Path) -> tuple[str, str]:
        """
        Extract chapter/section from file path.

        Examples:
        - MIL/C02_Basics/S01_Calculating.lean -> ("Basics", "Calculating")
        - lean/main/01_intro.lean -> ("01_intro", "01_intro")
        """
        parts = file_path.parts
        chapter = "Unknown"
        section = file_path.stem

        for part in parts:
            # Mathematics in Lean pattern: C##_Name
            if part.startswith('C') and '_' in part and part[1:3].isdigit():
                chapter = part.split('_', 1)[1] if '_' in part else part
            # Section pattern: S##_Name
            elif part.startswith('S') and '_' in part and part[1:3].isdigit():
                section = part.split('_', 1)[1] if '_' in part else part

        # Metaprogramming book pattern: ##_name.lean
        if chapter == "Unknown":
            stem = file_path.stem
            if re.match(r'^\d{2}_', stem):
                chapter = stem
                section = stem

        return chapter, section

    def _get_preceding_comments(self, content: str, pos: int) -> str:
        """Get comments before a position as prose context."""
        preceding = content[max(0, pos-500):pos]
        comments = []

        # Extract block comments
        for match in self.BLOCK_COMMENT_PATTERN.finditer(preceding):
            comment_text = match.group(1).strip()
            if comment_text:
                comments.append(comment_text)

        # Extract line comments (usually more relevant for immediate context)
        for match in self.LINE_COMMENT_PATTERN.finditer(preceding):
            comment_text = match.group(1).strip()
            if comment_text:
                comments.append(comment_text)

        result = ' '.join(comments)
        return result[-200:] if len(result) > 200 else result

    def _get_following_comments(self, content: str, pos: int) -> str:
        """Get comments after a position as prose context."""
        following = content[pos:min(len(content), pos+500)]
        comments = []

        for match in self.LINE_COMMENT_PATTERN.finditer(following):
            comment_text = match.group(1).strip()
            if comment_text:
                comments.append(comment_text)

        result = ' '.join(comments)
        return result[:200] if len(result) > 200 else result

    def _extract_type_signature(self, code: str) -> str:
        """Extract type signature from declaration code."""
        # Look for : ... := pattern
        match = re.search(r':\s*([^:=]+?)\s*:=', code, re.DOTALL)
        if match:
            sig = match.group(1).strip()
            # Clean up multiline signatures
            sig = re.sub(r'\s+', ' ', sig)
            return sig
        return ""


class BookExtractor:
    """
    Unified extractor for Lean 4 book repositories.

    Automatically detects format (Verso vs Traditional) and applies
    the appropriate parser. Can also accept explicit format hints
    from configuration.
    """

    def __init__(self):
        """Initialize with both parsers."""
        self.verso_parser = VersoParser()
        self.traditional_parser = TraditionalBookParser()

    def extract_book(self, repo_path: Path, book_config: dict) -> list[BookDeclaration]:
        """
        Extract all content from a book repository.

        Args:
            repo_path: Path to the cloned repository
            book_config: Configuration dict with keys:
                - name: Book identifier (e.g., "tpil4")
                - content_path: Relative path to content (e.g., "book/TPiL")
                - format: "verso", "traditional", or "auto"

        Returns:
            List of BookDeclaration objects
        """
        book_source = book_config['name']
        book_format = book_config.get('format', 'auto')
        content_rel_path = book_config.get('content_path', '')

        # Handle both relative and absolute content paths
        if content_rel_path:
            content_path = repo_path / content_rel_path
        else:
            content_path = repo_path

        if not content_path.exists():
            console.print(f"[yellow]Warning: Content path not found: {content_path}[/yellow]")
            return []

        declarations = []

        # Detect format if auto
        if book_format == 'auto':
            book_format = self._detect_format(content_path)
            console.print(f"[blue]Detected format for {book_source}: {book_format}[/blue]")

        # Get all .lean files
        lean_files = list(content_path.rglob('*.lean'))
        console.print(f"[blue]Found {len(lean_files)} .lean files in {book_source}[/blue]")

        # Skip patterns - files that aren't content
        skip_patterns = ['.lake', 'lakefile', 'Main.lean', 'lean-toolchain']

        with Progress(
            SpinnerColumn(),
            TextColumn(f"[progress.description]Extracting {book_source}..."),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting...", total=len(lean_files))

            for lean_file in lean_files:
                # Skip non-content files
                if any(skip in str(lean_file) for skip in skip_patterns):
                    progress.update(task, advance=1)
                    continue

                try:
                    if book_format == 'verso':
                        decls = self.verso_parser.parse_file(lean_file, book_source)
                    else:
                        decls = self.traditional_parser.parse_file(lean_file, book_source)
                    declarations.extend(decls)
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to parse {lean_file}: {e}[/yellow]")

                progress.update(task, advance=1)

        console.print(f"[green]Extracted {len(declarations)} declarations from {book_source}[/green]")
        return declarations

    def _detect_format(self, content_path: Path) -> str:
        """
        Detect book format by checking for Verso markers.

        Verso files contain #doc directives or %%% metadata blocks.
        Traditional files use standard Lean syntax with comments.
        """
        for lean_file in content_path.rglob('*.lean'):
            try:
                content = lean_file.read_text(encoding='utf-8')[:2000]
                if '#doc' in content or '%%%' in content:
                    return 'verso'
            except:
                pass
        return 'traditional'


def extract_books(
    sources: list[dict],
    base_path: Path = Path("data/sources/books")
) -> list[BookDeclaration]:
    """
    Extract content from multiple book repositories.

    Args:
        sources: List of book config dicts from lean_rag.yaml
        base_path: Base directory where books are cloned

    Returns:
        Combined list of all BookDeclaration objects
    """
    extractor = BookExtractor()
    all_declarations = []

    for source in sources:
        if source.get('type') != 'book':
            continue

        repo_path = Path(source['path'])
        if not repo_path.exists():
            console.print(f"[yellow]Skipping {source['name']} (not downloaded)[/yellow]")
            continue

        declarations = extractor.extract_book(repo_path, source)
        all_declarations.extend(declarations)

    console.print(f"\n[bold green]Total book declarations: {len(all_declarations)}[/bold green]")
    return all_declarations
