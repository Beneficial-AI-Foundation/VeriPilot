"""
PDF text and code block extractor for Lean documentation.

Extracts:
- Text content with page numbers
- Lean code blocks (```lean ... ```)
- Surrounding context for each code block
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


@dataclass
class PDFCodeBlock:
    """Represents a Lean code block extracted from a PDF."""
    code: str
    page_number: int
    block_index: int  # Index of block on this page
    context_before: str  # Text before the code block
    context_after: str  # Text after the code block
    source_file: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "code": self.code,
            "page_number": self.page_number,
            "block_index": self.block_index,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "source_file": self.source_file,
            "name": f"code_block_p{self.page_number}_{self.block_index}",
            "full_name": f"{Path(self.source_file).stem}.p{self.page_number}.{self.block_index}",
            "type": "pdf_code_block",
        }


class PDFExtractor:
    """
    Extract text and code blocks from PDF files.

    Uses PyMuPDF (fitz) for robust PDF parsing.
    """

    # Pattern for Lean code blocks in markdown
    CODE_BLOCK_PATTERN = re.compile(
        r'```lean\s*\n(.*?)\n```',
        re.DOTALL | re.MULTILINE
    )

    # Alternative: indented code (4+ spaces)
    INDENTED_CODE_PATTERN = re.compile(
        r'^((?:    |\t).+(?:\n(?:    |\t).+)*)',
        re.MULTILINE
    )

    def __init__(self):
        """Initialize PDF extractor."""
        try:
            import fitz  # PyMuPDF
            self.fitz = fitz
            self.available = True
        except ImportError:
            console.print("[yellow]PyMuPDF not installed. Install with: pip install pymupdf[/yellow]")
            self.available = False

    def extract_text(self, pdf_path: Path) -> dict[int, str]:
        """
        Extract text from PDF, organized by page.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary mapping page number to text content
        """
        if not self.available:
            raise RuntimeError("PyMuPDF not installed")

        doc = self.fitz.open(pdf_path)
        pages = {}

        for page_num in range(len(doc)):
            page = doc[page_num]
            pages[page_num + 1] = page.get_text()  # 1-indexed pages

        doc.close()
        return pages

    def extract_code_blocks(self, pdf_path: Path, context_chars: int = 200) -> list[PDFCodeBlock]:
        """
        Extract Lean code blocks from PDF with surrounding context.

        Args:
            pdf_path: Path to PDF file
            context_chars: Number of characters to include before/after code block

        Returns:
            List of extracted code blocks
        """
        if not self.available:
            console.print("[yellow]Skipping PDF extraction (PyMuPDF not installed)[/yellow]")
            return []

        console.print(f"[blue]Extracting code from {pdf_path.name}...[/blue]")

        pages_text = self.extract_text(pdf_path)
        all_blocks = []

        for page_num, text in pages_text.items():
            blocks = self._extract_blocks_from_text(
                text, page_num, str(pdf_path), context_chars
            )
            all_blocks.extend(blocks)

        console.print(f"[green]Extracted {len(all_blocks)} code blocks from {pdf_path.name}[/green]")
        return all_blocks

    def _extract_blocks_from_text(
        self,
        text: str,
        page_num: int,
        source_file: str,
        context_chars: int
    ) -> list[PDFCodeBlock]:
        """Extract code blocks from a single page of text."""
        blocks = []
        block_index = 0

        # Try markdown-style code blocks first
        for match in self.CODE_BLOCK_PATTERN.finditer(text):
            code = match.group(1).strip()
            if not code:
                continue

            # Get surrounding context
            start_pos = match.start()
            end_pos = match.end()

            context_before = text[max(0, start_pos - context_chars):start_pos].strip()
            context_after = text[end_pos:min(len(text), end_pos + context_chars)].strip()

            blocks.append(PDFCodeBlock(
                code=code,
                page_number=page_num,
                block_index=block_index,
                context_before=context_before,
                context_after=context_after,
                source_file=source_file,
            ))
            block_index += 1

        # If no markdown blocks found, try indented code
        if not blocks:
            blocks.extend(self._extract_indented_code(text, page_num, source_file, context_chars))

        return blocks

    def _extract_indented_code(
        self,
        text: str,
        page_num: int,
        source_file: str,
        context_chars: int
    ) -> list[PDFCodeBlock]:
        """Extract indented code blocks (fallback method)."""
        blocks = []
        block_index = 0

        for match in self.INDENTED_CODE_PATTERN.finditer(text):
            code = match.group(1)
            # Remove indentation
            lines = code.split('\n')
            dedented = '\n'.join(line[4:] if line.startswith('    ') else line[1:] if line.startswith('\t') else line for line in lines)
            dedented = dedented.strip()

            # Only include if it looks like Lean code (has keywords)
            if not self._looks_like_lean(dedented):
                continue

            start_pos = match.start()
            end_pos = match.end()

            context_before = text[max(0, start_pos - context_chars):start_pos].strip()
            context_after = text[end_pos:min(len(text), end_pos + context_chars)].strip()

            blocks.append(PDFCodeBlock(
                code=dedented,
                page_number=page_num,
                block_index=block_index,
                context_before=context_before,
                context_after=context_after,
                source_file=source_file,
            ))
            block_index += 1

        return blocks

    def _looks_like_lean(self, code: str) -> bool:
        """Heuristic to check if code looks like Lean."""
        lean_keywords = [
            'theorem', 'lemma', 'def', 'example', 'variable',
            'import', 'open', 'namespace', 'class', 'structure',
            'inductive', 'axiom', '#check', '#eval'
        ]
        code_lower = code.lower()
        return any(keyword in code_lower for keyword in lean_keywords)


def extract_pdf(pdf_path: Path, context_chars: int = 200) -> list[PDFCodeBlock]:
    """
    Convenience function to extract code blocks from a PDF.

    Args:
        pdf_path: Path to PDF file
        context_chars: Characters of context before/after code blocks

    Returns:
        List of extracted code blocks
    """
    extractor = PDFExtractor()
    return extractor.extract_code_blocks(pdf_path, context_chars)
