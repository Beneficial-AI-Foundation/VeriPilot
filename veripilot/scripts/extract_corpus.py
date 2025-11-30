#!/usr/bin/env python3
"""
Extract Lean corpus from multiple sources (git repos + PDFs + books).

Orchestrates:
- Lightweight regex extraction for large/sparse repos (mathlib4)
- LeanDojo extraction for small complete repos (sphere-eversion, etc.)
- Book extraction for Lean 4 tutorial repositories (Verso + traditional formats)
- PDF code block extraction for reference documents

Usage:
    python scripts/extract_corpus.py [--priority N] [--mode MODE] [--config FILE]

Examples:
    python scripts/extract_corpus.py --priority 1 --mode auto
    python scripts/extract_corpus.py --mode lightweight  # Force lightweight only
"""
import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_json(data: list, output_path: Path):
    """Save data to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    console.print(f"[green]Saved {len(data)} entities to {output_path}[/green]")


def merge_corpora(lightweight: list, leandojo: list, book: list, pdf: list) -> list:
    """
    Merge and deduplicate corpora from different sources.

    Deduplication strategy: Prefer LeanDojo over lightweight for same full_name.
    Books and PDFs are not deduplicated (different nature - educational content).
    """
    # Create combined list
    combined = []

    # Track seen full_names
    seen = set()

    # Add LeanDojo first (higher priority)
    for entity in leandojo:
        full_name = entity.get('full_name', '')
        if full_name and full_name not in seen:
            combined.append(entity)
            seen.add(full_name)

    # Add lightweight (skip if already in LeanDojo)
    for entity in lightweight:
        full_name = entity.get('full_name', '')
        if full_name and full_name not in seen:
            combined.append(entity)
            seen.add(full_name)

    # Add book content (no deduplication - educational examples)
    combined.extend(book)

    # Add PDF blocks (no deduplication - different nature)
    combined.extend(pdf)

    return combined


def is_buildable(repo_path: Path) -> bool:
    """Check if a Lean repository is buildable (has lakefile.lean)."""
    return (repo_path / "lakefile.lean").exists() or (repo_path / "lakefile.toml").exists()


def extract_git_source_lightweight(source: dict, pattern: str) -> list:
    """
    Extract from git repo using lightweight regex parser.

    Args:
        source: Source config dict
        pattern: Glob pattern for .lean files

    Returns:
        List of declaration dicts
    """
    from rag.lean.lightweight_extractor import LightweightLeanParser

    repo_path = Path(source['path'])
    console.print(f"[blue]Using lightweight extractor for {source['name']}...[/blue]")

    parser = LightweightLeanParser()
    declarations = parser.parse_directory(repo_path, pattern=pattern)

    # Convert to unified schema
    result = []
    for decl in declarations:
        entity = decl.to_dict()
        entity['corpus_source'] = source['name']
        entity['extraction_mode'] = 'lightweight'
        entity['type'] = decl.decl_type  # Ensure type field exists
        result.append(entity)

    return result


def extract_git_source_leandojo(source: dict, cache_dir: str, num_procs: int) -> list:
    """
    Extract from git repo using LeanDojo (full compiler-level extraction).

    Args:
        source: Source config dict
        cache_dir: LeanDojo cache directory
        num_procs: Number of parallel processes

    Returns:
        List of theorem dicts
    """
    from rag.lean.extractor import LeanDojoExtractor

    repo_path = Path(source['path'])
    console.print(f"[blue]Using LeanDojo extractor for {source['name']}...[/blue]")

    extractor = LeanDojoExtractor(cache_dir=cache_dir, num_procs=num_procs)

    try:
        extraction_result = extractor.extract_local_repo(
            repo_path=repo_path,
            corpus_name=source['name'],
        )

        # Flatten file structure and convert to unified schema
        result = []
        for file_data in extraction_result.files:
            for theorem in file_data.theorems:
                entity = theorem.to_dict()
                entity['corpus_source'] = source['name']
                entity['extraction_mode'] = 'lean_dojo'
                entity['type'] = 'theorem'  # LeanDojo extracts theorems/lemmas
                entity['namespace'] = ''  # LeanDojo doesn't track namespaces separately
                entity['proof_preview'] = theorem.proof[:100] if theorem.proof else ''
                result.append(entity)

        return result

    except Exception as e:
        console.print(f"[red]LeanDojo extraction failed for {source['name']}: {e}[/red]")
        console.print("[yellow]Falling back to lightweight extractor...[/yellow]")
        # Fallback to lightweight
        pattern = source.get('patterns', ['**/*.lean'])[0]
        return extract_git_source_lightweight(source, pattern)


def extract_book_source(source: dict) -> list:
    """
    Extract educational content from a Lean 4 book repository.

    Args:
        source: Source config dict with book-specific fields:
            - name: Book identifier
            - path: Local repo path
            - content_path: Relative path to content within repo
            - format: "verso", "traditional", or "auto"

    Returns:
        List of book declaration dicts
    """
    from rag.lean.book_extractor import BookExtractor

    repo_path = Path(source['path'])
    console.print(f"[blue]Extracting book content from {source['name']}...[/blue]")
    console.print(f"[blue]  Format: {source.get('format', 'auto')}, Content: {source.get('content_path', '(root)')}[/blue]")

    try:
        extractor = BookExtractor()
        declarations = extractor.extract_book(repo_path, source)

        # Convert to unified schema
        result = []
        for decl in declarations:
            entity = decl.to_dict()
            # Ensure required fields exist
            entity['corpus_source'] = source['name']
            entity['extraction_mode'] = 'book'
            entity['type'] = decl.decl_type
            result.append(entity)

        return result

    except Exception as e:
        console.print(f"[red]Book extraction failed for {source['name']}: {e}[/red]")
        import traceback
        traceback.print_exc()
        return []


def extract_pdf_source(source: dict) -> list:
    """
    Extract code blocks from PDF.

    Args:
        source: Source config dict

    Returns:
        List of PDF code block dicts
    """
    from rag.lean.pdf_extractor import extract_pdf

    pdf_path = Path(source['path'])
    console.print(f"[blue]Extracting code from {source['name']}...[/blue]")

    try:
        blocks = extract_pdf(pdf_path, context_chars=200)

        # Convert to unified schema
        result = []
        for block in blocks:
            entity = block.to_dict()
            entity['corpus_source'] = source['name']
            entity['extraction_mode'] = 'pdf'
            result.append(entity)

        return result

    except Exception as e:
        console.print(f"[red]PDF extraction failed for {source['name']}: {e}[/red]")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Extract Lean corpus from multiple sources"
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=None,
        help="Only extract sources with this priority (1, 2, or 3)"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "lightweight", "lean_dojo"],
        default="auto",
        help="Extraction mode: auto (decide per source), lightweight (regex only), lean_dojo (force LeanDojo)"
    )
    parser.add_argument(
        "--config",
        default="config/lean_rag.yaml",
        help="Path to config file (default: config/lean_rag.yaml)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/extracted",
        help="Output directory (default: data/extracted)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if output files exist"
    )
    parser.add_argument(
        "--num-procs",
        type=int,
        default=4,
        help="Number of parallel processes for LeanDojo (default: 4)"
    )
    args = parser.parse_args()

    # Load environment
    load_dotenv()

    # Paths
    script_dir = Path(__file__).parent
    veripilot_dir = script_dir.parent
    config_path = veripilot_dir / args.config
    output_dir = veripilot_dir / args.output_dir

    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        return 1

    # Load config
    config = load_config(config_path)
    sources = config['corpus']['sources']

    # Filter by priority
    if args.priority is not None:
        sources = [s for s in sources if s.get('priority') == args.priority]
        console.print(f"[blue]Extracting Priority {args.priority} sources only ({len(sources)} sources)[/blue]")
    else:
        console.print(f"[blue]Extracting all sources ({len(sources)} sources)[/blue]")

    if not sources:
        console.print("[yellow]No sources to extract[/yellow]")
        return 0

    console.print()

    # Check if output files exist (and load for incremental extraction)
    combined_path = output_dir / "combined_corpus.json"
    lightweight_path = output_dir / "lean_lightweight.json"
    leandojo_path = output_dir / "lean_leandojo.json"
    book_path = output_dir / "book_corpus.json"
    pdf_path = output_dir / "pdf_corpus.json"

    if combined_path.exists() and not args.force:
        console.print(f"[yellow]Output already exists at {combined_path}[/yellow]")
        console.print("[blue]Loading existing extractions for incremental update...[/blue]")

        # Load existing extractions
        lean_lightweight = json.load(open(lightweight_path)) if lightweight_path.exists() else []
        lean_leandojo = json.load(open(leandojo_path)) if leandojo_path.exists() else []
        book_entities = json.load(open(book_path)) if book_path.exists() else []
        pdf_blocks = json.load(open(pdf_path)) if pdf_path.exists() else []

        # Track which sources we already have
        existing_sources = set()
        for entity in lean_lightweight + lean_leandojo + book_entities + pdf_blocks:
            existing_sources.add(entity.get('corpus_source', ''))

        # Filter to only new sources
        sources = [s for s in sources if s['name'] not in existing_sources]

        if not sources:
            console.print("[green]All sources already extracted. Use --force to re-extract.[/green]")
            return 0

        console.print(f"[blue]Extracting {len(sources)} new sources only[/blue]")
    else:
        # Fresh extraction
        lean_lightweight = []
        lean_leandojo = []
        book_entities = []
        pdf_blocks = []

    # LeanDojo cache dir
    cache_dir = os.getenv("LEAN_DOJO_CACHE_DIR", str(veripilot_dir / "data" / "lean_dojo_cache"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting sources...", total=len(sources))

        for source in sources:
            progress.update(task, description=f"Processing {source['name']}...")

            if source['type'] == 'git':
                repo_path = Path(source['path'])

                if not repo_path.exists():
                    console.print(f"[yellow]Skipping {source['name']} (not downloaded)[/yellow]")
                    progress.advance(task)
                    continue

                # Decide extraction mode
                use_lightweight = False

                if args.mode == 'lightweight':
                    use_lightweight = True
                elif args.mode == 'lean_dojo':
                    use_lightweight = False
                else:  # auto mode
                    # Use lightweight if sparse_checkout specified
                    if source.get('sparse_checkout'):
                        use_lightweight = True
                    else:
                        # Try LeanDojo if buildable, else lightweight
                        use_lightweight = not is_buildable(repo_path)

                # Extract
                pattern = source.get('patterns', ['**/*.lean'])[0]

                if use_lightweight:
                    entities = extract_git_source_lightweight(source, pattern)
                    lean_lightweight.extend(entities)
                else:
                    entities = extract_git_source_leandojo(source, cache_dir, args.num_procs)
                    lean_leandojo.extend(entities)

            elif source['type'] == 'book':
                repo_path = Path(source['path'])

                if not repo_path.exists():
                    console.print(f"[yellow]Skipping {source['name']} (not downloaded)[/yellow]")
                    progress.advance(task)
                    continue

                entities = extract_book_source(source)
                book_entities.extend(entities)

            elif source['type'] == 'pdf':
                pdf_path = Path(source['path'])

                if not pdf_path.exists():
                    console.print(f"[yellow]Skipping {source['name']} (not downloaded)[/yellow]")
                    progress.advance(task)
                    continue

                entities = extract_pdf_source(source)
                pdf_blocks.extend(entities)

            else:
                console.print(f"[yellow]Unknown source type: {source['type']}[/yellow]")

            progress.advance(task)

    # Save separate outputs
    console.print()
    save_json(lean_lightweight, output_dir / "lean_lightweight.json")
    save_json(lean_leandojo, output_dir / "lean_leandojo.json")
    save_json(book_entities, output_dir / "book_corpus.json")
    save_json(pdf_blocks, output_dir / "pdf_corpus.json")

    # Combine and deduplicate
    combined = merge_corpora(lean_lightweight, lean_leandojo, book_entities, pdf_blocks)
    save_json(combined, output_dir / "combined_corpus.json")

    # Print summary
    console.print()
    console.print("[bold green]Extraction Complete[/bold green]")
    console.print(f"  Lightweight: {len(lean_lightweight)} declarations")
    console.print(f"  LeanDojo: {len(lean_leandojo)} theorems")
    console.print(f"  Books: {len(book_entities)} examples")
    console.print(f"  PDF blocks: {len(pdf_blocks)} code blocks")
    console.print(f"  Combined: {len(combined)} total entities")

    # Type breakdown
    if combined:
        type_counts = Counter(e.get('type', 'unknown') for e in combined)
        console.print()
        console.print("[bold]Type breakdown:[/bold]")
        for typ, count in type_counts.most_common():
            console.print(f"  {typ}: {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
