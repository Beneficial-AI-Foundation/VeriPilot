#!/usr/bin/env python3
"""
Extract Lean corpus using LeanDojo.

This script extracts theorem data from the dalek-verify-lean submodule
using LeanDojo's compiler-level extraction.

Usage:
    python scripts/extract_corpus.py [--output OUTPUT] [--config CONFIG]

Example:
    python scripts/extract_corpus.py --output data/extracted/lean_corpus.json
"""
import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Extract Lean corpus using LeanDojo"
    )
    parser.add_argument(
        "--repo",
        default="../lean-projects/dalek-verify-lean",
        help="Path to Lean repository (relative to veripilot/)",
    )
    parser.add_argument(
        "--output",
        default="data/extracted/lean_corpus.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--name",
        default="dalek-verify-lean",
        help="Corpus name",
    )
    parser.add_argument(
        "--num-procs",
        type=int,
        default=4,
        help="Number of parallel processes",
    )
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Resolve paths
    script_dir = Path(__file__).parent
    veripilot_dir = script_dir.parent
    repo_path = (veripilot_dir / args.repo).resolve()
    output_path = (veripilot_dir / args.output).resolve()

    console.print("[bold blue]VeriPilot Lean Corpus Extraction[/bold blue]")
    console.print(f"Repository: {repo_path}")
    console.print(f"Output: {output_path}")
    console.print()

    # Check if repo exists
    if not repo_path.exists():
        console.print(f"[red]Error: Repository not found at {repo_path}[/red]")
        console.print("[yellow]Make sure the dalek-verify-lean submodule is initialized:[/yellow]")
        console.print("  git submodule update --init --recursive")
        sys.exit(1)

    # Check if output already exists
    if output_path.exists():
        console.print(f"[yellow]Output file already exists: {output_path}[/yellow]")
        response = input("Overwrite? [y/N] ").strip().lower()
        if response != "y":
            console.print("[blue]Aborted.[/blue]")
            sys.exit(0)

    # Import extractor
    try:
        from rag.lean.extractor import LeanDojoExtractor
    except ImportError as e:
        console.print(f"[red]Import error: {e}[/red]")
        console.print("[yellow]Make sure you've installed dependencies:[/yellow]")
        console.print("  pip install -e .")
        sys.exit(1)

    # Create extractor
    cache_dir = os.getenv("LEAN_DOJO_CACHE_DIR", "/workspace/data/lean_dojo_cache")
    extractor = LeanDojoExtractor(
        cache_dir=cache_dir,
        num_procs=args.num_procs,
    )

    # Run extraction
    try:
        result = extractor.extract_local_repo(
            repo_path=repo_path,
            corpus_name=args.name,
        )

        # Save result
        result.save(output_path)

        # Print summary
        console.print()
        console.print("[bold green]Extraction Summary[/bold green]")
        console.print(f"  Corpus: {result.corpus_name}")
        console.print(f"  Commit: {result.commit}")
        console.print(f"  Files: {len(result.files)}")
        console.print(f"  Theorems: {result.total_theorems}")
        console.print(f"  Tactic steps: {result.total_tactics}")
        console.print(f"  Output: {output_path}")

    except Exception as e:
        console.print(f"[red]Extraction failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
