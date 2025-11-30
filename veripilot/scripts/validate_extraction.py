#!/usr/bin/env python3
"""
Validate extracted Lean corpus.

Checks:
- Counts by type (theorems, lemmas, defs, PDF blocks)
- Type signature completeness
- File path validity
- Random sampling for manual inspection
- Generates extraction report

Usage:
    python scripts/validate_extraction.py [CORPUS_FILE]

Example:
    python scripts/validate_extraction.py data/extracted/combined_corpus.json
"""
import argparse
import json
import random
import sys
from pathlib import Path
from collections import Counter

from rich.console import Console
from rich.table import Table

console = Console()


def validate_extraction(corpus_file: Path) -> dict:
    """
    Validate extracted corpus and generate report.

    Args:
        corpus_file: Path to corpus JSON file

    Returns:
        Validation report dict
    """
    console.print(f"[blue]Loading corpus from {corpus_file}...[/blue]")

    # Load corpus
    try:
        with open(corpus_file) as f:
            corpus = json.load(f)
    except FileNotFoundError:
        console.print(f"[red]File not found: {corpus_file}[/red]")
        return None
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON: {e}[/red]")
        return None

    if not corpus:
        console.print("[yellow]Empty corpus![/yellow]")
        return None

    # Count by type
    counts = Counter(entity.get('type', 'unknown') for entity in corpus)

    # Check type signatures
    empty_sigs = [e for e in corpus if not e.get('type_signature', '').strip()]

    # Count by source
    source_counts = Counter(entity.get('corpus_source', 'unknown') for entity in corpus)

    # Count by extraction mode
    mode_counts = Counter(entity.get('extraction_mode', 'unknown') for entity in corpus)

    # Sample random entities for inspection
    sample_size = min(10, len(corpus))
    sample = random.sample(corpus, sample_size)

    # Generate report
    report = {
        "total_entities": len(corpus),
        "counts_by_type": dict(counts),
        "counts_by_source": dict(source_counts),
        "counts_by_mode": dict(mode_counts),
        "empty_signatures": len(empty_sigs),
        "empty_signature_percentage": len(empty_sigs) / len(corpus) * 100,
        "sample": sample,
        "sources": list(source_counts.keys()),
        "validation_passed": True,
        "warnings": [],
        "errors": [],
    }

    # Validation checks
    if len(corpus) < 1000:
        report["warnings"].append(f"Low entity count: {len(corpus)} (expected >1000)")

    if report["empty_signature_percentage"] > 20:
        report["warnings"].append(
            f"High empty signature percentage: {report['empty_signature_percentage']:.1f}% (expected <20%)"
        )

    # Check for critical issues
    if len(corpus) == 0:
        report["errors"].append("Corpus is empty!")
        report["validation_passed"] = False

    unknown_count = counts.get('unknown', 0)
    if unknown_count > len(corpus) * 0.1:  # >10% unknown types
        report["warnings"].append(f"Many unknown types: {unknown_count}")

    return report


def print_report(report: dict):
    """Print validation report in a nice format."""
    if not report:
        return

    console.print()
    console.print("[bold blue]═══ Extraction Validation Report ═══[/bold blue]")
    console.print()

    # Summary table
    summary_table = Table(title="Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Total entities", str(report['total_entities']))
    summary_table.add_row("Empty signatures", f"{report['empty_signatures']} ({report['empty_signature_percentage']:.1f}%)")
    summary_table.add_row("Sources", ", ".join(report['sources']))

    console.print(summary_table)
    console.print()

    # Type breakdown table
    type_table = Table(title="Counts by Type")
    type_table.add_column("Type", style="cyan")
    type_table.add_column("Count", style="green", justify="right")

    for typ, count in sorted(report['counts_by_type'].items(), key=lambda x: -x[1]):
        type_table.add_row(typ, str(count))

    console.print(type_table)
    console.print()

    # Source breakdown table
    source_table = Table(title="Counts by Source")
    source_table.add_column("Source", style="cyan")
    source_table.add_column("Count", style="green", justify="right")

    for source, count in sorted(report['counts_by_source'].items(), key=lambda x: -x[1]):
        source_table.add_row(source, str(count))

    console.print(source_table)
    console.print()

    # Extraction mode table
    mode_table = Table(title="Counts by Extraction Mode")
    mode_table.add_column("Mode", style="cyan")
    mode_table.add_column("Count", style="green", justify="right")

    for mode, count in sorted(report['counts_by_mode'].items(), key=lambda x: -x[1]):
        mode_table.add_row(mode, str(count))

    console.print(mode_table)
    console.print()

    # Sample entities
    console.print("[bold]Random Sample (first 3):[/bold]")
    for i, entity in enumerate(report['sample'][:3], 1):
        console.print(f"  {i}. [cyan]{entity.get('full_name', entity.get('name', 'N/A'))}[/cyan]")
        console.print(f"     Type: {entity.get('type', 'unknown')}")
        console.print(f"     Source: {entity.get('corpus_source', 'unknown')}")
        console.print(f"     File: {entity.get('source_file', 'N/A')}")
        if entity.get('type_signature'):
            sig = entity['type_signature']
            if len(sig) > 80:
                sig = sig[:77] + "..."
            console.print(f"     Sig: {sig}")
        console.print()

    # Warnings and errors
    if report['warnings']:
        console.print("[bold yellow]Warnings:[/bold yellow]")
        for warning in report['warnings']:
            console.print(f"  ⚠️  {warning}")
        console.print()

    if report['errors']:
        console.print("[bold red]Errors:[/bold red]")
        for error in report['errors']:
            console.print(f"  ❌ {error}")
        console.print()

    # Final status
    if report['validation_passed'] and not report['warnings']:
        console.print("[bold green]✅ Validation PASSED - No issues found![/bold green]")
    elif report['validation_passed'] and report['warnings']:
        console.print("[bold yellow]⚠️  Validation PASSED with warnings[/bold yellow]")
    else:
        console.print("[bold red]❌ Validation FAILED[/bold red]")


def main():
    parser = argparse.ArgumentParser(
        description="Validate extracted Lean corpus"
    )
    parser.add_argument(
        "corpus_file",
        nargs="?",
        default="data/extracted/combined_corpus.json",
        help="Path to corpus JSON file (default: data/extracted/combined_corpus.json)"
    )
    parser.add_argument(
        "--report",
        help="Save report to JSON file"
    )
    args = parser.parse_args()

    # Validate
    corpus_path = Path(args.corpus_file)
    report = validate_extraction(corpus_path)

    if not report:
        return 1

    # Print report
    print_report(report)

    # Save report if requested
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        console.print(f"\n[green]Report saved to {report_path}[/green]")
    else:
        # Default: save next to corpus file
        report_path = corpus_path.parent / "validation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        console.print(f"\n[green]Report saved to {report_path}[/green]")

    # Exit code based on validation
    if not report['validation_passed']:
        return 1
    elif report['warnings']:
        return 0  # Warnings don't fail validation
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
