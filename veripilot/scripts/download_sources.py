#!/usr/bin/env python3
"""
Download corpus sources from config/lean_rag.yaml.

Supports:
- Git repositories (with sparse checkout for large repos)
- PDF downloads
- Priority filtering
- Resume capability (skip already-downloaded sources)

Usage:
    python scripts/download_sources.py [--priority 1] [--force]

Examples:
    python scripts/download_sources.py --priority 1  # Download Priority 1 only
    python scripts/download_sources.py --force       # Re-download even if exists
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import requests
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn

console = Console()


def download_git_repo(source: dict, force: bool = False) -> bool:
    """
    Download a git repository (with optional sparse checkout).

    Args:
        source: Source configuration dict
        force: Force re-download even if exists

    Returns:
        True if successful, False otherwise
    """
    name = source["name"]
    url = source["url"]
    path = Path(source["path"])

    if path.exists() and not force:
        console.print(f"[yellow]Skipping {name} (already exists at {path})[/yellow]")
        return True

    console.print(f"[blue]Cloning {name} from {url}...[/blue]")

    # Create parent directory
    path.parent.mkdir(parents=True, exist_ok=True)

    # Check if sparse checkout is specified
    sparse_folders = source.get("sparse_checkout", [])

    if sparse_folders:
        console.print(f"[blue]Using sparse checkout for {len(sparse_folders)} folders[/blue]")
        return _git_sparse_clone(url, path, sparse_folders)
    else:
        return _git_full_clone(url, path)


def _git_full_clone(url: str, path: Path) -> bool:
    """Clone entire repository."""
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        console.print(f"[green]Cloned to {path}[/green]")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Git clone failed: {e.stderr}[/red]")
        return False


def _git_sparse_clone(url: str, path: Path, folders: list[str]) -> bool:
    """Clone repository with sparse checkout (only specified folders)."""
    try:
        # Initialize empty repo
        path.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "init"],
            cwd=path,
            capture_output=True,
            check=True,
        )

        # Add remote
        subprocess.run(
            ["git", "remote", "add", "origin", url],
            cwd=path,
            capture_output=True,
            check=True,
        )

        # Enable sparse checkout
        subprocess.run(
            ["git", "config", "core.sparseCheckout", "true"],
            cwd=path,
            capture_output=True,
            check=True,
        )

        # Write sparse-checkout file
        sparse_file = path / ".git" / "info" / "sparse-checkout"
        sparse_file.parent.mkdir(parents=True, exist_ok=True)
        sparse_file.write_text("\n".join(folders) + "\n")

        console.print(f"[blue]Pulling sparse checkout: {', '.join(folders[:3])}{'...' if len(folders) > 3 else ''}[/blue]")

        # Pull (this will only download specified folders)
        subprocess.run(
            ["git", "pull", "--depth", "1", "origin", "master"],
            cwd=path,
            capture_output=True,
            check=False,  # master might not exist, try main
        )

        # Try main if master failed
        subprocess.run(
            ["git", "pull", "--depth", "1", "origin", "main"],
            cwd=path,
            capture_output=True,
            check=False,
        )

        # Verify something was downloaded
        lean_files = list(path.glob("**/*.lean"))
        if lean_files:
            console.print(f"[green]Sparse checkout complete: {len(lean_files)} .lean files[/green]")
            return True
        else:
            console.print(f"[yellow]Warning: No .lean files found after sparse checkout[/yellow]")
            return False

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Sparse checkout failed: {e.stderr if hasattr(e, 'stderr') else str(e)}[/red]")
        return False


def download_pdf(source: dict, force: bool = False) -> bool:
    """
    Download a PDF file.

    Args:
        source: Source configuration dict
        force: Force re-download even if exists

    Returns:
        True if successful, False otherwise
    """
    name = source["name"]
    url = source["url"]
    path = Path(source["path"])

    # Skip if URL is empty (local file)
    if not url or not url.strip():
        if path.exists():
            console.print(f"[green]Local file {name} found at {path}[/green]")
            return True
        else:
            console.print(f"[yellow]Skipping {name} (local file expected at {path}, not found)[/yellow]")
            console.print(f"[yellow]Please upload your PDF to: {path}[/yellow]")
            return False

    if path.exists() and not force:
        console.print(f"[yellow]Skipping {name} (already exists at {path})[/yellow]")
        return True

    console.print(f"[blue]Downloading {name} from {url}...[/blue]")

    # Create parent directory
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        # Get total size if available
        total_size = int(response.headers.get('content-length', 0))

        with open(path, 'wb') as f:
            if total_size == 0:
                # No content-length header
                f.write(response.content)
                console.print(f"[green]Downloaded to {path}[/green]")
            else:
                # Show progress bar
                with Progress(
                    *Progress.get_default_columns(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task(f"Downloading {name}", total=total_size)

                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))

                console.print(f"[green]Downloaded to {path} ({total_size / 1024 / 1024:.1f} MB)[/green]")

        return True

    except requests.RequestException as e:
        console.print(f"[red]Download failed: {e}[/red]")
        if path.exists():
            path.unlink()  # Remove partial download
        return False


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_manifest(manifest_path: Path, manifest: dict):
    """Save download manifest."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Download corpus sources for Lean RAG"
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=None,
        help="Only download sources with this priority (1, 2, or 3)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if source already exists"
    )
    parser.add_argument(
        "--config",
        default="config/lean_rag.yaml",
        help="Path to config file (default: config/lean_rag.yaml)"
    )
    args = parser.parse_args()

    # Load environment
    load_dotenv()

    # Paths
    script_dir = Path(__file__).parent
    veripilot_dir = script_dir.parent
    config_path = veripilot_dir / args.config

    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        return 1

    # Load config
    config = load_config(config_path)
    sources = config["corpus"]["sources"]

    # Filter by priority if specified
    if args.priority is not None:
        sources = [s for s in sources if s.get("priority") == args.priority]
        console.print(f"[blue]Downloading Priority {args.priority} sources only ({len(sources)} sources)[/blue]")
    else:
        console.print(f"[blue]Downloading all sources ({len(sources)} sources)[/blue]")

    if not sources:
        console.print("[yellow]No sources to download[/yellow]")
        return 0

    console.print()

    # Download each source
    manifest = {
        "sources": [],
        "total": len(sources),
        "successful": 0,
        "failed": 0,
    }

    for i, source in enumerate(sources, 1):
        name = source["name"]
        source_type = source["type"]

        console.print(f"[bold]({i}/{len(sources)}) Processing {name}[/bold]")

        success = False
        if source_type == "git":
            success = download_git_repo(source, force=args.force)
        elif source_type == "pdf":
            success = download_pdf(source, force=args.force)
        else:
            console.print(f"[yellow]Unknown source type: {source_type}[/yellow]")

        manifest["sources"].append({
            "name": name,
            "type": source_type,
            "path": source["path"],
            "success": success,
        })

        if success:
            manifest["successful"] += 1
        else:
            manifest["failed"] += 1

        console.print()

    # Save manifest
    manifest_path = veripilot_dir / "data" / "sources" / "manifest.json"
    save_manifest(manifest_path, manifest)
    console.print(f"[blue]Saved manifest to {manifest_path}[/blue]")

    # Summary
    console.print()
    console.print("[bold]Download Summary[/bold]")
    console.print(f"  Total: {manifest['total']}")
    console.print(f"  [green]Successful: {manifest['successful']}[/green]")
    console.print(f"  [red]Failed: {manifest['failed']}[/red]")

    return 0 if manifest["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
