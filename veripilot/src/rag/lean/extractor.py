"""
LeanDojo-based extractor for Lean 4 repositories.

This module provides compiler-level extraction of theorems, proofs,
tactics, and proof states from Lean 4 codebases using LeanDojo.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


@dataclass
class TacticStep:
    """Represents a single tactic step in a proof."""
    index: int
    tactic: str
    state_before: str
    state_after: str
    success: bool = True


@dataclass
class ExtractedTheorem:
    """Represents a fully extracted theorem/lemma."""
    name: str
    full_name: str
    type_signature: str
    proof: str
    file_path: str
    line_start: int
    line_end: int
    premises: list[str] = field(default_factory=list)
    tactics: list[TacticStep] = field(default_factory=list)
    doc_string: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "full_name": self.full_name,
            "type_signature": self.type_signature,
            "proof": self.proof,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "premises": self.premises,
            "tactics": [asdict(t) for t in self.tactics],
            "doc_string": self.doc_string,
        }


@dataclass
class ExtractedFile:
    """Represents a parsed Lean file."""
    path: str
    theorems: list[ExtractedTheorem]
    imports: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Complete extraction result for a corpus."""
    corpus_name: str
    corpus_path: str
    commit: str
    files: list[ExtractedFile]
    total_theorems: int
    total_tactics: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "corpus_name": self.corpus_name,
            "corpus_path": self.corpus_path,
            "commit": self.commit,
            "total_theorems": self.total_theorems,
            "total_tactics": self.total_tactics,
            "files": [
                {
                    "path": f.path,
                    "imports": f.imports,
                    "theorems": [t.to_dict() for t in f.theorems],
                }
                for f in self.files
            ],
        }

    def save(self, output_path: Path) -> None:
        """Save extraction result to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        console.print(f"[green]Saved extraction to {output_path}[/green]")

    @classmethod
    def load(cls, input_path: Path) -> "ExtractionResult":
        """Load extraction result from JSON file."""
        with open(input_path) as f:
            data = json.load(f)

        files = []
        for file_data in data["files"]:
            theorems = []
            for thm_data in file_data["theorems"]:
                tactics = [
                    TacticStep(**t) for t in thm_data.get("tactics", [])
                ]
                theorems.append(ExtractedTheorem(
                    name=thm_data["name"],
                    full_name=thm_data["full_name"],
                    type_signature=thm_data["type_signature"],
                    proof=thm_data["proof"],
                    file_path=thm_data["file_path"],
                    line_start=thm_data["line_start"],
                    line_end=thm_data["line_end"],
                    premises=thm_data.get("premises", []),
                    tactics=tactics,
                    doc_string=thm_data.get("doc_string"),
                ))
            files.append(ExtractedFile(
                path=file_data["path"],
                theorems=theorems,
                imports=file_data.get("imports", []),
            ))

        return cls(
            corpus_name=data["corpus_name"],
            corpus_path=data["corpus_path"],
            commit=data["commit"],
            files=files,
            total_theorems=data["total_theorems"],
            total_tactics=data["total_tactics"],
        )


class LeanDojoExtractor:
    """
    Extracts theorem data from Lean 4 repositories using LeanDojo.

    LeanDojo hooks into Lean's compiler to provide:
    - Perfect AST extraction
    - Proof states at each tactic step
    - Premise relationships (which lemmas each theorem uses)
    - Accurate type signatures
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        num_procs: int = 4,
    ):
        """
        Initialize the extractor.

        Args:
            cache_dir: Directory for LeanDojo cache. Defaults to LEAN_DOJO_CACHE_DIR
                      or ~/.cache/lean_dojo
            num_procs: Number of parallel processes for extraction
        """
        self.cache_dir = cache_dir or os.getenv(
            "LEAN_DOJO_CACHE_DIR",
            os.path.expanduser("~/.cache/lean_dojo")
        )
        self.num_procs = num_procs

        # Set environment variables for LeanDojo
        os.environ["CACHE_DIR"] = self.cache_dir
        os.environ["NUM_PROCS"] = str(self.num_procs)

        console.print(f"[blue]LeanDojo cache: {self.cache_dir}[/blue]")
        console.print(f"[blue]Parallel processes: {self.num_procs}[/blue]")

    def extract_local_repo(
        self,
        repo_path: str | Path,
        corpus_name: str = "local",
    ) -> ExtractionResult:
        """
        Extract theorems from a local Lean repository.

        Args:
            repo_path: Path to the Lean repository
            corpus_name: Name for the corpus (used in output)

        Returns:
            ExtractionResult containing all extracted data
        """
        repo_path = Path(repo_path).resolve()

        if not repo_path.exists():
            raise FileNotFoundError(f"Repository not found: {repo_path}")

        console.print(f"[bold]Extracting from: {repo_path}[/bold]")

        try:
            from lean_dojo import LeanGitRepo, trace
        except ImportError:
            console.print("[red]LeanDojo not installed. Install with: pip install lean-dojo[/red]")
            raise

        # Get commit hash
        commit = self._get_git_commit(repo_path)
        console.print(f"[blue]Commit: {commit}[/blue]")

        # Create LeanDojo repo reference
        # For local repos, we use the path directly
        repo = LeanGitRepo(str(repo_path), commit)

        # Run the tracer
        console.print("[yellow]Running LeanDojo tracer (this may take a while)...[/yellow]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Tracing repository...", total=None)
            traced_repo = trace(repo)
            progress.update(task, completed=True, description="Tracing complete")

        # Extract data from traced repo
        return self._process_traced_repo(
            traced_repo,
            corpus_name=corpus_name,
            corpus_path=str(repo_path),
            commit=commit,
        )

    def extract_github_repo(
        self,
        url: str,
        commit: str = "HEAD",
        corpus_name: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract theorems from a GitHub repository.

        Args:
            url: GitHub repository URL
            commit: Commit hash, tag, or branch name
            corpus_name: Name for the corpus (defaults to repo name)

        Returns:
            ExtractionResult containing all extracted data
        """
        try:
            from lean_dojo import LeanGitRepo, trace
        except ImportError:
            console.print("[red]LeanDojo not installed. Install with: pip install lean-dojo[/red]")
            raise

        if corpus_name is None:
            corpus_name = url.rstrip("/").split("/")[-1]

        console.print(f"[bold]Extracting from GitHub: {url}[/bold]")
        console.print(f"[blue]Commit: {commit}[/blue]")

        repo = LeanGitRepo(url, commit)

        console.print("[yellow]Running LeanDojo tracer (this may take a while)...[/yellow]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Tracing repository...", total=None)
            traced_repo = trace(repo)
            progress.update(task, completed=True, description="Tracing complete")

        return self._process_traced_repo(
            traced_repo,
            corpus_name=corpus_name,
            corpus_path=url,
            commit=commit,
        )

    def _process_traced_repo(
        self,
        traced_repo,
        corpus_name: str,
        corpus_path: str,
        commit: str,
    ) -> ExtractionResult:
        """Process a traced repository and extract all theorem data."""
        files = []
        total_theorems = 0
        total_tactics = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            traced_files = list(traced_repo.traced_files)
            task = progress.add_task(
                "Extracting theorems...",
                total=len(traced_files)
            )

            for traced_file in traced_files:
                theorems = []

                for theorem in traced_file.theorems:
                    # Extract tactic steps
                    tactics = []
                    for i, tactic in enumerate(theorem.tactics):
                        tactics.append(TacticStep(
                            index=i,
                            tactic=tactic.tactic_code,
                            state_before=tactic.state_before,
                            state_after=tactic.state_after,
                            success=getattr(tactic, "is_success", True),
                        ))
                        total_tactics += 1

                    # Create extracted theorem
                    extracted = ExtractedTheorem(
                        name=theorem.name,
                        full_name=theorem.full_name,
                        type_signature=theorem.type,
                        proof=theorem.proof,
                        file_path=traced_file.path,
                        line_start=theorem.start_pos[0] if theorem.start_pos else 0,
                        line_end=theorem.end_pos[0] if theorem.end_pos else 0,
                        premises=list(theorem.premises) if hasattr(theorem, "premises") else [],
                        tactics=tactics,
                        doc_string=getattr(theorem, "doc_string", None),
                    )
                    theorems.append(extracted)
                    total_theorems += 1

                files.append(ExtractedFile(
                    path=traced_file.path,
                    theorems=theorems,
                    imports=[],  # TODO: extract imports if available
                ))

                progress.update(task, advance=1)

        console.print(f"[green]Extracted {total_theorems} theorems, {total_tactics} tactic steps[/green]")

        return ExtractionResult(
            corpus_name=corpus_name,
            corpus_path=corpus_path,
            commit=commit,
            files=files,
            total_theorems=total_theorems,
            total_tactics=total_tactics,
        )

    def _get_git_commit(self, repo_path: Path) -> str:
        """Get the current git commit hash."""
        import subprocess
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()[:12]
        except subprocess.CalledProcessError:
            return "unknown"


def iterate_theorems(result: ExtractionResult) -> Iterator[ExtractedTheorem]:
    """Iterate over all theorems in an extraction result."""
    for file in result.files:
        yield from file.theorems


def iterate_tactics(result: ExtractionResult) -> Iterator[tuple[ExtractedTheorem, TacticStep]]:
    """Iterate over all tactic steps, yielding (theorem, tactic) pairs."""
    for theorem in iterate_theorems(result):
        for tactic in theorem.tactics:
            yield theorem, tactic
