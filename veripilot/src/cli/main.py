"""
VeriPilot Interactive CLI.

An interactive command-line interface for the VeriPilot Lean 4 verification copilot.
Provides model selection, file input, and context configuration through interactive prompts.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.llm_client import PROVIDERS, LLMClient, generate_proof, generate_with_aristotle
from agent.user_context import load_user_context
from parser import find_sorries, SorryLocation

app = typer.Typer(
    name="veripilot",
    help="VeriPilot - Lean 4 Verification Copilot",
    add_completion=False,
)
console = Console()


# Model menu options (order matters for display)
MODEL_OPTIONS = [
    ("gemini", "Gemini 3.0 Pro", "Direct Google API - recommended"),
    ("gemini-openrouter", "Gemini 3 Pro (OpenRouter)", "Via OpenRouter API"),
    ("claude", "Claude Sonnet 4.5", "Via Anthropic API"),
    ("claude-opus", "Claude Opus 4.5", "Via Anthropic API - highest quality"),
    ("aristotle", "Aristotle", "Lean specialist - file-based API"),
]


def display_banner():
    """Display the VeriPilot welcome banner."""
    banner = """
[bold cyan]VeriPilot[/bold cyan] - Lean 4 Verification Copilot
[dim]Dual-language Rust verification assistant[/dim]
    """
    console.print(Panel(banner.strip(), border_style="cyan"))


def select_model() -> str:
    """Interactive model selection menu."""
    console.print("\n[bold]Select LLM Provider:[/bold]\n")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Option", style="cyan", width=4)
    table.add_column("Model", style="white")
    table.add_column("Description", style="dim")

    for i, (key, name, desc) in enumerate(MODEL_OPTIONS, 1):
        default_marker = " [green](default)[/green]" if key == "gemini" else ""
        table.add_row(f"[{i}]", f"{name}{default_marker}", desc)

    console.print(table)
    console.print()

    choice = IntPrompt.ask(
        "Enter choice",
        default=1,
        choices=[str(i) for i in range(1, len(MODEL_OPTIONS) + 1)],
    )

    model_key = MODEL_OPTIONS[choice - 1][0]
    model_name = MODEL_OPTIONS[choice - 1][1]
    console.print(f"\n[green]Selected:[/green] {model_name}\n")
    return model_key


def get_lean_file_path() -> str:
    """Prompt for and validate Lean file path."""
    while True:
        path = Prompt.ask("\n[bold]Enter path to Lean file[/bold]")
        path = path.strip().strip("'\"")  # Remove quotes if present

        # Expand user path
        expanded = Path(path).expanduser()

        if not expanded.exists():
            console.print(f"[red]File not found:[/red] {path}")
            continue

        if not expanded.suffix == ".lean":
            console.print(f"[yellow]Warning:[/yellow] File does not have .lean extension")
            if not Confirm.ask("Continue anyway?", default=False):
                continue

        return str(expanded.resolve())


def select_sorry_lines(sorries: list[SorryLocation]) -> list[SorryLocation]:
    """
    Interactive menu to select which sorries to solve.

    Args:
        sorries: List of all sorry locations found in file

    Returns:
        Filtered list of sorries to process
    """
    if not sorries:
        return []

    console.print(f"\n[bold]Found {len(sorries)} sorry location(s):[/bold]\n")

    # Display table of all sorries
    table = Table(show_header=True, header_style="bold")
    table.add_column("Index", style="cyan", width=6)
    table.add_column("Line", style="yellow", width=6)
    table.add_column("Theorem", style="white")
    table.add_column("Preview", style="dim")

    for i, sorry in enumerate(sorries, 1):
        preview = sorry.theorem_signature[:40] if sorry.theorem_signature else "(unknown signature)"
        if len(preview) == 40:
            preview += "..."
        table.add_row(
            f"[{i}]",
            str(sorry.line),
            sorry.theorem_name or "(unnamed)",
            preview,
        )

    console.print(table)
    console.print()

    # Selection menu
    console.print("[bold]Select which sorries to solve:[/bold]")
    console.print("  [cyan][A][/cyan] All sorries (default)")
    console.print("  [cyan][S][/cyan] Select specific indices (e.g., 1,3,5)")
    console.print("  [cyan][R][/cyan] Select by line range (e.g., 21-56)")
    console.print()

    choice = Prompt.ask("Enter choice", default="A").strip().upper()

    if choice == "A" or not choice:
        console.print(f"[green]Selected:[/green] All {len(sorries)} sorries\n")
        return sorries

    elif choice == "S":
        console.print("\n[dim]Enter indices separated by commas (e.g., 1,3,5)[/dim]")
        indices_str = Prompt.ask("Indices")

        try:
            indices = [int(x.strip()) for x in indices_str.split(",")]
            selected = [sorries[i - 1] for i in indices if 1 <= i <= len(sorries)]

            if not selected:
                console.print("[yellow]No valid indices - using all sorries[/yellow]")
                return sorries

            console.print(f"[green]Selected:[/green] {len(selected)} sorry location(s)\n")
            return selected

        except (ValueError, IndexError) as e:
            console.print(f"[red]Invalid input:[/red] {e}")
            console.print("[yellow]Using all sorries[/yellow]\n")
            return sorries

    elif choice == "R":
        console.print("\n[dim]Enter line range (e.g., 21-56 to solve sorries on lines 21 through 56)[/dim]")
        range_str = Prompt.ask("Line range")

        try:
            start, end = range_str.split("-")
            start_line = int(start.strip())
            end_line = int(end.strip())

            # Filter sorries within the line range
            selected = [s for s in sorries if start_line <= s.line <= end_line]

            if not selected:
                console.print(f"[yellow]No sorries found in line range {start_line}-{end_line}[/yellow]")
                console.print("[yellow]Using all sorries[/yellow]\n")
                return sorries

            console.print(f"[green]Selected:[/green] {len(selected)} sorry location(s) in lines {start_line}-{end_line}\n")
            for s in selected:
                console.print(f"  [dim]Line {s.line}: {s.theorem_name or '(unnamed)'}[/dim]")
            console.print()
            return selected

        except (ValueError, IndexError) as e:
            console.print(f"[red]Invalid range:[/red] {e}")
            console.print("[yellow]Using all sorries[/yellow]\n")
            return sorries

    else:
        console.print(f"[yellow]Unknown choice '{choice}' - using all sorries[/yellow]\n")
        return sorries


def select_mode() -> tuple[int, Optional[str], Optional[str]]:
    """
    Select verification mode.

    Returns:
        (mode_number, context_path, custom_prompt)
        - mode 1: (1, None, None) - direct fill
        - mode 2: (2, context_path, None) - with context file
        - mode 3: (3, None, custom_prompt) - with custom prompt
    """
    console.print("\n[bold]Select mode:[/bold]\n")
    console.print("  [cyan][1][/cyan] Fill all sorries (direct)")
    console.print("  [cyan][2][/cyan] Add context file")
    console.print("  [cyan][3][/cyan] Custom prompt")
    console.print()

    mode = IntPrompt.ask("Enter choice", default=1, choices=["1", "2", "3"])

    if mode == 2:
        context_path = Prompt.ask("\n[bold]Enter context file path[/bold] (MD or TXT)")
        context_path = context_path.strip().strip("'\"")
        expanded = Path(context_path).expanduser()

        if not expanded.exists():
            console.print(f"[yellow]Warning:[/yellow] Context file not found: {context_path}")
            if not Confirm.ask("Continue without context?", default=True):
                return select_mode()  # Retry
            return (1, None, None)

        return (2, str(expanded.resolve()), None)

    elif mode == 3:
        console.print("\n[dim]Enter custom prompt (press Enter twice to finish):[/dim]")
        lines = []
        while True:
            line = Prompt.ask("", default="")
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)

        custom_prompt = "\n".join(lines).strip()
        if not custom_prompt:
            console.print("[yellow]Empty prompt - using direct mode[/yellow]")
            return (1, None, None)

        return (3, None, custom_prompt)

    return (1, None, None)


async def run_verification(
    file_path: str,
    model: str,
    context_path: Optional[str] = None,
    custom_prompt: Optional[str] = None,
):
    """Run the verification process."""
    console.print(f"\n[bold cyan]Verifying:[/bold cyan] {Path(file_path).name}")
    console.print(f"[bold cyan]Model:[/bold cyan] {PROVIDERS[model].name}")

    # Load user context if provided
    user_context = None
    if context_path:
        console.print(f"[bold cyan]Context:[/bold cyan] {Path(context_path).name}")
        user_context = load_user_context(context_path)
        if user_context:
            console.print(f"  [dim]Loaded {len(user_context)} chars of context[/dim]")

    if custom_prompt:
        console.print(f"[bold cyan]Custom prompt:[/bold cyan] {custom_prompt[:50]}...")
        user_context = custom_prompt

    # Parse the Lean file
    console.print("\n[dim]Parsing Lean file...[/dim]")
    try:
        all_sorries = find_sorries(file_path)
    except Exception as e:
        console.print(f"[red]Parse error:[/red] {e}")
        return

    if not all_sorries:
        console.print("[green]No sorries found in file![/green]")
        return

    # Let user select which sorries to solve
    sorries = select_sorry_lines(all_sorries)

    if not sorries:
        console.print("[yellow]No sorries selected - nothing to do.[/yellow]")
        return

    # Read file content
    with open(file_path) as f:
        file_content = f.read()

    # Process each sorry
    for i, sorry in enumerate(sorries, 1):
        console.print(f"\n[bold]Processing sorry {i}/{len(sorries)}:[/bold]")
        console.print(f"  [dim]Theorem:[/dim] {sorry.theorem_name}")
        console.print(f"  [dim]Line:[/dim] {sorry.line}")

        with console.status("[bold green]Generating proof..."):
            try:
                result = await generate_proof(
                    sorry=sorry,
                    file_content=file_content,
                    model=model,
                )

                if result.success and result.proof_code:
                    console.print(f"  [green]Success![/green]")
                    console.print(f"  [dim]Proof:[/dim]")
                    for line in result.proof_code.split("\n")[:5]:  # First 5 lines
                        console.print(f"    {line}")
                    if result.proof_code.count("\n") > 5:
                        console.print(f"    [dim]... ({result.proof_code.count(chr(10)) - 5} more lines)[/dim]")
                else:
                    console.print(f"  [red]Failed:[/red] {result.error or 'Unknown error'}")

            except Exception as e:
                console.print(f"  [red]Error:[/red] {e}")

    console.print("\n[bold green]Verification complete![/bold green]")


@app.command()
def main():
    """
    VeriPilot Interactive CLI.

    Run without arguments for interactive mode with menus.
    """
    display_banner()

    # Interactive prompts
    model = select_model()
    file_path = get_lean_file_path()
    mode, context_path, custom_prompt = select_mode()

    # Confirmation
    console.print("\n" + "─" * 40)
    console.print("[bold]Ready to verify:[/bold]")
    console.print(f"  File: {file_path}")
    console.print(f"  Model: {PROVIDERS[model].name}")
    if context_path:
        console.print(f"  Context: {context_path}")
    if custom_prompt:
        console.print(f"  Prompt: {custom_prompt[:30]}...")
    console.print("─" * 40 + "\n")

    if not Confirm.ask("Proceed?", default=True):
        console.print("[yellow]Cancelled.[/yellow]")
        raise typer.Exit()

    # Run verification
    try:
        asyncio.run(run_verification(
            file_path=file_path,
            model=model,
            context_path=context_path,
            custom_prompt=custom_prompt,
        ))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def version():
    """Show VeriPilot version."""
    console.print("[cyan]VeriPilot[/cyan] v0.1.0")


@app.command()
def models():
    """List available LLM providers."""
    display_banner()
    console.print("\n[bold]Available LLM Providers:[/bold]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Key", style="cyan")
    table.add_column("Name")
    table.add_column("Model ID", style="dim")
    table.add_column("API Key Env Var", style="yellow")

    for key, config in PROVIDERS.items():
        table.add_row(key, config.name, config.model, config.env_key)

    console.print(table)


if __name__ == "__main__":
    app()
