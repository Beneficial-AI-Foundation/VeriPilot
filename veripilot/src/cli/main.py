"""
VeriPilot Interactive CLI.

An interactive command-line interface for the VeriPilot Lean 4 verification copilot.
Provides model selection, file input, and context configuration through interactive prompts.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table
from dotenv import load_dotenv

# Version info
__version__ = "0.1.0"

# Load .env file for API keys
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.llm_client import PROVIDERS, LLMClient, generate_proof
from agent.user_context import load_user_context
from agent.react import AgentMode, get_available_modes, ReActAgent
from parser import find_sorries, SorryLocation
from verifier import verify_proof, verify_proof_lsp, VerifierService
from rag.lean.llamaindex_lean import LeanRAG

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
]

# Default max verification attempts
MAX_ATTEMPTS = 4


def find_lean_project_root(file_path: str) -> Optional[str]:
    """
    Find the Lean project root by walking up from file to find lakefile.lean or lakefile.toml.

    Args:
        file_path: Path to a Lean file

    Returns:
        Path to project root directory, or None if not found
    """
    path = Path(file_path).resolve()
    for parent in [path.parent] + list(path.parents):
        if (parent / "lakefile.lean").exists() or (parent / "lakefile.toml").exists():
            return str(parent)
    return None


def _supports_color() -> bool:
    """Check if terminal supports ANSI colors."""
    # Check NO_COLOR env var (https://no-color.org/)
    if os.environ.get("NO_COLOR"):
        return False
    # Check if stdout is a TTY
    if not sys.stdout.isatty():
        return False
    # Check for dumb terminal
    if os.environ.get("TERM") == "dumb":
        return False
    return True


def display_banner():
    """Display the VeriPilot ASCII art banner."""
    # Color codes (ANSI 256)
    if _supports_color():
        C_VERI = "\033[38;5;69m"    # Royal Blue - VERI prefix
        C_PILOT = "\033[38;5;117m"  # Sky Blue - PILOT suffix
        C_GREY = "\033[38;5;240m"   # Dark grey - decorative elements
        C_RESET = "\033[0m"
        C_BOLD = "\033[1m"
    else:
        C_VERI = C_PILOT = C_GREY = C_RESET = C_BOLD = ""

    # ASCII art banner - fits within 80 columns
    banner = f"""
{C_VERI}██╗   ██╗███████╗██████╗ ██╗{C_PILOT}██████╗ ██╗██╗     ╔█████╗ ████████╗
{C_VERI}██║   ██║██╔════╝██╔══██╗██║{C_PILOT}██╔══██╗██║██║     ██╔══██╗╚══██╔══╝
{C_VERI}██║   ██║█████╗  ██████╔╝██║{C_PILOT}██████╔╝██║██║     ██║  ██║   ██║
{C_VERI}╚██╗ ██╔╝██╔══╝  ██╔══██╗██║{C_PILOT}██╔═══╝ ██║██║     ██║  ██║   ██║
{C_VERI} ╚████╔╝ ███████╗██║  ██║██║{C_PILOT}██║     ██║███████╗╚█████╔╝   ██║
{C_VERI}  ╚═══╝  ╚══════╝╚═╝  ╚═╝╚═╝{C_PILOT}╚═╝     ╚═╝╚══════╝╚═════╝    ╚═╝   {C_RESET}
      {C_GREY}::{C_RESET} {C_BOLD}Formal Verification Copilot{C_RESET} {C_GREY}::{C_RESET} v{__version__}
"""
    print(banner)


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


def select_temperature() -> float:
    """Interactive temperature selection menu."""
    console.print("[bold]Select Temperature:[/bold]\n")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Option", style="cyan", width=4)
    table.add_column("Setting", style="white")
    table.add_column("Description", style="dim")

    options = [
        ("0.2", "Low", "More deterministic, conservative proofs"),
        ("0.4", "Medium", "Balanced creativity and consistency"),
        ("0.7", "High", "More exploration, diverse attempts"),
        ("custom", "Custom", "Enter value between 0.0-1.0"),
    ]

    for i, (temp, name, desc) in enumerate(options, 1):
        default_marker = " [green](recommended)[/green]" if temp == "0.2" else ""
        table.add_row(f"[{i}]", f"{name} ({temp}){default_marker}", desc)

    console.print(table)
    console.print()

    choice = IntPrompt.ask(
        "Enter choice",
        default=1,
        choices=["1", "2", "3", "4"],
    )

    if choice == 4:
        temp_str = Prompt.ask(
            "Enter temperature (0.0-1.0)",
            default="0.2"
        )
        try:
            temp_float = float(temp_str)
            if 0.0 <= temp_float <= 1.0:
                console.print(f"\n[green]Selected:[/green] {temp_float}\n")
                return temp_float
            else:
                console.print("[yellow]Invalid range, using 0.2[/yellow]")
                return 0.2
        except ValueError:
            console.print("[yellow]Invalid input, using 0.2[/yellow]")
            return 0.2

    temp_map = {1: 0.2, 2: 0.4, 3: 0.7}
    selected = temp_map[choice]
    console.print(f"\n[green]Selected:[/green] {selected}\n")
    return selected


def select_verification_mode() -> AgentMode:
    """Interactive verification mode selection menu."""
    console.print("[bold]Select Verification Mode:[/bold]\n")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Option", style="cyan", width=4)
    table.add_column("Mode", style="white")
    table.add_column("Description", style="dim")

    modes = get_available_modes()
    for i, (mode_value, display_name, desc) in enumerate(modes, 1):
        phase_marker = ""
        if "Phase 2" in desc:
            phase_marker = " [yellow](coming soon)[/yellow]"
        elif "Phase 3" in desc:
            phase_marker = " [yellow](coming soon)[/yellow]"
        default_marker = " [green](default)[/green]" if mode_value == "just_retry" else ""
        table.add_row(f"[{i}]", f"{display_name}{default_marker}{phase_marker}", desc.split("[")[0].strip())

    console.print(table)
    console.print()

    choice = IntPrompt.ask(
        "Enter choice",
        default=1,
        choices=[str(i) for i in range(1, len(modes) + 1)],
    )

    mode_value = modes[choice - 1][0]
    mode_name = modes[choice - 1][1]
    console.print(f"\n[green]Selected:[/green] {mode_name}\n")
    return AgentMode(mode_value)


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
    temperature: float = 0.2,
    context_path: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    use_lsp: bool = True,
    agent_mode: AgentMode = AgentMode.JUST_RETRY,
):
    """
    Run the verification process.

    Args:
        file_path: Path to Lean file
        model: LLM model to use
        temperature: Temperature for proof generation
        context_path: Optional context file path
        custom_prompt: Optional custom prompt
        use_lsp: Use LSP for instant verification (default True)
        agent_mode: Verification agent mode (JUST_RETRY, REACT, etc.)
    """
    console.print(f"\n[bold cyan]Verifying:[/bold cyan] {Path(file_path).name}")
    console.print(f"[bold cyan]Model:[/bold cyan] {PROVIDERS[model].name}")
    console.print(f"[bold cyan]Temperature:[/bold cyan] {temperature}")
    console.print(f"[bold cyan]Agent Mode:[/bold cyan] {agent_mode.value}")

    # Find project root
    project_dir = find_lean_project_root(file_path)
    if not project_dir:
        console.print("[red]Error:[/red] Could not find Lean project root (lakefile.lean or lakefile.toml)")
        console.print("[dim]Make sure the file is inside a Lean project directory[/dim]")
        return
    console.print(f"[bold cyan]Project:[/bold cyan] {Path(project_dir).name}")

    # Initialize VerifierService with MCP warm-up in background
    verifier_service = None
    if use_lsp:
        console.print("[dim]Starting LSP verifier (warming up MCP)...[/dim]")
        verifier_service = VerifierService(project_dir)
        await verifier_service.start(wait_for_warmup=True)  # Wait for MCP to be ready
        if verifier_service.status.mcp_available:
            console.print("[bold cyan]Verifier:[/bold cyan] LSP (MCP lean-lsp) ready")
        else:
            console.print("[yellow]Verifier:[/yellow] MCP failed, using lake build fallback")
    else:
        console.print("[bold cyan]Verifier:[/bold cyan] lake build")

    # Initialize RAG for context retrieval
    rag = None
    try:
        console.print("[dim]Initializing RAG backends...[/dim]")
        rag = LeanRAG()
        await rag.initialize()
        console.print("[bold cyan]RAG:[/bold cyan] Initialized")
    except Exception as e:
        console.print(f"[yellow]RAG unavailable:[/yellow] {e}")
        console.print("[dim]Continuing without RAG context...[/dim]")

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
        if verifier_service:
            await verifier_service.stop()
        return

    if not all_sorries:
        console.print("[green]No sorries found in file![/green]")
        if verifier_service:
            await verifier_service.stop()
        return

    # Let user select which sorries to solve
    sorries = select_sorry_lines(all_sorries)

    if not sorries:
        console.print("[yellow]No sorries selected - nothing to do.[/yellow]")
        if verifier_service:
            await verifier_service.stop()
        return

    # Read file content for proof generation
    with open(file_path) as f:
        file_content = f.read()

    # Track results
    verified_proofs = []
    failed_proofs = []

    try:
        # Process each sorry
        for i, sorry in enumerate(sorries, 1):
            console.print(f"\n[bold]Processing sorry {i}/{len(sorries)}:[/bold]")
            console.print(f"  [dim]Theorem:[/dim] {sorry.theorem_name}")
            console.print(f"  [dim]Line:[/dim] {sorry.line}")

            try:
                # Step 1: Generate initial proof
                console.print("  [dim]Generating initial proof...[/dim]")
                initial_result = await generate_proof(
                    sorry=sorry,
                    file_content=file_content,
                    rag=rag,
                    model=model,
                    temperature=temperature,
                )

                if not initial_result.success or not initial_result.proof_code:
                    console.print(f"  [red]Failed to generate proof:[/red] {initial_result.error or 'Unknown error'}")
                    failed_proofs.append((sorry.theorem_name, "Generation failed"))
                    continue

                # Step 2: Verify with retry loop or ReAct agent
                if agent_mode != AgentMode.JUST_RETRY and verifier_service and use_lsp:
                    # Use ReAct agent for reasoning-based verification
                    console.print(f"  [dim]Verifying with {agent_mode.value} agent...[/dim]")
                    react_agent = ReActAgent(
                        mode=agent_mode,
                        max_attempts=MAX_ATTEMPTS,
                        model=model,
                        temperature=temperature,
                        project_dir=project_dir,
                    )
                    react_result = await react_agent.verify(
                        sorry=sorry,
                        initial_proof=initial_result.proof_code,
                        file_content=file_content,
                        verifier_service=verifier_service,
                        rag=rag,
                        project_dir=project_dir,
                    )
                    # Convert to VerificationResult for consistent handling
                    verification = react_result.to_verification_result()
                elif verifier_service and use_lsp:
                    # Use LSP verification (instant, never modifies original)
                    mcp_status = "warm" if verifier_service.status.mcp_available else "warming up"
                    console.print(f"  [dim]Verifying via LSP ({mcp_status})...[/dim]")
                    verification = await verify_proof_lsp(
                        sorry=sorry,
                        proof_result=initial_result,
                        verifier_service=verifier_service,
                        rag=rag,
                        max_attempts=MAX_ATTEMPTS,
                    )
                else:
                    # Fall back to lake build verification
                    console.print("  [dim]Verifying with lake build...[/dim]")
                    verification = await verify_proof(
                        sorry=sorry,
                        proof_result=initial_result,
                        rag=rag,
                        project_dir=project_dir,
                        max_attempts=MAX_ATTEMPTS,
                    )

                # Show result
                if verification.success:
                    method = verification.build_output.split()[2] if "via" in verification.build_output else "lake"
                    console.print(f"  [green]✓ Verified on attempt {verification.attempts}![/green]")
                    console.print(f"  [dim]Proof:[/dim]")
                    for line in verification.proof_code.split("\n")[:5]:
                        console.print(f"    {line}")
                    if verification.proof_code.count("\n") > 5:
                        console.print(f"    [dim]... ({verification.proof_code.count(chr(10)) - 5} more lines)[/dim]")

                    if verification.output_file:
                        console.print(f"  [green]Output:[/green] {Path(verification.output_file).name}")
                    verified_proofs.append((sorry.theorem_name, verification.attempts, verification.output_file))
                else:
                    console.print(f"  [red]✗ Failed after {verification.attempts} attempt(s)[/red]")
                    if verification.errors:
                        console.print(f"  [dim]Last error:[/dim] {verification.errors[-1][:100]}")
                    if verification.output_file:
                        console.print(f"  [yellow]Final attempt:[/yellow] {Path(verification.output_file).name}")
                    failed_proofs.append((sorry.theorem_name, f"Failed after {verification.attempts} attempts"))

                # Show log file location
                if verification.log_file:
                    console.print(f"  [dim]Log:[/dim] {Path(verification.log_file).name}")

            except ValueError as e:
                # API key or configuration errors - fatal, exit immediately
                error_msg = str(e)
                if "API_KEY" in error_msg:
                    console.print(f"\n[red bold]Configuration Error:[/red bold] {error_msg}")
                    console.print("[yellow]Check your .env file and ensure the API key is set.[/yellow]")
                    raise typer.Exit(1)
                else:
                    console.print(f"  [red]Error:[/red] {e}")
                    failed_proofs.append((sorry.theorem_name, str(e)))
            except Exception as e:
                console.print(f"  [red]Error:[/red] {e}")
                failed_proofs.append((sorry.theorem_name, str(e)))

    finally:
        # Clean up verifier service
        if verifier_service:
            await verifier_service.stop()

    # Summary
    console.print("\n" + "─" * 50)
    console.print("[bold]Process Complete[/bold]")
    if verified_proofs:
        console.print(f"[green]✓ {len(verified_proofs)} proof(s) verified successfully[/green]")
        for name, attempts, output_file in verified_proofs:
            output_name = Path(output_file).name if output_file else "?"
            console.print(f"  [dim]{name}:[/dim] {output_name} (attempt {attempts})")
    if failed_proofs:
        console.print(f"[red]✗ {len(failed_proofs)} proof(s) failed[/red]")
        for name, reason in failed_proofs:
            console.print(f"  [dim]{name}:[/dim] {reason}")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """
    VeriPilot Interactive CLI.

    Run without arguments for interactive mode with menus.
    Use subcommands like 'version' or 'models' for specific info.
    """
    # If a subcommand was invoked, let it handle execution
    if ctx.invoked_subcommand is not None:
        return

    display_banner()

    # Interactive prompts
    model = select_model()
    temperature = select_temperature()
    agent_mode = select_verification_mode()
    file_path = get_lean_file_path()
    mode, context_path, custom_prompt = select_mode()

    # Confirmation
    console.print("\n" + "─" * 40)
    console.print("[bold]Ready to verify:[/bold]")
    console.print(f"  File: {file_path}")
    console.print(f"  Model: {PROVIDERS[model].name}")
    console.print(f"  Temperature: {temperature}")
    console.print(f"  Agent: {agent_mode.value}")
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
            temperature=temperature,
            context_path=context_path,
            custom_prompt=custom_prompt,
            agent_mode=agent_mode,
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
    if _supports_color():
        C_VERI = "\033[38;5;69m"
        C_PILOT = "\033[38;5;117m"
        C_RESET = "\033[0m"
        print(f"{C_VERI}Veri{C_PILOT}Pilot{C_RESET} v{__version__}")
    else:
        print(f"VeriPilot v{__version__}")


@app.command()
def models():
    """List available LLM providers."""
    # Show version line (not full banner)
    if _supports_color():
        C_VERI = "\033[38;5;69m"
        C_PILOT = "\033[38;5;117m"
        C_RESET = "\033[0m"
        print(f"\n{C_VERI}Veri{C_PILOT}Pilot{C_RESET} v{__version__} - Available LLM Providers\n")
    else:
        print(f"\nVeriPilot v{__version__} - Available LLM Providers\n")

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
