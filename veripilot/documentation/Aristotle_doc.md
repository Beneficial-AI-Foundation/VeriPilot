# Aristotle - The Era of Vibe Proving

## Project Description

The Aristotle SDK is a Terminal Interface and Python library that provides tools and utilities for interacting with the Aristotle API. Aristotle is capable of proving and formally verifying graduate and research level problems in math, software, and more.

**Sign up for access**: [aristotle.harmonic.fun](https://aristotle.harmonic.fun)

---

## Installation

```bash
pip install aristotlelib
```

Or if you have an older version:

```bash
pip install --upgrade aristotlelib
```

---

## Setting up your API Key

To avoid entering your API key each time, add it to your terminal configuration file:

```bash
export ARISTOTLE_API_KEY="your-api-key-here"
```

- **For zsh users** (macOS default): Add to `~/.zshrc`
- **For bash users**: Add to `~/.bashrc` or `~/.bash_profile`

After adding the line, restart your terminal.

---

## Quickest Start: CLI

### 1. Set your API Key

```bash
export ARISTOTLE_API_KEY="your-api-key-here"
```

### 2. Run aristotle

Type `aristotle` in your terminal to get started!

The interface will guide you through:
- Submitting Lean files to have sorries filled
- Autoformalizing mathematical content from English (including LaTeX, markdown, and text)
- Prompting Aristotle directly
- Viewing your previous submissions

---

## Lean Toolchain and Mathlib Versions

Aristotle uses the following versions:

- **Lean Toolchain**: `leanprover/lean4:v4.24.0`
- **Mathlib version**: `v4.24.0` - Oct 14, 2025 (`f897ebcf72cd16f89ab4577d0c826cd14afaafc7`)

⚠️ **Note**: If your project uses different versions, you might encounter compatibility issues.

---

## Aristotle's Modes - Terminal UI

### 1. Fill Sorries in a Lean file

Directly integrates into your existing Lean projects. Simply submit the path to the file, and Aristotle fills in all sorries. It automatically imports necessary files from your Lean project.

### 2. Upload a paper

Upload mode enables you to work with mathematical problems described entirely in natural language. Provide a file containing your mathematical question, theorem statements, or problem description in plain English, LaTeX, or markdown, and Aristotle will formalize it into Lean and generate a complete proof.

You can include context:
- Attaching a Lean file will include the Lean file and any of its imports as context
- Upload a file or folder of English, markdown, or LaTeX

### 3. Type in your prompt

Similar to upload mode, but lets you type your prompt directly instead of requiring a file.

You can include the same context options as upload mode.

### 4. View history

Displays all your previous Aristotle submissions in a table format. Review past projects, check status, see completion times, and quickly access previous solutions.

---

## Command Line Interface

```bash
# Set your API key
export ARISTOTLE_API_KEY="your-api-key-here"

# Prove theorems from a file (formal Lean)
aristotle prove-from-file path/to/theorem.lean

# Prove from informal input (natural language)
aristotle prove-from-file path/to/problem.txt --informal

# Provide formal context for informal problems
aristotle prove-from-file path/to/problem.txt --informal --formal-input-context path/to/context.lean

# Specify output file
aristotle prove-from-file path/to/theorem.lean --output-file solution.lean

# See all options
aristotle prove-from-file --help
```

---

## Lean Mode Features

### Guide Aristotle in English

You can provide natural language hints to guide Aristotle's proof search. Include your English proof sketch in the header comment, tagged with `PROVIDED SOLUTION:`.

```lean
/--
  Given x, y ∈ [0, π/2], show that cos(sqrt(x ^ 2 + y ^ 2)) ≤ cos x * cos y.

  PROVIDED SOLUTION:
  Set r := sqrt(x^2 + y^2). If r > π/2, then the inequality holds trivially.
  So consider the case r ≤ π/2. Write x = r cos φ, y = r sin φ.
  [... detailed proof sketch ...]
-/
theorem final (x y : ℝ) (hx : 0 ≤ x) (hx' : x ≤ Real.pi / 2)
    (hy : 0 ≤ y) (hy' : y ≤ Real.pi / 2) :
    Real.cos (Real.sqrt (x ^ 2 + y ^ 2)) ≤ Real.cos x * Real.cos y := by
  sorry
```

### Find Counterexamples and Negations Automatically

Aristotle can disprove statements and find counterexamples. When a statement is false, Aristotle leaves a comment with the counterexample.

```lean
/-
Aristotle found this block to be false.
Here is a proof of the negation:
theorem my_favorite_theorem (k : ℕ) :
  ∑' n : ℕ, (1 : ℝ) / Nat.choose (n + k + 1) n = 1 + 1 / k := by
    negate_state
    use 0; norm_num
    erw [tsum_eq_zero_of_not_summable] <;> norm_num
    exact_mod_cast mt (summable_nat_add_iff 1 |> Iff.mp) Real.not_summable_natCast_inv
-/
theorem my_favorite_theorem (k : ℕ) :
  ∑' n : ℕ, (1 : ℝ) / Nat.choose (n + k + 1) n = 1 + 1 / k := by
  sorry
```

Custom `negate_state` tactic:

```lean
import Mathlib
open Lean Meta Elab Tactic in
elab "revert_all" : tactic => do
  let goals ← getGoals
  let mut newGoals : List MVarId := []
  for mvarId in goals do
    newGoals := newGoals.append [(← mvarId.revertAll)]
  setGoals newGoals

open Lean.Elab.Tactic in
macro "negate_state" : tactic => `(tactic|
  (
    guard_goal_nums 1
    revert_all
    refine @(((by admit) : ∀ {p : Prop}, ¬p → p) ?_)
    try push_neg
  )
)
```

### Integrate Seamlessly into Lean Projects

Aristotle's Terminal interface automatically uploads and manages imports and dependencies.

#### Simple usage (one line):

```python
import asyncio
import aristotlelib

async def main():
    solution_path = await aristotlelib.Project.prove_from_file(
        input_file_path="path/to/your/file.lean"
    )
    print(f"Solution saved to: {solution_path}")

asyncio.run(main())
```

#### Manual control over dependencies:

```python
import asyncio
import aristotlelib

async def main():
    # Create a new project
    project = await aristotlelib.Project.create()
    print(f"Created project: {project.project_id}")

    # Manually add files needed for import
    await project.add_context(["path/to/context1.lean", "path/to/context2.lean"])

    # Solve with input content and optional formal context
    await project.solve(
        input_content="theorem my_theorem : True := trivial",
        formal_input_context="path/to/formal_context.lean"  # Optional
    )

    # Wait for completion
    while project.status not in [
        aristotlelib.ProjectStatus.COMPLETE,
        aristotlelib.ProjectStatus.FAILED
    ]:
        await asyncio.sleep(30)  # Poll every 30 seconds
        await project.refresh()
        print(f"Status: {project.status}")

    if project.status == aristotlelib.ProjectStatus.COMPLETE:
        solution_path = await project.get_solution()
        print(f"Solution saved to: {solution_path}")

asyncio.run(main())
```

---

## API Reference

### Project Class

Main class for interacting with Aristotle projects.

#### `Project.create(context_file_paths=None, validate_lean_project_root=True)`

Create a new Aristotle project.

**Parameters**:
- `context_file_paths` (list[Path | str], optional): List of file paths to include for import (up to 10 at a time)
- `validate_lean_project_root` (bool): Whether to validate Lean project structure (recommended: True)

**Returns**: Project instance

#### `Project.prove_from_file(...)`

Convenience method to prove a theorem from a file with automatic import resolution.

**Parameters**:
- `input_file_path` (Path | str): Path to the input Lean file (or text file for informal mode)
- `input_content` (str): Alternatively, provide the content directly as a string
- `auto_add_imports` (bool): Automatically add imported files as context
- `context_file_paths` (list[Path | str], optional): Manual context files. Cannot be used with `auto_add_imports`
- `validate_lean_project` (bool): Validate that this is a valid Lean project
- `wait_for_completion` (bool): Whether to wait for project completion before returning
- `polling_interval_seconds` (int): Seconds to wait between status checks
- `max_polling_failures` (int): Max polling failures before requiring manual status check
- `output_file_path` (Path | str, optional): Desired path to the output Lean file
- `project_input_type` (ProjectInputType): `FORMAL_LEAN` (default) or `INFORMAL`
- `formal_input_context` (Path | str, optional): Lean file with formal context (informal mode only)

**Returns**: str - Path to solution file, or project ID if `wait_for_completion` is False

#### `project.add_context(context_file_paths, batch_size=10, validate_lean_project_root=True)`

Add files used as imports to existing project (up to 10 per request).

**Parameters**:
- `context_file_paths` (list[Path | str]): Files to add as context
- `batch_size` (int): Files to upload per batch (max 10)
- `validate_lean_project_root` (bool): Validate project structure

#### `project.solve(input_file_path=None, input_content=None, formal_input_context=None)`

Solve the project with either a file or text.

**Parameters**:
- `input_file_path` (Path | str, optional): Path to input file
- `input_content` (str, optional): Text content to solve
- `formal_input_context` (Path | str, optional): Lean file with formal context (informal mode only)

⚠️ **Note**: Exactly one of `input_file_path` or `input_content` must be provided.

#### `project.get_solution(output_path=None)`

Download the solution file.

**Parameters**:
- `output_path` (Path | str, optional): Where to save the solution

**Returns**: Path to the downloaded solution file

#### `project.refresh()`

Refresh the project status from the API.

---

### Project Input Type

```python
class ProjectInputType(Enum):
    FORMAL_LEAN = 2
    INFORMAL = 3
```

**Descriptions**:
- `FORMAL_LEAN`: Input is formal Lean code with theorem statements (default)
- `INFORMAL`: Input is natural language description of mathematical problems

---

### Project Status

```python
class ProjectStatus(Enum):
    NOT_STARTED = "NOT_STARTED"
    QUEUED = "QUEUED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    PENDING_RETRY = "PENDING_RETRY"
```

**Status Descriptions**:

- **NOT_STARTED**: Project created but no solve request submitted. Call `project.solve()` to begin.
- **QUEUED**: Solve request submitted and waiting in queue.
- **IN_PROGRESS**: Aristotle actively working on proving the theorem.
- **COMPLETE**: Project complete! Call `project.get_solution()` to download results.
- **FAILED**: Internal error occurred. Team has been notified.
- **PENDING_RETRY**: Internal error detected that should not recur. Will be re-queued shortly.

---

## Error Handling

SDK exception types:
- `AristotleAPIError`: API-related errors
- `LeanProjectError`: Lean project validation errors

---

## Lean Project Requirements

Aristotle works best with properly structured Lean projects:

**Required**:
- `lakefile.toml` configuration file or `lakefile.lean` (legacy)
- `lean-toolchain` file
- Proper import structure

**SDK automatically**:
- Detects your project root
- Validates file paths are within the project
- Resolves imports to include dependencies
- Handles file size limits (100MB max per file)

---

## Examples

### Basic theorem proving

```python
import asyncio
import aristotlelib
import logging

async def prove_simple_theorem():
    # Set API key
    aristotlelib.set_api_key("your-key")

    # Set logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s"
    )

    # Prove a simple theorem and save it to output.lean
    await aristotlelib.Project.prove_from_file(
        "examples/simple.lean",
        output_file_path="examples/output.lean"
    )

asyncio.run(prove_simple_theorem())
```

### Check status of existing projects

```python
import asyncio
import aristotlelib

async def get_project_status():
    # Load an existing project
    project = await aristotlelib.Project.from_id("existing-project-id")

    # Check status
    print(f"Project status: {project.status}")

    if project.status == aristotlelib.ProjectStatus.COMPLETE:
        # Download the solution
        await project.get_solution(output_path="examples/output.lean")

asyncio.run(get_project_status())
```

### Get all your projects

```python
import asyncio
import aristotlelib
from aristotlelib.project import ProjectStatus

async def list_projects():
    # Get all projects
    projects, pagination_key = await aristotlelib.Project.list_projects(limit=10)

    for project in projects:
        print(f"Project {project.project_id}: {project.status}")

    # Filter by status - single status
    active_projects, _ = await aristotlelib.Project.list_projects(
        limit=10,
        status=ProjectStatus.IN_PROGRESS
    )

    # Filter by multiple statuses
    filtered_projects, _ = await aristotlelib.Project.list_projects(
        limit=10,
        status=[ProjectStatus.QUEUED, ProjectStatus.IN_PROGRESS, ProjectStatus.PENDING_RETRY]
    )

    # Get next page if available
    while pagination_key:
        more_projects, pagination_key = await aristotlelib.Project.list_projects(
            pagination_key=pagination_key
        )
        print(f"Found {len(more_projects)} more projects")

asyncio.run(list_projects())
```

---

## Tips and Tricks

### Replace sorry with admit

By default, Aristotle attempts to fill in all sorries. If you only want Aristotle to fill specific sorries, replace others with `admit`. Useful for:
- Incomplete structures and defs
- Faster results on specific proofs

### Warnings from aesop

The following warning is **expected** and does not indicate a problem:
```
aesop: failed to prove the goal after exhaustive search
```

This occurs when aesop is used as a non-terminal tactic. To suppress:

```lean
aesop (config := { warnOnNonterminal := false })
```
