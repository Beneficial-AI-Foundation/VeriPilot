"""
Lake build runner for VeriPilot.

Executes `lake build` asynchronously and captures output.

CRITICAL: lake build must run in an actual Lean project directory
(with lakefile.lean), NOT in the veripilot Python directory.
"""

import asyncio
import time
from pathlib import Path
from typing import Optional

from . import BuildResult


async def run_lake_build(
    project_dir: str,
    timeout: int = 300,
) -> BuildResult:
    """
    Run `lake build` in the specified Lean project directory.

    Args:
        project_dir: Path to Lean project (must contain lakefile.lean)
        timeout: Maximum time to wait in seconds (default 5 minutes)

    Returns:
        BuildResult with success status, output, and timing
    """
    start_time = time.time()

    # Validate project directory
    project_path = Path(project_dir)
    if not project_path.exists():
        return BuildResult(
            success=False,
            stdout="",
            stderr=f"Project directory does not exist: {project_dir}",
            return_code=-1,
            elapsed_time=0.0,
        )

    lakefile = project_path / "lakefile.lean"
    lakefile_toml = project_path / "lakefile.toml"
    if not lakefile.exists() and not lakefile_toml.exists():
        return BuildResult(
            success=False,
            stdout="",
            stderr=f"No lakefile found in: {project_dir}",
            return_code=-1,
            elapsed_time=0.0,
        )

    try:
        proc = await asyncio.create_subprocess_exec(
            "lake",
            "build",
            cwd=project_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )

        elapsed = time.time() - start_time

        return BuildResult(
            success=(proc.returncode == 0),
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            return_code=proc.returncode or 0,
            elapsed_time=elapsed,
        )

    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        # Try to kill the process
        try:
            proc.kill()
            await proc.wait()
        except Exception:
            pass

        return BuildResult(
            success=False,
            stdout="",
            stderr=f"lake build timed out after {timeout}s",
            return_code=-1,
            elapsed_time=elapsed,
        )

    except FileNotFoundError:
        return BuildResult(
            success=False,
            stdout="",
            stderr="lake command not found. Is Lean/Lake installed?",
            return_code=-1,
            elapsed_time=time.time() - start_time,
        )

    except Exception as e:
        return BuildResult(
            success=False,
            stdout="",
            stderr=f"Error running lake build: {e}",
            return_code=-1,
            elapsed_time=time.time() - start_time,
        )


async def run_lake_build_module(
    project_dir: str,
    module: str,
    timeout: int = 300,
) -> BuildResult:
    """
    Run `lake build <module>` for faster targeted builds.

    Args:
        project_dir: Path to Lean project
        module: Module name to build (e.g., "DalekLean.Specs.SubLoop")
        timeout: Maximum time to wait in seconds

    Returns:
        BuildResult with success status, output, and timing
    """
    start_time = time.time()

    project_path = Path(project_dir)
    if not project_path.exists():
        return BuildResult(
            success=False,
            stdout="",
            stderr=f"Project directory does not exist: {project_dir}",
            return_code=-1,
            elapsed_time=0.0,
        )

    try:
        proc = await asyncio.create_subprocess_exec(
            "lake",
            "build",
            module,
            cwd=project_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )

        elapsed = time.time() - start_time

        return BuildResult(
            success=(proc.returncode == 0),
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            return_code=proc.returncode or 0,
            elapsed_time=elapsed,
        )

    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        try:
            proc.kill()
            await proc.wait()
        except Exception:
            pass

        return BuildResult(
            success=False,
            stdout="",
            stderr=f"lake build {module} timed out after {timeout}s",
            return_code=-1,
            elapsed_time=elapsed,
        )

    except Exception as e:
        return BuildResult(
            success=False,
            stdout="",
            stderr=f"Error running lake build {module}: {e}",
            return_code=-1,
            elapsed_time=time.time() - start_time,
        )


async def check_lake_available() -> bool:
    """
    Check if lake command is available.

    Returns:
        True if lake is available, False otherwise
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "lake",
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0
    except Exception:
        return False


def get_module_from_file(file_path: str, project_dir: str) -> Optional[str]:
    """
    Derive Lean module name from file path.

    Args:
        file_path: Absolute path to .lean file
        project_dir: Project root directory

    Returns:
        Module name (e.g., "DalekLean.Specs.SubLoop") or None
    """
    try:
        file_path = Path(file_path).resolve()
        project_path = Path(project_dir).resolve()

        # Get relative path from project root
        rel_path = file_path.relative_to(project_path)

        # Remove .lean extension and convert path separators to dots
        module_path = str(rel_path).replace("/", ".").replace("\\", ".")
        if module_path.endswith(".lean"):
            module_path = module_path[:-5]

        return module_path

    except ValueError:
        # file_path not relative to project_dir
        return None
