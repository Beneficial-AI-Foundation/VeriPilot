"""
Lean Search Tools for VeriPilot.

Provides theorem and tactic search capabilities via:
1. Kimina Lean Server (project-numina/kimina-lean-server) - FastAPI REST API
2. LeanSearch (frenzymath/LeanSearch) - CLI-based semantic search

Based on research: docs/claude-helpers/resources/Perplexity_leanDex_search.md

Usage:
    # Async HTTP client for Kimina server
    result = await kimina_verify(proof_code)
    results = await kimina_search(query)

    # Subprocess wrapper for LeanSearch CLI
    output = leansearch_cli(query, leansearch_path)

    # Unified interface (tries Kimina first, falls back to CLI)
    results = await search_lean_library(query)
"""

import asyncio
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LeanSearchResult:
    """Result from a Lean search query."""

    name: str
    type_signature: str = ""
    doc_string: Optional[str] = None
    proof_preview: Optional[str] = None
    source: str = "leansearch"  # "kimina", "leansearch", "rag"
    score: float = 0.0
    file_path: Optional[str] = None
    line_number: Optional[int] = None


@dataclass
class KiminaVerifyResult:
    """Result from Kimina proof verification."""

    success: bool
    goals: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


# =============================================================================
# Kimina Lean Server (project-numina/kimina-lean-server)
# =============================================================================

# Default Kimina server URL (can be overridden via env var)
KIMINA_SERVER_URL = os.environ.get("KIMINA_SERVER_URL", "http://localhost:8000")


async def kimina_verify(
    proof_code: str,
    server_url: str = KIMINA_SERVER_URL,
    timeout: float = 30.0,
) -> KiminaVerifyResult:
    """
    Verify a Lean proof using Kimina Lean Server.

    Args:
        proof_code: The Lean code to verify
        server_url: Kimina server URL (default: http://localhost:8000)
        timeout: Request timeout in seconds

    Returns:
        KiminaVerifyResult with verification status and any errors/goals

    Raises:
        ConnectionError: If server is not available
        TimeoutError: If verification takes too long
    """
    if not HTTPX_AVAILABLE:
        raise ImportError("httpx required for Kimina client. Install with: pip install httpx")

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{server_url}/verify",
                json={"code": proof_code},
            )
            response.raise_for_status()
            data = response.json()

            return KiminaVerifyResult(
                success=data.get("success", False),
                goals=data.get("goals", []),
                errors=data.get("errors", []),
                messages=data.get("messages", []),
            )
    except httpx.ConnectError as e:
        logger.warning(f"Kimina server not available at {server_url}: {e}")
        raise ConnectionError(f"Kimina server not available: {e}")
    except httpx.TimeoutException as e:
        logger.warning(f"Kimina verification timed out: {e}")
        raise TimeoutError(f"Verification timed out: {e}")


async def kimina_search(
    query: str,
    server_url: str = KIMINA_SERVER_URL,
    top_k: int = 10,
    timeout: float = 10.0,
) -> list[LeanSearchResult]:
    """
    Search for Lean theorems using Kimina server (if search endpoint available).

    Note: This endpoint may not be available in all Kimina versions.
    The main purpose of Kimina is proof verification, not search.

    Args:
        query: Natural language or Lean expression to search for
        server_url: Kimina server URL
        top_k: Maximum number of results
        timeout: Request timeout in seconds

    Returns:
        List of LeanSearchResult objects
    """
    if not HTTPX_AVAILABLE:
        raise ImportError("httpx required for Kimina client. Install with: pip install httpx")

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                f"{server_url}/search",
                params={"q": query, "limit": top_k},
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("results", []):
                results.append(LeanSearchResult(
                    name=item.get("name", ""),
                    type_signature=item.get("type", ""),
                    doc_string=item.get("doc", None),
                    proof_preview=item.get("proof", None),
                    source="kimina",
                    score=item.get("score", 0.0),
                ))
            return results

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.info("Kimina search endpoint not available (404)")
            return []
        raise
    except httpx.ConnectError as e:
        logger.warning(f"Kimina server not available at {server_url}: {e}")
        return []


# =============================================================================
# LeanSearch CLI (frenzymath/LeanSearch)
# =============================================================================

# Default path to LeanSearch installation
LEANSEARCH_PATH = os.environ.get("LEANSEARCH_PATH", "/workspace/tools/LeanSearch")


def leansearch_cli(
    query: str,
    leansearch_path: str = LEANSEARCH_PATH,
    timeout: float = 30.0,
) -> str:
    """
    Search Lean 4 libraries via LeanSearch CLI.

    This wraps the LeanSearch CLI tool (python search.py <query>).

    Args:
        query: The search query (natural language or Lean expression)
        leansearch_path: Path to LeanSearch installation directory
        timeout: Command timeout in seconds

    Returns:
        Raw output from search.py

    Raises:
        FileNotFoundError: If LeanSearch not installed at specified path
        TimeoutError: If search takes too long
    """
    search_script = os.path.join(leansearch_path, "search.py")

    if not os.path.exists(search_script):
        raise FileNotFoundError(
            f"LeanSearch not found at {leansearch_path}. "
            f"Clone from: https://github.com/frenzymath/LeanSearch"
        )

    try:
        result = subprocess.run(
            ["python", "search.py", query],
            capture_output=True,
            text=True,
            cwd=leansearch_path,
            timeout=timeout,
        )

        if result.returncode != 0:
            logger.warning(f"LeanSearch failed: {result.stderr}")
            return ""

        return result.stdout

    except subprocess.TimeoutExpired:
        logger.warning(f"LeanSearch timed out after {timeout}s")
        raise TimeoutError(f"LeanSearch timed out after {timeout}s")


def parse_leansearch_output(output: str) -> list[LeanSearchResult]:
    """
    Parse LeanSearch CLI output into structured results.

    The exact format depends on LeanSearch version, but typically includes:
    - Declaration name
    - Type signature
    - Documentation

    Args:
        output: Raw CLI output

    Returns:
        List of LeanSearchResult objects
    """
    results = []

    if not output.strip():
        return results

    # Simple line-based parsing (adjust based on actual LeanSearch output format)
    current_name = None
    current_sig = ""
    current_doc = None

    for line in output.strip().split("\n"):
        line = line.strip()

        if not line:
            if current_name:
                results.append(LeanSearchResult(
                    name=current_name,
                    type_signature=current_sig,
                    doc_string=current_doc,
                    source="leansearch",
                ))
                current_name = None
                current_sig = ""
                current_doc = None
            continue

        # Try to detect declaration names (usually start with a letter, contain no spaces)
        if line.startswith("•") or line.startswith("-"):
            # Bullet point format
            parts = line.lstrip("•-").strip().split(":", 1)
            if parts:
                current_name = parts[0].strip()
                if len(parts) > 1:
                    current_sig = parts[1].strip()
        elif ":" in line and not line.startswith(" "):
            # Name : Type format
            parts = line.split(":", 1)
            current_name = parts[0].strip()
            current_sig = parts[1].strip() if len(parts) > 1 else ""
        elif current_name and line.startswith("  "):
            # Indented doc string
            if current_doc is None:
                current_doc = line.strip()
            else:
                current_doc += " " + line.strip()

    # Don't forget the last result
    if current_name:
        results.append(LeanSearchResult(
            name=current_name,
            type_signature=current_sig,
            doc_string=current_doc,
            source="leansearch",
        ))

    return results


# =============================================================================
# Unified Search Interface
# =============================================================================

async def search_lean_library(
    query: str,
    use_kimina: bool = True,
    use_leansearch: bool = True,
    kimina_url: str = KIMINA_SERVER_URL,
    leansearch_path: str = LEANSEARCH_PATH,
    top_k: int = 10,
) -> list[LeanSearchResult]:
    """
    Search Lean libraries using available tools.

    Tries Kimina server first (if enabled and available), then falls back
    to LeanSearch CLI.

    Args:
        query: Search query (natural language or Lean expression)
        use_kimina: Whether to try Kimina server
        use_leansearch: Whether to try LeanSearch CLI
        kimina_url: Kimina server URL
        leansearch_path: Path to LeanSearch installation
        top_k: Maximum results to return

    Returns:
        List of LeanSearchResult objects from the first successful source
    """
    results = []

    # Try Kimina first (faster, REST API)
    if use_kimina:
        try:
            results = await kimina_search(query, server_url=kimina_url, top_k=top_k)
            if results:
                logger.info(f"Got {len(results)} results from Kimina")
                return results[:top_k]
        except Exception as e:
            logger.debug(f"Kimina search failed: {e}")

    # Fall back to LeanSearch CLI
    if use_leansearch:
        try:
            output = leansearch_cli(query, leansearch_path=leansearch_path)
            results = parse_leansearch_output(output)
            if results:
                logger.info(f"Got {len(results)} results from LeanSearch CLI")
                return results[:top_k]
        except FileNotFoundError:
            logger.debug(f"LeanSearch not installed at {leansearch_path}")
        except Exception as e:
            logger.debug(f"LeanSearch failed: {e}")

    logger.info("No results from any Lean search tool")
    return []


# =============================================================================
# Async Helpers
# =============================================================================

def run_search_sync(query: str, **kwargs) -> list[LeanSearchResult]:
    """Synchronous wrapper for search_lean_library."""
    return asyncio.run(search_lean_library(query, **kwargs))


async def is_kimina_available(server_url: str = KIMINA_SERVER_URL) -> bool:
    """Check if Kimina server is running and responding."""
    if not HTTPX_AVAILABLE:
        return False

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{server_url}/health")
            return response.status_code == 200
    except Exception:
        return False


def is_leansearch_installed(leansearch_path: str = LEANSEARCH_PATH) -> bool:
    """Check if LeanSearch is installed at the specified path."""
    search_script = os.path.join(leansearch_path, "search.py")
    return os.path.exists(search_script)
