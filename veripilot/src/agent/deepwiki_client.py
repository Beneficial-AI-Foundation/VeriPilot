"""
DeepWiki MCP client for dynamic repo knowledge queries.

Connects to DeepWiki's free MCP server (https://mcp.deepwiki.com/mcp)
to query documentation for public GitHub repos at runtime. No auth required.

Based on Karpathy's pattern: give your agent DeepWiki MCP access to
look up lemmas, API patterns, and code from any public repo.
Ref: docs/claude-helpers/resources/IMP_resources/karpathy_deepwiki.md
"""

import json
import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

DEEPWIKI_MCP_URL = "https://mcp.deepwiki.com/mcp"

# Known repos for Lean/verification work
KNOWN_REPOS = {
    "mathlib4": "leanprover-community/mathlib4",
    "aeneas": "AeneasVerif/aeneas",
    "lean4": "leanprover/lean4",
}


class DeepWikiClient:
    """
    Client for the DeepWiki MCP server.

    Provides three tools via JSON-RPC 2.0 over Streamable HTTP:
    - ask_question: Ask a natural language question about a repo
    - read_wiki_contents: Read documentation for a topic in a repo
    - read_wiki_structure: Get the table of contents for a repo's docs

    Results are cached per-session to avoid redundant queries.
    """

    def __init__(self, timeout: float = 30.0):
        self._timeout = timeout
        self._cache: dict[str, str] = {}
        self._http_client: Optional[httpx.AsyncClient] = None
        self._request_id = 0

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=self._timeout)
        return self._http_client

    async def close(self):
        """Close HTTP client connection."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _resolve_repo(self, repo: str) -> str:
        """Resolve short name (e.g. 'mathlib4') to full repo path."""
        return KNOWN_REPOS.get(repo, repo)

    async def _call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> Optional[str]:
        """
        Call a DeepWiki MCP tool via JSON-RPC 2.0.

        Returns tool result as string, or None on error.
        """
        cache_key = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
        if cache_key in self._cache:
            logger.debug(f"DeepWiki cache hit: {tool_name}")
            return self._cache[cache_key]

        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
            "id": self._next_id(),
        }

        try:
            client = await self._get_client()
            response = await client.post(
                DEEPWIKI_MCP_URL,
                json=request,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                logger.warning(f"DeepWiki RPC error: {data['error']}")
                return None

            result = data.get("result", {})
            content = result.get("content", []) if isinstance(result, dict) else []
            text_parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]

            if not text_parts:
                # Fallback: result might be a plain string
                if isinstance(result, str):
                    self._cache[cache_key] = result
                    return result
                logger.warning(f"DeepWiki returned no text content for {tool_name}")
                return None

            result_text = "\n".join(text_parts)
            self._cache[cache_key] = result_text
            logger.debug(f"DeepWiki success: {tool_name} ({len(result_text)} chars)")
            return result_text

        except httpx.TimeoutException:
            logger.warning(f"DeepWiki timeout for {tool_name} (>{self._timeout}s)")
            return None
        except Exception as e:
            logger.warning(f"DeepWiki error for {tool_name}: {e}")
            return None

    async def ask_question(self, repo: str, question: str) -> Optional[str]:
        """
        Ask a natural language question about a repo.

        Args:
            repo: Short name ('mathlib4', 'aeneas', 'lean4') or full 'org/repo'
            question: Natural language question

        Returns:
            Answer as markdown string, or None on error
        """
        return await self._call_tool(
            "ask_question",
            {"repo": self._resolve_repo(repo), "question": question},
        )

    async def read_wiki_contents(self, repo: str, topic: str) -> Optional[str]:
        """
        Read documentation for a specific topic in a repo.

        Args:
            repo: Short name or full 'org/repo'
            topic: Topic path (e.g., "Tactics/Arithmetic")

        Returns:
            Documentation content as markdown, or None on error
        """
        return await self._call_tool(
            "read_wiki_contents",
            {"repo": self._resolve_repo(repo), "topic": topic},
        )

    async def read_wiki_structure(self, repo: str) -> Optional[str]:
        """
        Get table of contents for a repo's documentation.

        Args:
            repo: Short name or full 'org/repo'

        Returns:
            Table of contents as structured text, or None on error
        """
        return await self._call_tool(
            "read_wiki_structure",
            {"repo": self._resolve_repo(repo)},
        )


# Module-level singleton
_global_client: Optional[DeepWikiClient] = None


def get_deepwiki_client(timeout: float = 30.0) -> DeepWikiClient:
    """Get or create module-level DeepWiki client singleton."""
    global _global_client
    if _global_client is None:
        _global_client = DeepWikiClient(timeout=timeout)
    return _global_client
