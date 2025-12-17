"""
MCP client for lean-lsp integration.

Spawns and communicates with the lean-lsp-mcp server to get
instant diagnostics without running lake build.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticItem:
    """A single diagnostic message from the Lean LSP."""

    line: int
    column: int
    end_line: Optional[int]
    end_column: Optional[int]
    severity: str  # "error", "warning", "info"
    message: str

    @classmethod
    def from_dict(cls, d: dict) -> "DiagnosticItem":
        """Parse from MCP tool result."""
        return cls(
            line=d.get("line", 0),
            column=d.get("column", 0),
            end_line=d.get("end_line"),
            end_column=d.get("end_column"),
            severity=d.get("severity", "error"),
            message=d.get("message", ""),
        )


@dataclass
class GoalState:
    """Proof state at a position."""

    goals_before: Optional[str]
    goals_after: Optional[str]
    line_context: Optional[str]

    @property
    def is_complete(self) -> bool:
        """Check if proof is complete (no remaining goals)."""
        if self.goals_after is None:
            return False
        after = self.goals_after.strip().lower()
        return after == "" or after == "no goals" or "no goals" in after


class LeanMCPClient:
    """
    Client for the lean-lsp MCP server.

    Provides instant diagnostics and goal state queries for Lean files.
    """

    def __init__(
        self,
        command: str = "/workspace/.local/bin/uvx",
        args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
    ):
        """
        Initialize MCP client configuration.

        Args:
            command: Path to uvx or the MCP server command
            args: Arguments to pass (default: ["lean-lsp-mcp"])
            env: Additional environment variables
        """
        self.command = command
        self.args = args or ["lean-lsp-mcp"]

        # Build environment with elan path
        self.env = {
            "PATH": f"/workspace/.elan/bin:{os.environ.get('PATH', '')}",
            "ELAN_HOME": "/workspace/.elan",
        }
        if env:
            self.env.update(env)

        self._session: Optional[ClientSession] = None
        self._connected = False

    @asynccontextmanager
    async def connect(self):
        """
        Context manager for MCP server connection.

        Usage:
            async with client.connect():
                diagnostics = await client.get_diagnostics(file_path)
        """
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                # Initialize the session
                await session.initialize()
                self._session = session
                self._connected = True
                try:
                    yield self
                finally:
                    self._connected = False
                    self._session = None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """
        Call an MCP tool and return the result.

        Args:
            name: Tool name (e.g., "lean_diagnostic_messages")
            arguments: Tool arguments

        Returns:
            Tool result (parsed JSON)

        Raises:
            RuntimeError: If not connected
        """
        if not self._connected or not self._session:
            raise RuntimeError("Not connected to MCP server. Use 'async with client.connect():'")

        result = await self._session.call_tool(name, arguments)

        # Parse result content
        if result.content:
            for item in result.content:
                if hasattr(item, "text"):
                    import json

                    try:
                        return json.loads(item.text)
                    except json.JSONDecodeError:
                        return item.text
        return None

    async def get_diagnostics(
        self,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> list[DiagnosticItem]:
        """
        Get compiler diagnostics for a Lean file.

        Args:
            file_path: Absolute path to the Lean file
            start_line: Optional start line filter
            end_line: Optional end line filter

        Returns:
            List of diagnostic items
        """
        args: dict[str, Any] = {"file_path": file_path}
        if start_line is not None:
            args["start_line"] = start_line
        if end_line is not None:
            args["end_line"] = end_line

        result = await self.call_tool("lean_diagnostic_messages", args)

        if not result:
            return []

        items = result.get("items", [])
        return [DiagnosticItem.from_dict(item) for item in items]

    async def get_goal(
        self,
        file_path: str,
        line: int,
        column: Optional[int] = None,
    ) -> Optional[GoalState]:
        """
        Get proof goal state at a position.

        Args:
            file_path: Absolute path to the Lean file
            line: Line number (1-indexed)
            column: Optional column number (1-indexed)

        Returns:
            GoalState or None if not in a proof context
        """
        args: dict[str, Any] = {"file_path": file_path, "line": line}
        if column is not None:
            args["column"] = column

        result = await self.call_tool("lean_goal", args)

        if not result:
            return None

        return GoalState(
            goals_before=result.get("goals_before"),
            goals_after=result.get("goals_after"),
            line_context=result.get("line_context"),
        )

    async def multi_attempt(
        self,
        file_path: str,
        line: int,
        snippets: list[str],
    ) -> list[dict[str, Any]]:
        """
        Try multiple tactics at a position without modifying the file.

        Args:
            file_path: Absolute path to the Lean file
            line: Line number (1-indexed)
            snippets: List of tactics to try

        Returns:
            List of results for each tactic
        """
        result = await self.call_tool(
            "lean_multi_attempt",
            {"file_path": file_path, "line": line, "snippets": snippets},
        )

        if not result:
            return []

        return result if isinstance(result, list) else []


# Singleton for reuse
_client_instance: Optional[LeanMCPClient] = None


def get_mcp_client() -> LeanMCPClient:
    """Get or create the singleton MCP client."""
    global _client_instance
    if _client_instance is None:
        _client_instance = LeanMCPClient()
    return _client_instance
