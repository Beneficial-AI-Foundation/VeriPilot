"""
MCP client for lean-lsp integration.

Spawns and communicates with the lean-lsp-mcp server to get
instant diagnostics without running lake build.
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from time import time
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError

logger = logging.getLogger(__name__)

# MCP error code for file worker termination
MCP_WORKER_TERMINATED = -32801


class MCPWorkerCrashError(Exception):
    """Raised when MCP file worker terminates (error -32801)."""
    pass


class MCPUnavailableError(Exception):
    """Raised when MCP client is unavailable (circuit tripped or restart failed)."""
    pass


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class FailureTracker:
    """Per-file circuit breaker with half-open recovery.

    States:
    - CLOSED: Normal operation, calls allowed
    - OPEN: Too many failures, calls blocked until cooldown expires
    - HALF_OPEN: Cooldown expired, one probe call allowed to test recovery
    """

    def __init__(self, threshold: int = 3, cooldown_seconds: float = 30.0):
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = 0.0
        self.trip_time = 0.0

    def record_failure(self) -> None:
        """Record a failed call. Opens circuit after threshold failures."""
        self.failure_count += 1
        self.last_failure_time = time()
        if self.failure_count >= self.threshold:
            if self.state != CircuitState.OPEN:
                logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")
                self.state = CircuitState.OPEN
                self.trip_time = time()

    def record_success(self) -> None:
        """Record a successful call. Closes circuit if in half-open state."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker CLOSED after successful recovery")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
        elif self.state == CircuitState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)

    def should_allow_call(self) -> bool:
        """Check whether a call should be allowed through the circuit."""
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.HALF_OPEN:
            return True
        # OPEN state: check if cooldown expired
        if time() - self.trip_time >= self.cooldown_seconds:
            logger.info(
                f"Circuit breaker entering HALF_OPEN after {self.cooldown_seconds}s cooldown"
            )
            self.state = CircuitState.HALF_OPEN
            return True
        return False

    def is_tripped(self) -> bool:
        """Check if circuit is currently tripped (OPEN or HALF_OPEN)."""
        return self.state in (CircuitState.OPEN, CircuitState.HALF_OPEN)


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

        # Call serialization
        self._call_lock = asyncio.Lock()
        self._is_restarting = False

        # Restart tracking
        self._restart_count = 0
        self._max_restarts = 3
        self._backoff_seconds = 1.5
        self._successful_calls = 0

        # Per-file circuit breakers
        self._file_trackers: dict[str, FailureTracker] = {}

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

    async def _restart_connection(self, warmup_timeout: float = 120.0) -> bool:
        """
        Restart MCP connection after crash. Returns True if successful.

        Prepares client state for reconnection. The actual reconnection
        happens when the caller re-enters the connect() context.
        """
        self._restart_count += 1
        if self._restart_count > self._max_restarts:
            logger.error(f"Restart limit reached ({self._max_restarts})")
            return False

        self._is_restarting = True
        try:
            if self._session:
                try:
                    self._connected = False
                    self._session = None
                except Exception:
                    pass

            backoff = self._backoff_seconds * self._restart_count
            logger.info(f"Restarting MCP after {backoff}s backoff (attempt {self._restart_count})")
            await asyncio.sleep(backoff)

            logger.info(f"MCP connection ready for restart (attempt {self._restart_count})")
            return True
        finally:
            self._is_restarting = False

    def _record_successful_call(self) -> None:
        """Record a successful call. Resets restart counter after 10 consecutive successes."""
        self._successful_calls += 1
        if self._successful_calls >= 10:
            self._restart_count = 0
            self._successful_calls = 0
            logger.debug("Restart counter reset after stable operation")

    def _get_tracker(self, file_path: str) -> FailureTracker:
        """Get or create a per-file circuit breaker tracker."""
        if file_path not in self._file_trackers:
            self._file_trackers[file_path] = FailureTracker(threshold=3, cooldown_seconds=30.0)
        return self._file_trackers[file_path]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """
        Call an MCP tool and return the result.

        Serialized via asyncio.Lock to prevent concurrent call collisions.
        Detects -32801 file worker crashes and raises MCPWorkerCrashError.

        Args:
            name: Tool name (e.g., "lean_diagnostic_messages")
            arguments: Tool arguments

        Returns:
            Tool result (parsed JSON)

        Raises:
            RuntimeError: If not connected
            MCPWorkerCrashError: If MCP file worker terminated (-32801)
        """
        if not self._connected or not self._session:
            raise RuntimeError("Not connected to MCP server. Use 'async with client.connect():'")

        while self._is_restarting:
            await asyncio.sleep(0.1)

        async with self._call_lock:
            try:
                result = await self._session.call_tool(name, arguments)
            except McpError as e:
                if (
                    hasattr(e, "error")
                    and hasattr(e.error, "code")
                    and e.error.code == MCP_WORKER_TERMINATED
                ):
                    self._successful_calls = 0
                    raise MCPWorkerCrashError(
                        f"MCP file worker crashed (error {MCP_WORKER_TERMINATED}): {e}"
                    )
                raise

            if result.content:
                for item in result.content:
                    if hasattr(item, "text"):
                        try:
                            parsed = json.loads(item.text)
                            self._record_successful_call()
                            return parsed
                        except json.JSONDecodeError:
                            self._record_successful_call()
                            return item.text
            self._record_successful_call()
            return None

    async def get_diagnostics(
        self,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> list[DiagnosticItem]:
        """
        Get compiler diagnostics for a Lean file.

        Checks per-file circuit breaker before calling.

        Args:
            file_path: Absolute path to the Lean file
            start_line: Optional start line filter
            end_line: Optional end line filter

        Returns:
            List of diagnostic items

        Raises:
            MCPUnavailableError: If circuit breaker is tripped for this file
        """
        tracker = self._get_tracker(file_path)
        if not tracker.should_allow_call():
            raise MCPUnavailableError(f"Circuit tripped for {file_path}")

        args: dict[str, Any] = {"file_path": file_path}
        if start_line is not None:
            args["start_line"] = start_line
        if end_line is not None:
            args["end_line"] = end_line

        try:
            result = await self.call_tool("lean_diagnostic_messages", args)
        except (MCPWorkerCrashError, Exception) as e:
            tracker.record_failure()
            raise

        if not result:
            tracker.record_success()
            return []

        if isinstance(result, str):
            logger.warning(f"MCP returned string instead of dict for diagnostics: {result[:200]}")
            tracker.record_success()
            return []

        tracker.record_success()
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

        Checks per-file circuit breaker before calling.

        Args:
            file_path: Absolute path to the Lean file
            line: Line number (1-indexed)
            column: Optional column number (1-indexed)

        Returns:
            GoalState or None if not in a proof context

        Raises:
            MCPUnavailableError: If circuit breaker is tripped for this file
        """
        tracker = self._get_tracker(file_path)
        if not tracker.should_allow_call():
            raise MCPUnavailableError(f"Circuit tripped for {file_path}")

        args: dict[str, Any] = {"file_path": file_path, "line": line}
        if column is not None:
            args["column"] = column

        try:
            result = await self.call_tool("lean_goal", args)
        except (MCPWorkerCrashError, Exception) as e:
            tracker.record_failure()
            raise

        if not result:
            tracker.record_success()
            return None

        if isinstance(result, str):
            logger.warning(f"MCP returned string instead of dict for goal: {result[:200]}")
            tracker.record_success()
            return None

        tracker.record_success()
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
        [DEPRECATED] Try multiple tactics at a position.

        WARNING: multi_attempt hangs after 1-2 calls and crashes the LSP.
        Use edit_file + lean_goal loop instead (Plan 03).

        Args:
            file_path: Absolute path to the Lean file
            line: Line number (1-indexed)
            snippets: List of tactics to try

        Returns:
            List of results for each tactic

        Raises:
            MCPUnavailableError: If circuit breaker is tripped for this file
        """
        logger.warning("multi_attempt is deprecated. Use edit_file + lean_goal instead.")

        tracker = self._get_tracker(file_path)
        if not tracker.should_allow_call():
            raise MCPUnavailableError(f"Circuit tripped for {file_path}")

        try:
            result = await self.call_tool(
                "lean_multi_attempt",
                {"file_path": file_path, "line": line, "snippets": snippets},
            )
        except (MCPWorkerCrashError, Exception) as e:
            tracker.record_failure()
            raise

        if not result:
            tracker.record_success()
            return []

        tracker.record_success()
        return result if isinstance(result, list) else []


# Singleton for reuse
_client_instance: Optional[LeanMCPClient] = None


def get_mcp_client() -> LeanMCPClient:
    """Get or create the singleton MCP client."""
    global _client_instance
    if _client_instance is None:
        _client_instance = LeanMCPClient()
    return _client_instance
