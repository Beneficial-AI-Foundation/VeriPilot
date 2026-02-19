"""
MCP client for lean-lsp integration.

Spawns and communicates with the lean-lsp-mcp server to get
instant diagnostics without running lake build.
"""

import asyncio
import json
import logging
import os
import shutil
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from time import time
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError

logger = logging.getLogger(__name__)

# MCP error code for file worker termination
MCP_WORKER_TERMINATED = -32801

# Tool timeout configuration (seconds)
SAFE_TOOLS = {"lean_goal", "lean_term_goal", "lean_hover_info", "lean_local_search", "lean_diagnostic_messages"}
SAFE_TIMEOUT = 10.0
DANGEROUS_TIMEOUT = 30.0


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


def join_goals(goals: Optional[list | str]) -> Optional[str]:
    """
    Convert goal list to string.

    MCP lean_goal returns goals_before/goals_after as arrays.
    Join them into a single string for consistent handling.
    """
    if goals is None:
        return None
    if isinstance(goals, str):
        return goals
    if isinstance(goals, list):
        if not goals:
            return ""
        return "\n\n".join(str(g) for g in goals)
    return str(goals)


class LeanMCPClient:
    """
    Client for the lean-lsp MCP server.

    Provides instant diagnostics and goal state queries for Lean files.
    """

    def __init__(
        self,
        command: Optional[str] = None,
        args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
    ):
        """
        Initialize MCP client configuration.

        Args:
            command: Path to uvx or the MCP server command (auto-detected if None)
            args: Arguments to pass (default: ["lean-lsp-mcp"])
            env: Additional environment variables
        """
        self.command = command or shutil.which("uvx") or "uvx"
        self.args = args or ["lean-lsp-mcp"]

        # Build environment with elan path (auto-detect ELAN_HOME)
        elan_home = os.environ.get("ELAN_HOME", str(Path.home() / ".elan"))
        self.env = {
            "PATH": f"{elan_home}/bin:{os.environ.get('PATH', '')}",
            "ELAN_HOME": elan_home,
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

        # File backups for safe edit/revert (disk-based for crash safety)
        self._file_backups: dict[str, Path] = {}  # file_path -> backup_path

        # File warmup tracking
        self._warmed_files: set[str] = set()

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

    @staticmethod
    def recover_orphaned_backups(directory: str) -> list[str]:
        """Restore any .vp_backup files left by a crashed session.

        Call this before starting verification on a directory to ensure
        no files are left in a modified state from a previous crash.

        Returns list of restored file paths.
        """
        restored = []
        for backup in Path(directory).rglob("*.vp_backup"):
            original = backup.with_suffix(".lean")
            if original.exists():
                shutil.copy2(backup, original)
                backup.unlink()
                logger.info(f"Recovered orphaned backup: {original}")
                restored.append(str(original))
            else:
                logger.warning(f"Orphaned backup has no original, removing: {backup}")
                backup.unlink()
        return restored

    async def warmup_file(self, file_path: str, timeout: float = 30.0) -> bool:
        """
        Warm up a file by requesting diagnostics.

        Ensures the LSP has fully processed the file before querying
        goals or running tactics. Prevents cold-start issues.

        Returns True if warmup succeeded.
        """
        # Recover any orphaned backups for this file before warming up
        backup = Path(file_path).with_suffix(".vp_backup")
        if backup.exists():
            shutil.copy2(backup, file_path)
            backup.unlink()
            logger.warning(f"Recovered orphaned backup before warmup: {file_path}")

        if file_path in self._warmed_files:
            logger.debug(f"File already warmed: {file_path}")
            return True

        logger.info(f"Warming up file: {file_path}")
        try:
            await asyncio.wait_for(self.get_diagnostics(file_path), timeout=timeout)
            self._warmed_files.add(file_path)
            logger.info(f"File warmup complete: {file_path}")
            return True
        except asyncio.TimeoutError:
            logger.warning(f"File warmup timed out after {timeout}s: {file_path}")
            return False
        except Exception as e:
            logger.warning(f"File warmup failed for {file_path}: {e}")
            return False

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
            # Apply per-tool timeout
            timeout = SAFE_TIMEOUT if name in SAFE_TOOLS else DANGEROUS_TIMEOUT
            try:
                result = await asyncio.wait_for(
                    self._session.call_tool(name, arguments),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.error(f"Tool '{name}' timed out after {timeout}s")
                self._successful_calls = 0
                raise MCPWorkerCrashError(f"Tool '{name}' timed out after {timeout}s")
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
            goals_before=join_goals(result.get("goals_before")),
            goals_after=join_goals(result.get("goals_after")),
            line_context=result.get("line_context"),
        )

    @staticmethod
    def _indent_tactic(target_line: str, tactic: str) -> str:
        """Replace 'sorry' in target_line with tactic, indenting continuation lines.

        For multi-line tactics, continuation lines inherit the indentation
        of the sorry token so that Lean's whitespace-sensitive parser
        sees them inside the same tactic block.
        """
        sorry_idx = target_line.find("sorry")
        if sorry_idx < 0:
            return target_line

        indent = target_line[:sorry_idx]

        if "\n" in tactic:
            tactic_lines = tactic.split("\n")
            indented = tactic_lines[0] + "\n" + "\n".join(
                (indent + ln) if ln.strip() else ln
                for ln in tactic_lines[1:]
            )
            return target_line.replace("sorry", indented, 1)

        return target_line.replace("sorry", tactic, 1)

    async def edit_file(
        self,
        file_path: str,
        line: int,
        tactic: str,
        column: Optional[int] = None,
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Edit a Lean file with a tactic and check if it makes progress.

        Implements Lean_MCP_pt2.md Section 4.1: safe edit-check-revert loop.

        Workflow:
        1. Backup file content
        2. Get baseline goal state
        3. Replace sorry at target line with tactic
        4. Wait for Lean to reprocess
        5. Check new goal state
        6. Revert if no progress or error

        Returns:
            (success, new_goal_state, error_msg)
        """
        path = Path(file_path)
        if not path.exists():
            return (False, None, f"File not found: {file_path}")

        # 1. Backup file to disk (survives process crashes)
        backup_path = path.with_suffix(".vp_backup")
        shutil.copy2(file_path, backup_path)
        self._file_backups[file_path] = backup_path
        original_content = path.read_text()

        try:
            # 2. Get baseline goal state
            baseline_goal = await self.get_goal(file_path, line, column)
            if baseline_goal is None:
                return (False, None, "No proof context at target position")

            # 3. Replace sorry with tactic (preserving indent)
            lines = original_content.splitlines(keepends=True)
            if line < 1 or line > len(lines):
                return (False, None, f"Line {line} out of range (file has {len(lines)} lines)")

            target_line = lines[line - 1]
            if "sorry" not in target_line.lower():
                return (False, None, f"No 'sorry' found on line {line}")

            new_line = self._indent_tactic(target_line, tactic)
            lines[line - 1] = new_line
            path.write_text("".join(lines))
            logger.debug(f"Edited {file_path}:{line} — replaced sorry with: {tactic}")

            # 4. Wait for Lean to reprocess
            await asyncio.sleep(0.5)

            # For multi-line tactics, check all inserted lines
            tactic_end_line = line + tactic.count('\n')

            # 5. Check new goal state (read from last tactic line)
            new_goal = await self.get_goal(
                file_path, tactic_end_line, column,
            )

            # Check for errors across all tactic lines
            diagnostics = await self.get_diagnostics(
                file_path, start_line=line,
                end_line=tactic_end_line,
            )
            errors = [d for d in diagnostics if d.severity == "error"]

            if errors:
                error_msg = "; ".join(e.message for e in errors[:3])
                logger.debug(f"Tactic caused errors: {error_msg}")
                await self.revert_file(file_path)
                return (False, None, f"Tactic caused errors: {error_msg}")

            if new_goal is None:
                await self.revert_file(file_path)
                return (False, None, "Lost proof context after tactic")

            if new_goal.is_complete:
                # Check full-file diagnostics before declaring success.
                # Some errors (e.g., @[progress] self-application causing
                # "fail to show termination") appear at the theorem declaration
                # line, not at the tactic line we edited.
                full_diagnostics = await self.get_diagnostics(file_path)
                full_errors = [d for d in full_diagnostics if d.severity == "error"]
                if full_errors:
                    error_msg = "; ".join(e.message for e in full_errors[:3])
                    logger.warning(f"Proof appears complete but file has errors: {error_msg}")
                    await self.revert_file(file_path)
                    return (False, None, f"Proof closed goals but caused errors: {error_msg}")
                logger.info(f"Tactic completed proof at {file_path}:{line}")
                return (True, "no goals", None)

            # Check if goals changed
            if new_goal.goals_after != baseline_goal.goals_after:
                logger.info(f"Tactic made progress at {file_path}:{line}")
                return (True, new_goal.goals_after, None)

            # No progress
            logger.debug(f"Tactic made no progress at {file_path}:{line}")
            await self.revert_file(file_path)
            return (False, None, "Tactic made no progress")

        except Exception as e:
            logger.error(f"Error during edit_file: {e}")
            await self.revert_file(file_path)
            return (False, None, f"Exception during edit: {e}")

    async def edit_file_with_capture(
        self,
        file_path: str,
        line: int,
        tactic: str,
        column: Optional[int] = None,
    ) -> tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """
        Like edit_file(), but also returns the modified file content.

        Returns:
            (success, new_goal_state, error_msg, modified_content)
            modified_content is the file text with the snippet inserted,
            captured before any revert. None only if the edit never happened
            (file not found, no sorry on line, etc.).
        """
        path = Path(file_path)
        if not path.exists():
            return (False, None, f"File not found: {file_path}", None)

        # 1. Backup file to disk (survives process crashes)
        backup_path = path.with_suffix(".vp_backup")
        shutil.copy2(file_path, backup_path)
        self._file_backups[file_path] = backup_path
        original_content = path.read_text()

        try:
            # 2. Get baseline goal state
            baseline_goal = await self.get_goal(file_path, line, column)
            if baseline_goal is None:
                return (
                    False, None,
                    "No proof context at target position", None,
                )

            # 3. Replace sorry with tactic (preserving indent)
            lines = original_content.splitlines(keepends=True)
            if line < 1 or line > len(lines):
                msg = (
                    f"Line {line} out of range "
                    f"(file has {len(lines)} lines)"
                )
                return (False, None, msg, None)

            target_line = lines[line - 1]
            if "sorry" not in target_line.lower():
                return (
                    False, None,
                    f"No 'sorry' found on line {line}", None,
                )

            new_line = self._indent_tactic(target_line, tactic)
            lines[line - 1] = new_line
            path.write_text("".join(lines))

            # Capture the modified content immediately after write
            modified_content = path.read_text()

            logger.debug(
                f"Edited {file_path}:{line} -- "
                f"replaced sorry with: {tactic}"
            )

            # 4. Wait for Lean to reprocess
            await asyncio.sleep(0.5)

            # For multi-line tactics, check all inserted lines
            tactic_end_line = line + tactic.count('\n')

            # 5. Check new goal state (read from last tactic line)
            new_goal = await self.get_goal(
                file_path, tactic_end_line, column,
            )

            # Check for errors across all tactic lines
            diagnostics = await self.get_diagnostics(
                file_path, start_line=line,
                end_line=tactic_end_line,
            )
            errors = [d for d in diagnostics if d.severity == "error"]

            if errors:
                error_msg = "; ".join(
                    e.message for e in errors[:3]
                )
                logger.debug(f"Tactic caused errors: {error_msg}")
                await self.revert_file(file_path)
                return (
                    False, None,
                    f"Tactic caused errors: {error_msg}",
                    modified_content,
                )

            if new_goal is None:
                await self.revert_file(file_path)
                return (
                    False, None,
                    "Lost proof context after tactic",
                    modified_content,
                )

            if new_goal.is_complete:
                # Check full-file diagnostics before success
                full_diags = await self.get_diagnostics(file_path)
                full_errors = [
                    d for d in full_diags if d.severity == "error"
                ]
                if full_errors:
                    error_msg = "; ".join(
                        e.message for e in full_errors[:3]
                    )
                    logger.warning(
                        "Proof appears complete but file has "
                        f"errors: {error_msg}"
                    )
                    await self.revert_file(file_path)
                    msg = (
                        "Proof closed goals but caused "
                        f"errors: {error_msg}"
                    )
                    return (False, None, msg, modified_content)
                logger.info(
                    f"Tactic completed proof at {file_path}:{line}"
                )
                return (True, "no goals", None, modified_content)

            # Check if goals changed
            if new_goal.goals_after != baseline_goal.goals_after:
                logger.info(
                    f"Tactic made progress at {file_path}:{line}"
                )
                return (
                    True, new_goal.goals_after,
                    None, modified_content,
                )

            # No progress
            logger.debug(
                f"Tactic made no progress at {file_path}:{line}"
            )
            await self.revert_file(file_path)
            return (
                False, None,
                "Tactic made no progress", modified_content,
            )

        except Exception as e:
            logger.error(f"Error during edit_file_with_capture: {e}")
            # Try to capture modified content if write happened
            try:
                mc = path.read_text()
                if mc == original_content:
                    mc = None
            except Exception:
                mc = None
            await self.revert_file(file_path)
            return (
                False, None,
                f"Exception during edit: {e}", mc,
            )

    async def revert_file(self, file_path: str) -> bool:
        """Revert a file from its disk-based backup (.vp_backup)."""
        backup_path = self._file_backups.get(file_path)
        if not backup_path or not backup_path.exists():
            # Fallback: check for backup file on disk even if not tracked
            fallback = Path(file_path).with_suffix(".vp_backup")
            if fallback.exists():
                backup_path = fallback
            else:
                logger.warning(f"No backup found for {file_path}")
                return False

        try:
            shutil.copy2(backup_path, file_path)
            backup_path.unlink()  # Clean up backup
            self._file_backups.pop(file_path, None)
            logger.debug(f"Reverted {file_path} from disk backup")
            await asyncio.sleep(0.5)  # Let Lean reprocess
            return True
        except Exception as e:
            logger.error(f"Failed to revert {file_path}: {e}")
            return False

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
        logger.info(f"multi_attempt at {file_path}:{line} with {len(snippets)} snippets")
        for i, snippet in enumerate(snippets, 1):
            preview = snippet[:100] + "..." if len(snippet) > 100 else snippet
            logger.debug(f"  Snippet {i}: {preview}")

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
