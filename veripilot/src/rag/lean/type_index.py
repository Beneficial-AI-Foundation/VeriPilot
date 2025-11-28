"""
DuckDB type index for fast Lean declaration lookup.

This module provides <100ms queries for exact name and
type signature matching using DuckDB.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import duckdb
from rich.console import Console

from interfaces.rag_provider import Indexer, RetrievalResult, RetrievalLevel

console = Console()


class LeanTypeIndex(Indexer):
    """
    DuckDB-based type index for Lean declarations.

    Provides fast (<100ms) lookups by:
    - Exact name match
    - Full name match (with namespace)
    - Type signature pattern matching
    - Declaration type filtering
    """

    TABLE_NAME = "lean_declarations"

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the type index.

        Args:
            db_path: Path to DuckDB database file. Defaults to DUCKDB_PATH env var.
        """
        self.db_path = db_path or os.getenv(
            "DUCKDB_PATH",
            "/workspace/data/indices/lean/types.duckdb"
        )

        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    def _get_conn(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = duckdb.connect(self.db_path)
            self._ensure_schema()
        return self._conn

    def _ensure_schema(self) -> None:
        """Create table and indexes if they don't exist."""
        conn = self._conn
        if conn is None:
            return

        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                full_name TEXT NOT NULL,
                decl_type TEXT NOT NULL,
                type_signature TEXT NOT NULL,
                namespace TEXT,
                file_path TEXT NOT NULL,
                line_start INTEGER,
                line_end INTEGER,
                doc_string TEXT,
                proof_preview TEXT,
                premises TEXT,
                tactics TEXT,
                embedding_id INTEGER
            )
        """)

        # Create indexes for fast lookup
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_name
            ON {self.TABLE_NAME}(name)
        """)
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_full_name
            ON {self.TABLE_NAME}(full_name)
        """)
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_decl_type
            ON {self.TABLE_NAME}(decl_type)
        """)
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_namespace
            ON {self.TABLE_NAME}(namespace)
        """)

        console.print(f"[green]DuckDB schema ready: {self.db_path}[/green]")

    async def index(self, data: list[dict]) -> int:
        """
        Index declarations into DuckDB.

        Args:
            data: List of declaration dictionaries with keys:
                  name, full_name, decl_type, type_signature, namespace,
                  file_path, line_start, line_end, doc_string, proof_preview,
                  premises, tactics

        Returns:
            Number of items indexed
        """
        conn = self._get_conn()

        indexed = 0
        for i, item in enumerate(data):
            conn.execute(f"""
                INSERT OR REPLACE INTO {self.TABLE_NAME}
                (id, name, full_name, decl_type, type_signature, namespace,
                 file_path, line_start, line_end, doc_string, proof_preview,
                 premises, tactics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                i,
                item.get("name", ""),
                item.get("full_name", ""),
                item.get("decl_type", "theorem"),
                item.get("type_signature", ""),
                item.get("namespace", ""),
                item.get("file_path", ""),
                item.get("line_start", 0),
                item.get("line_end", 0),
                item.get("doc_string"),
                item.get("proof_preview"),
                ",".join(item.get("premises", [])),
                ",".join(item.get("tactics", [])),
            ))
            indexed += 1

        conn.commit()
        return indexed

    async def clear(self) -> None:
        """Clear all indexed data."""
        conn = self._get_conn()
        conn.execute(f"DELETE FROM {self.TABLE_NAME}")
        conn.commit()
        console.print("[yellow]Type index cleared[/yellow]")

    async def count(self) -> int:
        """Return the number of indexed items."""
        conn = self._get_conn()
        result = conn.execute(f"SELECT COUNT(*) FROM {self.TABLE_NAME}").fetchone()
        return result[0] if result else 0

    async def query_by_name(
        self,
        name: str,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        """
        Query by exact or partial name match.

        Args:
            name: Name to search for
            limit: Maximum results

        Returns:
            List of matching declarations
        """
        conn = self._get_conn()

        # Try exact match first, then partial
        rows = conn.execute(f"""
            SELECT name, full_name, type_signature, namespace, file_path,
                   line_start, line_end, doc_string, proof_preview, decl_type
            FROM {self.TABLE_NAME}
            WHERE name = ? OR full_name = ?
            LIMIT ?
        """, (name, name, limit)).fetchall()

        if not rows:
            # Try partial match
            rows = conn.execute(f"""
                SELECT name, full_name, type_signature, namespace, file_path,
                       line_start, line_end, doc_string, proof_preview, decl_type
                FROM {self.TABLE_NAME}
                WHERE name LIKE ? OR full_name LIKE ?
                LIMIT ?
            """, (f"%{name}%", f"%{name}%", limit)).fetchall()

        return [self._row_to_result(row, score=1.0) for row in rows]

    async def query_by_signature(
        self,
        pattern: str,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        """
        Query by type signature pattern.

        Args:
            pattern: Type signature pattern to match
            limit: Maximum results

        Returns:
            List of matching declarations
        """
        conn = self._get_conn()

        rows = conn.execute(f"""
            SELECT name, full_name, type_signature, namespace, file_path,
                   line_start, line_end, doc_string, proof_preview, decl_type
            FROM {self.TABLE_NAME}
            WHERE type_signature LIKE ?
            LIMIT ?
        """, (f"%{pattern}%", limit)).fetchall()

        return [self._row_to_result(row, score=0.9) for row in rows]

    async def query_by_namespace(
        self,
        namespace: str,
        limit: int = 50,
    ) -> list[RetrievalResult]:
        """
        Query by namespace prefix.

        Args:
            namespace: Namespace prefix to match
            limit: Maximum results

        Returns:
            List of declarations in the namespace
        """
        conn = self._get_conn()

        rows = conn.execute(f"""
            SELECT name, full_name, type_signature, namespace, file_path,
                   line_start, line_end, doc_string, proof_preview, decl_type
            FROM {self.TABLE_NAME}
            WHERE namespace LIKE ?
            ORDER BY name
            LIMIT ?
        """, (f"{namespace}%", limit)).fetchall()

        return [self._row_to_result(row, score=0.8) for row in rows]

    async def query(
        self,
        query: str,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        """
        General query - searches name and signature.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching declarations
        """
        conn = self._get_conn()

        # Search in name, full_name, and type_signature
        rows = conn.execute(f"""
            SELECT name, full_name, type_signature, namespace, file_path,
                   line_start, line_end, doc_string, proof_preview, decl_type
            FROM {self.TABLE_NAME}
            WHERE name LIKE ?
               OR full_name LIKE ?
               OR type_signature LIKE ?
            LIMIT ?
        """, (f"%{query}%", f"%{query}%", f"%{query}%", limit)).fetchall()

        return [self._row_to_result(row, score=0.85) for row in rows]

    def _row_to_result(
        self,
        row: tuple,
        score: float = 1.0,
    ) -> RetrievalResult:
        """Convert a database row to a RetrievalResult."""
        return RetrievalResult(
            name=row[0],
            full_name=row[1],
            type_signature=row[2],
            namespace=row[3] or "",
            file_path=row[4],
            line_start=row[5] or 0,
            line_end=row[6] or 0,
            doc_string=row[7],
            proof=row[8],  # proof_preview
            score=score,
            level=RetrievalLevel.TYPE_INDEX,
            metadata={"decl_type": row[9]} if len(row) > 9 else {},
        )

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # Synchronous versions for non-async contexts

    def index_sync(self, data: list[dict]) -> int:
        """Synchronous version of index."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.index(data))

    def query_sync(self, query: str, limit: int = 10) -> list[RetrievalResult]:
        """Synchronous version of query."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.query(query, limit))
