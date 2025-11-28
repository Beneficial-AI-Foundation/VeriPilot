"""
Neo4j knowledge graph for Lean declarations.

This module provides graph-based retrieval for:
- Dependency relationships (DEPENDS_ON)
- Tactic usage patterns (USES_TACTIC)
- Namespace organization (IN_NAMESPACE)
- Similarity relationships (SIMILAR_TO)
"""
from __future__ import annotations

import os
from typing import Optional

from rich.console import Console

from interfaces.rag_provider import Indexer, RetrievalResult, RetrievalLevel

console = Console()


class LeanKnowledgeGraph(Indexer):
    """
    Neo4j-based knowledge graph for Lean declarations.

    Stores and retrieves:
    - Theorems, lemmas, definitions as nodes
    - Dependencies between declarations
    - Tactic usage patterns
    - Namespace hierarchy
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "")

        if not self.uri:
            console.print("[yellow]Warning: NEO4J_URI not set[/yellow]")

        self._driver = None

    def _get_driver(self):
        """Get or create Neo4j driver."""
        if self._driver is None:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            console.print(f"[green]Connected to Neo4j[/green]")
        return self._driver

    async def index(self, data: list[dict]) -> int:
        """Index declarations into Neo4j."""
        driver = self._get_driver()
        indexed = 0

        with driver.session() as session:
            for item in data:
                node_type = self._get_node_type(item.get("decl_type", "theorem"))
                full_name = item.get("full_name", item.get("name", ""))

                session.run(f"""
                    MERGE (d:{node_type} {{full_name: $full_name}})
                    SET d.name = $name,
                        d.type_signature = $type_signature,
                        d.namespace = $namespace,
                        d.file_path = $file_path
                """, {
                    "full_name": full_name,
                    "name": item.get("name", ""),
                    "type_signature": item.get("type_signature", ""),
                    "namespace": item.get("namespace", ""),
                    "file_path": item.get("file_path", ""),
                })
                indexed += 1

                # Create dependency relationships
                for premise in item.get("premises", []):
                    session.run(f"""
                        MATCH (d1:{node_type} {{full_name: $from_name}})
                        MERGE (d2:Declaration {{full_name: $to_name}})
                        MERGE (d1)-[:DEPENDS_ON]->(d2)
                    """, {"from_name": full_name, "to_name": premise})

        return indexed

    async def clear(self) -> None:
        driver = self._get_driver()
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    async def count(self) -> int:
        driver = self._get_driver()
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN COUNT(n) as count")
            record = result.single()
            return record["count"] if record else 0

    async def query_dependencies(self, full_name: str, max_depth: int = 3) -> list[RetrievalResult]:
        """Query dependencies of a declaration."""
        driver = self._get_driver()
        results = []

        with driver.session() as session:
            records = session.run("""
                MATCH path = (d)-[:DEPENDS_ON*1..3]->(dep)
                WHERE d.full_name = $full_name
                RETURN DISTINCT dep.name AS name, dep.full_name AS full_name,
                       dep.type_signature AS type_signature, dep.namespace AS namespace,
                       dep.file_path AS file_path, length(path) AS depth
                LIMIT 20
            """, {"full_name": full_name})

            for record in records:
                results.append(RetrievalResult(
                    name=record["name"] or "",
                    full_name=record["full_name"] or "",
                    type_signature=record["type_signature"] or "",
                    namespace=record["namespace"] or "",
                    file_path=record["file_path"] or "",
                    score=1.0 / (record["depth"] + 1),
                    level=RetrievalLevel.GRAPH,
                ))

        return results

    async def query_similar(self, full_name: str, limit: int = 5) -> list[RetrievalResult]:
        """Query similar declarations by shared dependencies."""
        driver = self._get_driver()
        results = []

        with driver.session() as session:
            records = session.run("""
                MATCH (d1)-[:DEPENDS_ON]-(shared)-[:DEPENDS_ON]-(d2)
                WHERE d1.full_name = $full_name AND d1 <> d2
                WITH d2, COUNT(DISTINCT shared) AS shared_count
                ORDER BY shared_count DESC LIMIT $limit
                RETURN d2.name AS name, d2.full_name AS full_name,
                       d2.type_signature AS type_signature, d2.namespace AS namespace,
                       d2.file_path AS file_path, shared_count
            """, {"full_name": full_name, "limit": limit})

            for record in records:
                results.append(RetrievalResult(
                    name=record["name"] or "",
                    full_name=record["full_name"] or "",
                    type_signature=record["type_signature"] or "",
                    namespace=record["namespace"] or "",
                    file_path=record["file_path"] or "",
                    score=min(record["shared_count"] / 10, 1.0),
                    level=RetrievalLevel.GRAPH,
                ))

        return results

    def _get_node_type(self, decl_type: str) -> str:
        mapping = {
            "theorem": "Theorem", "lemma": "Lemma", "definition": "Definition",
            "def": "Definition", "structure": "Structure",
        }
        return mapping.get(decl_type.lower(), "Declaration")

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None
