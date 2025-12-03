"""
Cascading retrieval across multiple backends.

Implements the four-level cascade:
1. Type Index (DuckDB) - <100ms
2. BM25 Keyword (DuckDB FTS) - <500ms
3. Semantic Embeddings (Qdrant) - <2s
4. Graph Traversal (Neo4j) - <5s
"""
from __future__ import annotations

import asyncio
import os
from typing import Optional

from rich.console import Console

from interfaces.rag_provider import (
    RAGProvider,
    RetrievalQuery,
    RetrievalResult,
    RetrievalLevel,
)

console = Console()


class CascadingRetriever(RAGProvider):
    """
    Cascading retriever with timeout handling and early termination.

    Queries backends from fastest to slowest, stopping early
    when sufficient high-confidence results are found.
    """

    def __init__(
        self,
        type_index=None,
        qdrant_client=None,
        neo4j_graph=None,
        embedder=None,
        config: Optional[dict] = None,
    ):
        self._type_index = type_index
        self._qdrant = qdrant_client
        self._neo4j = neo4j_graph
        self._embedder = embedder
        self._config = config or {}
        self._collection_name = config.get("collection_name", "lean_proofs") if config else "lean_proofs"

        # Timeout configuration (in seconds)
        self._timeouts = {
            RetrievalLevel.TYPE_INDEX: 0.1,
            RetrievalLevel.BM25: 0.5,
            RetrievalLevel.SEMANTIC: 2.0,
            RetrievalLevel.GRAPH: 5.0,
        }

        # Weights for score combination
        self._weights = {
            RetrievalLevel.TYPE_INDEX: 1.0,
            RetrievalLevel.BM25: 0.9,
            RetrievalLevel.SEMANTIC: 0.8,
            RetrievalLevel.GRAPH: 0.7,
        }

    @property
    def language(self) -> str:
        return "lean"

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Execute cascading retrieval with early termination."""
        results = []
        seen_names = set()

        # Level 1: Type Index (DuckDB) - <100ms
        if RetrievalLevel.TYPE_INDEX in query.include_levels and self._type_index:
            try:
                type_results = await asyncio.wait_for(
                    self._query_type_index(query.text),
                    timeout=self._timeouts[RetrievalLevel.TYPE_INDEX],
                )
                for r in type_results:
                    if r.full_name not in seen_names:
                        seen_names.add(r.full_name)
                        results.append(r)
            except asyncio.TimeoutError:
                pass

        # Early termination check
        if len(results) >= query.top_k:
            return self._rank_and_limit(results, query.top_k)

        # Level 2: BM25 Keyword (DuckDB FTS) - <500ms
        if RetrievalLevel.BM25 in query.include_levels and self._type_index:
            try:
                bm25_results = await asyncio.wait_for(
                    self._query_bm25(query.text),
                    timeout=self._timeouts[RetrievalLevel.BM25],
                )
                for r in bm25_results:
                    if r.full_name not in seen_names:
                        seen_names.add(r.full_name)
                        results.append(r)
            except asyncio.TimeoutError:
                pass

        # Level 3: Semantic Embeddings (Qdrant) - <2s
        if RetrievalLevel.SEMANTIC in query.include_levels and self._qdrant and self._embedder:
            try:
                semantic_results = await asyncio.wait_for(
                    self._query_semantic(query.text),
                    timeout=self._timeouts[RetrievalLevel.SEMANTIC],
                )
                for r in semantic_results:
                    if r.full_name not in seen_names:
                        seen_names.add(r.full_name)
                        results.append(r)
            except asyncio.TimeoutError:
                pass

        # Level 4: Graph Traversal (Neo4j) - <5s
        if len(results) < query.top_k and RetrievalLevel.GRAPH in query.include_levels and self._neo4j:
            try:
                # Use existing results to seed graph search
                seed_names = [r.full_name for r in results[:3]]
                graph_results = await asyncio.wait_for(
                    self._query_graph(seed_names),
                    timeout=self._timeouts[RetrievalLevel.GRAPH],
                )
                for r in graph_results:
                    if r.full_name not in seen_names:
                        seen_names.add(r.full_name)
                        results.append(r)
            except asyncio.TimeoutError:
                pass

        return self._rank_and_limit(results, query.top_k)

    async def retrieve_by_name(self, name: str) -> Optional[RetrievalResult]:
        """Retrieve a specific declaration by name."""
        if self._type_index:
            results = await self._type_index.query_by_name(name, limit=1)
            return results[0] if results else None
        return None

    async def retrieve_similar(self, reference: RetrievalResult, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve declarations similar to a reference."""
        if self._neo4j:
            return await self._neo4j.query_similar(reference.full_name, limit=top_k)
        return []

    async def retrieve_dependencies(self, name: str, max_depth: int = 2) -> list[RetrievalResult]:
        """Retrieve dependencies of a declaration."""
        if self._neo4j:
            return await self._neo4j.query_dependencies(name, max_depth=max_depth)
        return []

    async def close(self) -> None:
        """Close all backend connections."""
        if self._type_index:
            self._type_index.close()
        if self._neo4j:
            self._neo4j.close()
        if self._qdrant:
            self._qdrant.close()

    # Private query methods

    async def _query_type_index(self, query: str) -> list[RetrievalResult]:
        """Query DuckDB type index."""
        return await self._type_index.query(query, limit=10)

    async def _query_bm25(self, query: str) -> list[RetrievalResult]:
        """Query DuckDB FTS for BM25-style keyword search."""
        # Use the type_index's BM25 method (DuckDB FTS)
        results = await self._type_index.query_bm25(query, limit=10)
        # Ensure level is set correctly
        for r in results:
            r.level = RetrievalLevel.BM25
        return results

    async def _query_semantic(self, query: str) -> list[RetrievalResult]:
        """Query Qdrant with semantic embeddings."""
        # Compute query embedding
        query_embedding = await self._embedder.embed_query(query)

        # Search in Qdrant using query_points (v1.16.1 API)
        search_results = self._qdrant.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            limit=10,
            with_payload=True,
        )

        results = []
        for hit in search_results.points:
            payload = hit.payload or {}
            score = hit.score  # Qdrant returns similarity score directly

            results.append(RetrievalResult(
                name=payload.get("declaration_name", ""),
                full_name=payload.get("full_name", ""),
                type_signature=payload.get("type_signature", ""),
                namespace=payload.get("namespace", ""),
                file_path=payload.get("file_path", ""),
                doc_string=payload.get("doc_string"),
                proof=payload.get("proof_preview"),
                score=score,
                level=RetrievalLevel.SEMANTIC,
            ))

        return results

    async def _query_graph(self, seed_names: list[str]) -> list[RetrievalResult]:
        """Query Neo4j for related declarations."""
        if not seed_names:
            return []

        results = []
        for seed in seed_names[:3]:  # Limit seeds
            deps = await self._neo4j.query_dependencies(seed, max_depth=2)
            results.extend(deps)

            similar = await self._neo4j.query_similar(seed, limit=3)
            results.extend(similar)

        return results

    def _rank_and_limit(self, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        """Rank results by weighted score and limit."""
        # Apply level weights
        for r in results:
            r.score *= self._weights.get(r.level, 0.5)

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:top_k]
