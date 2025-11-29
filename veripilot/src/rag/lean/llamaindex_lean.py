"""
LlamaIndex integration for Lean RAG.

Provides a LlamaIndex-compatible retriever that uses our
cascading retrieval system under the hood.

Backend stack:
- DuckDB: Type index (L1) + BM25/FTS (L2)
- Qdrant: Semantic search (L3)
- Neo4j: Graph traversal (L4)
"""
from __future__ import annotations

import os
from typing import Optional

from rich.console import Console

from interfaces.rag_provider import RetrievalQuery, RetrievalResult

console = Console()


class LeanRAG:
    """
    Main Lean RAG interface for VeriPilot.

    Provides a unified interface for:
    - Initializing all backends (DuckDB, Qdrant, Neo4j)
    - Running cascading retrieval queries
    - LlamaIndex integration
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Lean RAG with all backends.

        Args:
            config_path: Path to lean_rag.yaml config file
        """
        self._config = self._load_config(config_path)
        self._type_index = None
        self._qdrant = None
        self._neo4j = None
        self._embedder = None
        self._retriever = None

    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from YAML file."""
        if config_path is None:
            # Default path
            from pathlib import Path
            config_path = Path(__file__).parent.parent.parent.parent / "config" / "lean_rag.yaml"

        if not os.path.exists(config_path):
            console.print(f"[yellow]Config not found: {config_path}[/yellow]")
            return {}

        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)

    async def initialize(self) -> None:
        """Initialize all backend connections."""
        from dotenv import load_dotenv
        load_dotenv()

        console.print("[blue]Initializing Lean RAG backends...[/blue]")

        # Initialize DuckDB type index
        try:
            from rag.lean.type_index import LeanTypeIndex
            self._type_index = LeanTypeIndex()
            count = await self._type_index.count()
            console.print(f"[green]  DuckDB: {count} declarations[/green]")
        except Exception as e:
            console.print(f"[yellow]  DuckDB init failed: {e}[/yellow]")

        # Initialize embeddings
        try:
            from rag.shared.embeddings import BGEEmbeddings
            self._embedder = BGEEmbeddings()
            console.print(f"[green]  Embeddings: {self._embedder.model_name}[/green]")
        except Exception as e:
            console.print(f"[yellow]  Embeddings init failed: {e}[/yellow]")

        # Initialize Qdrant
        try:
            from qdrant_client import QdrantClient

            url = os.getenv("QDRANT_URL")
            api_key = os.getenv("QDRANT_API_KEY")

            if url and api_key:
                self._qdrant = QdrantClient(
                    url=url,
                    api_key=api_key,
                )
                # Check connection by listing collections
                collections = self._qdrant.get_collections()
                console.print(f"[green]  Qdrant: connected ({len(collections.collections)} collections)[/green]")
            else:
                console.print("[yellow]  Qdrant: credentials not set[/yellow]")
        except Exception as e:
            console.print(f"[yellow]  Qdrant init failed: {e}[/yellow]")

        # Initialize Neo4j
        try:
            from rag.lean.graph_rag_lean import LeanKnowledgeGraph
            uri = os.getenv("NEO4J_URI")
            if uri:
                self._neo4j = LeanKnowledgeGraph()
                count = await self._neo4j.count()
                console.print(f"[green]  Neo4j: {count} nodes[/green]")
            else:
                console.print("[yellow]  Neo4j: URI not set[/yellow]")
        except Exception as e:
            console.print(f"[yellow]  Neo4j init failed: {e}[/yellow]")

        # Initialize cascading retriever
        from rag.shared.cascading_retriever import CascadingRetriever
        self._retriever = CascadingRetriever(
            type_index=self._type_index,
            qdrant_client=self._qdrant,
            neo4j_graph=self._neo4j,
            embedder=self._embedder,
            config=self._config,
        )

        console.print("[green]Lean RAG initialized[/green]")

    async def retrieve(
        self,
        query: str,
        top_k: int = 6,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant Lean declarations for a query.

        Args:
            query: Natural language or type-based query
            top_k: Number of results to return

        Returns:
            List of retrieval results
        """
        if self._retriever is None:
            await self.initialize()

        retrieval_query = RetrievalQuery(text=query, top_k=top_k)
        return await self._retriever.retrieve(retrieval_query)

    async def retrieve_by_name(self, name: str) -> Optional[RetrievalResult]:
        """Retrieve a specific declaration by name."""
        if self._retriever is None:
            await self.initialize()
        return await self._retriever.retrieve_by_name(name)

    async def retrieve_dependencies(
        self,
        name: str,
        max_depth: int = 2,
    ) -> list[RetrievalResult]:
        """Retrieve dependencies of a declaration."""
        if self._retriever is None:
            await self.initialize()
        return await self._retriever.retrieve_dependencies(name, max_depth)

    async def retrieve_similar(
        self,
        name: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve declarations similar to the named one."""
        if self._retriever is None:
            await self.initialize()

        ref = await self.retrieve_by_name(name)
        if ref is None:
            return []
        return await self._retriever.retrieve_similar(ref, top_k)

    def get_llamaindex_retriever(self):
        """
        Get a LlamaIndex-compatible retriever.

        Returns a retriever that can be used with LlamaIndex query engines.
        """
        try:
            from llama_index.core.retrievers import BaseRetriever
            from llama_index.core.schema import NodeWithScore, TextNode
        except ImportError:
            raise ImportError(
                "llama-index not installed. Install with: pip install llama-index"
            )

        rag = self

        class LeanLlamaIndexRetriever(BaseRetriever):
            """LlamaIndex-compatible retriever wrapper."""

            async def _aretrieve(self, query_bundle) -> list[NodeWithScore]:
                results = await rag.retrieve(query_bundle.query_str)

                nodes = []
                for r in results:
                    content = self._format_result(r)
                    node = TextNode(
                        text=content,
                        metadata={
                            "name": r.name,
                            "full_name": r.full_name,
                            "type_signature": r.type_signature,
                            "namespace": r.namespace,
                            "file_path": r.file_path,
                            "retrieval_level": r.level.value,
                        },
                    )
                    nodes.append(NodeWithScore(node=node, score=r.score))

                return nodes

            def _retrieve(self, query_bundle) -> list[NodeWithScore]:
                import asyncio
                return asyncio.run(self._aretrieve(query_bundle))

            def _format_result(self, r: RetrievalResult) -> str:
                parts = [
                    f"**{r.name}**",
                    f"Namespace: `{r.namespace}`" if r.namespace else "",
                    f"Type: `{r.type_signature}`",
                    f"File: {r.file_path}",
                ]
                if r.doc_string:
                    parts.append(f"Description: {r.doc_string}")
                if r.proof:
                    parts.append(f"Proof:\n```lean\n{r.proof[:300]}\n```")
                return "\n".join(p for p in parts if p)

        return LeanLlamaIndexRetriever()

    async def close(self) -> None:
        """Close all backend connections."""
        if self._retriever:
            await self._retriever.close()


# Convenience function for quick usage
async def create_lean_rag(config_path: Optional[str] = None) -> LeanRAG:
    """Create and initialize a LeanRAG instance."""
    rag = LeanRAG(config_path)
    await rag.initialize()
    return rag
