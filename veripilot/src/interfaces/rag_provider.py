"""
Abstract interfaces for RAG providers.

This module defines the abstract base classes for RAG components,
ensuring a consistent interface across Lean and Verus implementations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any


class RetrievalLevel(Enum):
    """Retrieval cascade levels in order of speed."""
    TYPE_INDEX = "type_index"      # <100ms - DuckDB exact match
    BM25 = "bm25"                  # <500ms - Weaviate keyword
    SEMANTIC = "semantic"          # <2s - Weaviate embeddings
    GRAPH = "graph"                # <5s - Neo4j traversal


@dataclass
class RetrievalResult:
    """A single retrieval result from the RAG system."""
    # Identity
    name: str
    full_name: str

    # Content
    type_signature: str
    proof: Optional[str] = None
    doc_string: Optional[str] = None

    # Location
    file_path: str = ""
    line_start: int = 0
    line_end: int = 0
    namespace: str = ""

    # Retrieval metadata
    score: float = 0.0
    level: RetrievalLevel = RetrievalLevel.SEMANTIC
    relevance_explanation: Optional[str] = None

    # Additional data (tactics, premises, etc.)
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.full_name}: {self.type_signature}"


@dataclass
class RetrievalQuery:
    """A query to the RAG system."""
    text: str
    top_k: int = 6
    include_levels: list[RetrievalLevel] = field(
        default_factory=lambda: list(RetrievalLevel)
    )
    filters: dict[str, Any] = field(default_factory=dict)
    timeout_ms: int = 2000

    # For type-aware retrieval
    type_hint: Optional[str] = None
    # For goal-based retrieval
    goal_state: Optional[str] = None


class RAGProvider(ABC):
    """
    Abstract base class for RAG providers.

    Implementations should provide retrieval capabilities for
    a specific language (Lean or Verus).
    """

    @property
    @abstractmethod
    def language(self) -> str:
        """Return the language this provider handles ('lean' or 'verus')."""
        pass

    @abstractmethod
    async def retrieve(
        self,
        query: RetrievalQuery,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The retrieval query

        Returns:
            List of retrieval results, ordered by relevance
        """
        pass

    @abstractmethod
    async def retrieve_by_name(
        self,
        name: str,
    ) -> Optional[RetrievalResult]:
        """
        Retrieve a specific declaration by name.

        Args:
            name: The full name of the declaration

        Returns:
            The declaration if found, None otherwise
        """
        pass

    @abstractmethod
    async def retrieve_similar(
        self,
        reference: RetrievalResult,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        Retrieve declarations similar to a reference.

        Args:
            reference: A reference declaration
            top_k: Number of results to return

        Returns:
            List of similar declarations
        """
        pass

    @abstractmethod
    async def retrieve_dependencies(
        self,
        name: str,
        max_depth: int = 2,
    ) -> list[RetrievalResult]:
        """
        Retrieve dependencies of a declaration.

        Args:
            name: The full name of the declaration
            max_depth: Maximum depth of dependency traversal

        Returns:
            List of dependencies
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close connections and clean up resources."""
        pass


class Indexer(ABC):
    """
    Abstract base class for indexers.

    Implementations handle indexing extracted data into
    the various backends (DuckDB, Weaviate, Neo4j).
    """

    @abstractmethod
    async def index(self, data: Any) -> int:
        """
        Index data into the backend.

        Args:
            data: The data to index (type depends on implementation)

        Returns:
            Number of items indexed
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all indexed data."""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Return the number of indexed items."""
        pass


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    Implementations provide text embedding capabilities
    using various models (BGE, OpenAI, etc.).
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            Embedding vector
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass
