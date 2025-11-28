"""
Embedding utilities using BGE models.

This module provides local embedding capabilities using
BAAI BGE models via sentence-transformers.
"""
from __future__ import annotations

import os
from typing import Optional

from rich.console import Console

from interfaces.rag_provider import EmbeddingProvider as EmbeddingProviderABC

console = Console()


class BGEEmbeddings(EmbeddingProviderABC):
    """
    BGE (BAAI General Embedding) model for text embeddings.

    Uses sentence-transformers for efficient local inference.
    Default model: BAAI/bge-small-en-v1.5 (~130MB, 384 dims)
    """

    # Model configurations
    MODELS = {
        "BAAI/bge-small-en-v1.5": {"dimension": 384, "size_mb": 130},
        "BAAI/bge-base-en-v1.5": {"dimension": 768, "size_mb": 440},
        "BAAI/bge-large-en-v1.5": {"dimension": 1024, "size_mb": 1340},
        "BAAI/bge-m3": {"dimension": 1024, "size_mb": 2200},
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        Initialize BGE embeddings.

        Args:
            model_name: Model identifier. Defaults to EMBEDDING_MODEL env var
                       or 'BAAI/bge-small-en-v1.5'
            device: Device to use ('cpu', 'cuda', 'mps'). Auto-detected if None.
            normalize: Whether to normalize embeddings (recommended for BGE)
        """
        self._model_name = model_name or os.getenv(
            "EMBEDDING_MODEL",
            "BAAI/bge-small-en-v1.5"
        )
        self._device = device
        self._normalize = normalize
        self._model = None

        # Validate model name
        if self._model_name not in self.MODELS:
            console.print(f"[yellow]Warning: Unknown model {self._model_name}[/yellow]")
            console.print(f"[yellow]Known models: {list(self.MODELS.keys())}[/yellow]")

        self._dimension = self.MODELS.get(
            self._model_name,
            {"dimension": 384}
        )["dimension"]

    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        console.print(f"[blue]Loading embedding model: {self._model_name}[/blue]")

        self._model = SentenceTransformer(
            self._model_name,
            device=self._device,
        )

        # Print device info
        device = str(self._model.device)
        console.print(f"[green]Model loaded on device: {device}[/green]")

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        self._load_model()

        # BGE models benefit from a query prefix for retrieval
        # For documents, we don't use a prefix
        embedding = self._model.encode(
            text,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )

        return embedding.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        self._load_model()

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=self._normalize,
            show_progress_bar=len(texts) > 100,
            batch_size=32,
        )

        return embeddings.tolist()

    async def embed_query(self, query: str) -> list[float]:
        """
        Embed a query with the appropriate prefix.

        BGE models perform better when queries use the prefix:
        "Represent this sentence for searching relevant passages:"
        """
        self._load_model()

        # Add query prefix for BGE models
        prefixed = f"Represent this sentence for searching relevant passages: {query}"

        embedding = self._model.encode(
            prefixed,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )

        return embedding.tolist()

    def embed_sync(self, text: str) -> list[float]:
        """Synchronous version of embed."""
        self._load_model()

        embedding = self._model.encode(
            text,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )

        return embedding.tolist()

    def embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous version of embed_batch."""
        if not texts:
            return []

        self._load_model()

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=self._normalize,
            show_progress_bar=len(texts) > 100,
            batch_size=32,
        )

        return embeddings.tolist()


def get_embeddings(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> BGEEmbeddings:
    """
    Factory function to get an embedding provider.

    Args:
        model_name: Model to use. Defaults to env var or bge-small
        device: Device to use. Auto-detected if None.

    Returns:
        Configured embedding provider
    """
    return BGEEmbeddings(model_name=model_name, device=device)


def create_embedding_text(
    name: str,
    type_signature: str,
    proof: Optional[str] = None,
    doc_string: Optional[str] = None,
    tactics: Optional[list[str]] = None,
    max_proof_chars: int = 500,
) -> str:
    """
    Create text for embedding from theorem components.

    Combines name, type signature, docstring, and proof snippet
    into a single text suitable for embedding.

    Args:
        name: Theorem/lemma name
        type_signature: Type signature
        proof: Full proof text (will be truncated)
        doc_string: Documentation string
        tactics: List of tactic names used
        max_proof_chars: Maximum characters from proof to include

    Returns:
        Combined text for embedding
    """
    parts = [
        f"Name: {name}",
        f"Type: {type_signature}",
    ]

    if doc_string:
        parts.append(f"Description: {doc_string}")

    if proof:
        proof_snippet = proof[:max_proof_chars]
        if len(proof) > max_proof_chars:
            proof_snippet += "..."
        parts.append(f"Proof: {proof_snippet}")

    if tactics:
        parts.append(f"Tactics: {', '.join(tactics[:10])}")

    return "\n".join(parts)
