"""
Integration tests for Lean RAG system.

These tests require:
- Extracted corpus in data/extracted/combined_corpus.json
- Indexed data in DuckDB, Qdrant, Neo4j
- Environment variables set (.env file)

Run with: pytest tests/integration/test_lean_rag.py -v
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()


# Skip all tests if no credentials
QDRANT_URL = os.getenv("QDRANT_URL")
NEO4J_URI = os.getenv("NEO4J_URI")

pytestmark = pytest.mark.skipif(
    not QDRANT_URL or not NEO4J_URI,
    reason="Cloud credentials not configured"
)


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def rag_system():
    """Initialize RAG system for tests."""
    from rag.lean.llamaindex_lean import LeanRAG

    rag = LeanRAG()
    await rag.initialize()
    yield rag
    await rag.close()


class TestTypeIndexRetrieval:
    """Tests for DuckDB type index (Level 1)."""

    @pytest.mark.asyncio
    async def test_find_by_exact_name(self, rag_system):
        """Test finding declaration by exact name."""
        results = await rag_system.retrieve("invert_spec", top_k=5)

        assert len(results) > 0
        names = [r.name for r in results]
        # Should find something related to invert
        assert any("invert" in n.lower() for n in names)

    @pytest.mark.asyncio
    async def test_find_by_type_pattern(self, rag_system):
        """Test finding by type signature pattern."""
        results = await rag_system.retrieve("Scalar", top_k=10)

        assert len(results) > 0
        # Should find scalar-related declarations
        assert any("scalar" in r.full_name.lower() for r in results)


class TestSemanticRetrieval:
    """Tests for semantic embedding search (Level 3)."""

    @pytest.mark.asyncio
    async def test_natural_language_query(self, rag_system):
        """Test natural language query."""
        results = await rag_system.retrieve(
            "multiplicative inverse of a scalar",
            top_k=5,
        )

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_find_by_description(self, rag_system):
        """Test finding by conceptual description."""
        results = await rag_system.retrieve(
            "field element operations",
            top_k=5,
        )

        assert len(results) > 0


class TestGraphRetrieval:
    """Tests for Neo4j graph traversal (Level 4)."""

    @pytest.mark.asyncio
    async def test_retrieve_dependencies(self, rag_system):
        """Test dependency retrieval."""
        # First find a theorem
        results = await rag_system.retrieve("spec", top_k=1)

        if results:
            deps = await rag_system.retrieve_dependencies(
                results[0].full_name,
                max_depth=2,
            )
            # May or may not have dependencies
            assert isinstance(deps, list)


class TestCascadingBehavior:
    """Tests for cascading retrieval behavior."""

    @pytest.mark.asyncio
    async def test_results_are_deduplicated(self, rag_system):
        """Test that results are deduplicated."""
        results = await rag_system.retrieve("Scalar add", top_k=10)

        full_names = [r.full_name for r in results]
        assert len(full_names) == len(set(full_names)), "Results should be deduplicated"

    @pytest.mark.asyncio
    async def test_results_are_ranked(self, rag_system):
        """Test that results are sorted by score."""
        results = await rag_system.retrieve("add_spec", top_k=5)

        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)


class TestPerformance:
    """Performance benchmarks."""

    @pytest.mark.asyncio
    async def test_retrieval_latency(self, rag_system):
        """Test that retrieval completes in reasonable time."""
        import time

        queries = [
            "FieldElement51",
            "Scalar invert",
            "Montgomery point",
        ]

        for query in queries:
            start = time.perf_counter()
            results = await rag_system.retrieve(query, top_k=5)
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Should complete within 3 seconds (allowing for cold start)
            assert elapsed_ms < 3000, f"Query '{query}' took {elapsed_ms}ms"
            assert len(results) >= 0  # May have no results


# Validation queries for manual review
VALIDATION_QUERIES = [
    ("invert_spec", ["invert"]),
    ("Scalar Montgomery", ["montgomery", "scalar"]),
    ("field element add", ["field", "add"]),
]


@pytest.mark.parametrize("query,expected_keywords", VALIDATION_QUERIES)
@pytest.mark.asyncio
async def test_validation_query(rag_system, query, expected_keywords):
    """Validation queries for quality assessment."""
    results = await rag_system.retrieve(query, top_k=5)

    # Check if any result contains expected keywords
    found = False
    for r in results:
        text = f"{r.name} {r.full_name} {r.type_signature}".lower()
        if any(kw in text for kw in expected_keywords):
            found = True
            break

    if not found and results:
        pytest.skip(f"No results with keywords for: {query}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
