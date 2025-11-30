#!/usr/bin/env python3
"""
Test connections to cloud services (Qdrant, Neo4j) and local DuckDB.

Usage:
    pytest tests/integration/test_connections.py -v

Or run directly:
    python tests/integration/test_connections.py

This script verifies all database connections are working before
running the extraction and indexing pipeline.
"""
import os
import sys
import tempfile
from pathlib import Path

# pytest is optional - only needed when running via pytest
try:
    import pytest
except ImportError:
    pytest = None

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


class TestQdrantConnection:
    """Test Qdrant Cloud connection."""

    def test_connection_and_crud(self):
        """Test Qdrant connection with create, insert, query, delete."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct

        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")

        assert url, "QDRANT_URL not set in .env"
        assert api_key, "QDRANT_API_KEY not set in .env"

        client = QdrantClient(url=url, api_key=api_key)

        # Create test collection (delete if exists, then create)
        test_collection = "_veripilot_test_collection"
        if client.collection_exists(collection_name=test_collection):
            client.delete_collection(collection_name=test_collection)

        client.create_collection(
            collection_name=test_collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

        try:
            # Insert test vector
            client.upsert(
                collection_name=test_collection,
                points=[
                    PointStruct(
                        id=1,
                        vector=[0.1] * 384,
                        payload={"test": "data"},
                    )
                ],
            )

            # Query using query_points() method (modern API in 1.16.x)
            results = client.query_points(
                collection_name=test_collection,
                query=[0.1] * 384,
                limit=1,
                with_payload=True,
            )

            assert len(results.points) == 1, "Query should return 1 result"
            assert results.points[0].payload["test"] == "data"

        finally:
            # Cleanup
            client.delete_collection(collection_name=test_collection)


class TestNeo4jConnection:
    """Test Neo4j Aura connection."""

    def test_connection_and_query(self):
        """Test Neo4j connection with simple query."""
        from neo4j import GraphDatabase

        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")

        assert uri, "NEO4J_URI not set in .env"
        assert password, "NEO4J_PASSWORD not set in .env"

        driver = GraphDatabase.driver(uri, auth=(user, password))

        try:
            with driver.session() as session:
                result = session.run("RETURN 1 AS test")
                record = result.single()
                assert record is not None, "Query should return a record"
                assert record["test"] == 1, "Query should return 1"
        finally:
            driver.close()


class TestDuckDBFTS:
    """Test DuckDB with FTS extension."""

    def test_fts_extension(self):
        """Test DuckDB FTS extension loads and works."""
        import duckdb

        # Use in-memory database instead of tempfile
        conn = duckdb.connect(':memory:')

        # Install and load FTS
        conn.execute("INSTALL fts")
        conn.execute("LOAD fts")

        # Create test table
        conn.execute("""
            CREATE TABLE test_fts (
                id INTEGER PRIMARY KEY,
                content VARCHAR
            )
        """)
        conn.execute("INSERT INTO test_fts VALUES (1, 'theorem about natural numbers')")
        conn.execute("INSERT INTO test_fts VALUES (2, 'lemma for list induction')")

        # Create FTS index
        conn.execute("""
            PRAGMA create_fts_index(
                'test_fts',
                'id',
                'content',
                stemmer = 'none',
                stopwords = 'none',
                lower = 1
            )
        """)

        # Test FTS query
        result = conn.execute("""
            SELECT id, content, fts_main_test_fts.match_bm25(id, 'natural') AS score
            FROM test_fts
            WHERE score IS NOT NULL
            ORDER BY score DESC
        """).fetchall()

        assert len(result) >= 1, "FTS should find at least 1 result"
        assert result[0][0] == 1, "First result should be id=1 (natural numbers)"

        conn.close()


class TestEmbeddings:
    """Test BGE embedding model."""

    def test_embedding_model_loads(self):
        """Test BGE embedding model can be loaded and used."""
        from sentence_transformers import SentenceTransformer

        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        model = SentenceTransformer(model_name)

        # Test embedding
        test_text = "theorem about natural numbers"
        embedding = model.encode(test_text)

        assert len(embedding) == 384, f"Expected 384 dimensions, got {len(embedding)}"


def run_all_tests():
    """Run all tests and print summary (for direct execution)."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print()
    console.print("[bold]VeriPilot Connection Tests[/bold]")
    console.print("=" * 50)
    console.print()

    tests = [
        ("Qdrant Cloud", TestQdrantConnection().test_connection_and_crud),
        ("Neo4j Aura", TestNeo4jConnection().test_connection_and_query),
        ("DuckDB + FTS", TestDuckDBFTS().test_fts_extension),
        ("BGE Embeddings", TestEmbeddings().test_embedding_model_loads),
    ]

    results = []
    for name, test_fn in tests:
        console.print(f"Testing {name}...", end=" ")
        try:
            test_fn()
            results.append((name, True, "OK"))
            console.print("[green]✓ PASS[/green]")
        except Exception as e:
            results.append((name, False, str(e)))
            console.print("[red]✗ FAIL[/red]")

    # Summary table
    console.print()
    table = Table(title="Connection Test Results")
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details")

    all_passed = True
    for name, success, message in results:
        status = "[green]PASS[/green]" if success else "[red]FAIL[/red]"
        display_msg = message[:60] + "..." if len(message) > 60 else message
        table.add_row(name, status, display_msg)
        if not success:
            all_passed = False

    console.print(table)

    if all_passed:
        console.print()
        console.print("[bold green]All connections working! Ready to proceed.[/bold green]")
        return 0
    else:
        console.print()
        console.print("[bold red]Some connections failed. Check .env configuration.[/bold red]")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
