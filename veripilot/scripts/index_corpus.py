#!/usr/bin/env python3
"""
Index extracted Lean corpus into all backends.

This script takes the JSON output from extract_corpus.py and indexes it into:
- DuckDB (type index + BM25/FTS)
- Qdrant (semantic embeddings)
- Neo4j (knowledge graph)

Usage:
    python scripts/index_corpus.py --input data/extracted/combined_corpus.json
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

console = Console()


async def main():
    parser = argparse.ArgumentParser(description="Index Lean corpus into RAG backends")
    parser.add_argument(
        "--input",
        default="data/extracted/combined_corpus.json",
        help="Input JSON file from extraction",
    )
    parser.add_argument(
        "--skip-qdrant",
        action="store_true",
        help="Skip Qdrant indexing",
    )
    parser.add_argument(
        "--skip-neo4j",
        action="store_true",
        help="Skip Neo4j indexing",
    )
    parser.add_argument(
        "--skip-duckdb",
        action="store_true",
        help="Skip DuckDB indexing",
    )
    args = parser.parse_args()

    load_dotenv()

    # Resolve paths
    script_dir = Path(__file__).parent
    veripilot_dir = script_dir.parent
    input_path = (veripilot_dir / args.input).resolve()

    console.print("[bold blue]VeriPilot Corpus Indexing[/bold blue]")
    console.print(f"Input: {input_path}")

    if not input_path.exists():
        console.print(f"[red]Error: Input file not found: {input_path}[/red]")
        console.print("[yellow]Run extract_corpus.py first[/yellow]")
        sys.exit(1)

    # Load extracted data
    console.print("[blue]Loading extracted corpus...[/blue]")
    with open(input_path) as f:
        corpus = json.load(f)

    # Corpus is a flat list of declarations
    if isinstance(corpus, list):
        declarations = corpus
    elif isinstance(corpus, dict) and "entities" in corpus:
        declarations = corpus["entities"]
    else:
        # Legacy format: corpus['files'][].theorems[]
        declarations = []
        for file_data in corpus.get("files", []):
            for thm in file_data.get("theorems", []):
                full_name = thm["full_name"]
                namespace = ".".join(full_name.split(".")[:-1]) if "." in full_name else ""
                tactics = [t["tactic"].split()[0] for t in thm.get("tactics", [])]
                declarations.append({
                    "name": thm["name"],
                    "full_name": full_name,
                    "decl_type": "theorem",
                    "type_signature": thm["type_signature"],
                    "namespace": namespace,
                    "file_path": thm["file_path"],
                    "line_start": thm["line_start"],
                    "line_end": thm["line_end"],
                    "doc_string": thm.get("doc_string"),
                    "proof_preview": thm["proof"][:500] if thm.get("proof") else None,
                    "premises": thm.get("premises", []),
                    "tactics": tactics,
                })

    console.print(f"  Total declarations: {len(declarations)}")
    console.print()

    # Index into DuckDB
    if not args.skip_duckdb:
        console.print("[bold]Indexing to DuckDB...[/bold]")
        try:
            from rag.lean.type_index import LeanTypeIndex

            type_index = LeanTypeIndex()
            count = await type_index.index(declarations)
            console.print(f"[green]  Indexed {count} declarations to DuckDB[/green]")
            type_index.close()
        except Exception as e:
            console.print(f"[red]  DuckDB indexing failed: {e}[/red]")
            import traceback
            traceback.print_exc()

    # Index into Neo4j
    if not args.skip_neo4j:
        console.print("[bold]Indexing to Neo4j...[/bold]")
        try:
            from rag.lean.graph_rag_lean import LeanKnowledgeGraph

            graph = LeanKnowledgeGraph()
            count = await graph.index(declarations)
            console.print(f"[green]  Indexed {count} nodes to Neo4j[/green]")
            graph.close()
        except Exception as e:
            console.print(f"[red]  Neo4j indexing failed: {e}[/red]")
            console.print("[yellow]  Make sure NEO4J_URI is set in .env[/yellow]")
            import traceback
            traceback.print_exc()

    # Index into Qdrant (with embeddings)
    if not args.skip_qdrant:
        console.print("[bold]Indexing to Qdrant (with embeddings)...[/bold]")
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, PointStruct, VectorParams

            from rag.shared.embeddings import BGEEmbeddings, create_embedding_text

            # Connect to Qdrant Cloud
            url = os.getenv("QDRANT_URL")
            api_key = os.getenv("QDRANT_API_KEY")

            if not url or not api_key:
                console.print("[yellow]  QDRANT_URL or QDRANT_API_KEY not set[/yellow]")
                console.print("[yellow]  Skipping Qdrant indexing[/yellow]")
            else:
                client = QdrantClient(url=url, api_key=api_key)

                # Initialize embeddings
                embedder = BGEEmbeddings()
                embedding_dim = embedder.dimension

                # Recreate collection
                collection_name = "lean_proofs"
                console.print(f"  Creating collection: {collection_name}")
                client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )

                # Index with progress
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    console=console,
                ) as progress:
                    task = progress.add_task(
                        "Embedding and indexing...",
                        total=len(declarations),
                    )

                    batch_size = 32
                    for i in range(0, len(declarations), batch_size):
                        batch = declarations[i:i + batch_size]

                        # Create embedding texts
                        texts = [
                            create_embedding_text(
                                name=d.get("name", ""),
                                type_signature=d.get("type_signature", ""),
                                proof=d.get("proof_preview"),
                                doc_string=d.get("doc_string"),
                                tactics=d.get("tactics"),
                            )
                            for d in batch
                        ]

                        # Generate embeddings
                        embeddings = await embedder.embed_batch(texts)

                        # Create points for Qdrant
                        points = [
                            PointStruct(
                                id=i + j,
                                vector=embeddings[j],
                                payload={
                                    "declaration_name": d.get("name", ""),
                                    "full_name": d.get("full_name", ""),
                                    "type_signature": d.get("type_signature", ""),
                                    "namespace": d.get("namespace", ""),
                                    "file_path": d.get("file_path", ""),
                                    "proof_preview": d.get("proof_preview") or "",
                                    "doc_string": d.get("doc_string") or "",
                                    "decl_type": d.get("decl_type", ""),
                                    "corpus_source": d.get("corpus_source", ""),
                                },
                            )
                            for j, d in enumerate(batch)
                        ]

                        # Upsert to Qdrant
                        client.upsert(
                            collection_name=collection_name,
                            points=points,
                        )

                        progress.update(task, advance=len(batch))

                # Verify count
                collection_info = client.get_collection(collection_name)
                console.print(
                    f"[green]  Indexed {collection_info.points_count} vectors "
                    f"to Qdrant[/green]"
                )
                client.close()

        except Exception as e:
            console.print(f"[red]  Qdrant indexing failed: {e}[/red]")
            import traceback
            traceback.print_exc()

    console.print()
    console.print("[bold green]Indexing complete![/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
