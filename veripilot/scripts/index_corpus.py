#!/usr/bin/env python3
"""
Index extracted Lean corpus into all backends.

This script takes the JSON output from extract_corpus.py and indexes it into:
- DuckDB (type index)
- Weaviate (embeddings + BM25)
- Neo4j (knowledge graph)

Usage:
    python scripts/index_corpus.py --input data/extracted/lean_corpus.json
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


async def main():
    parser = argparse.ArgumentParser(description="Index Lean corpus into RAG backends")
    parser.add_argument(
        "--input",
        default="data/extracted/lean_corpus.json",
        help="Input JSON file from extraction",
    )
    parser.add_argument(
        "--skip-weaviate",
        action="store_true",
        help="Skip Weaviate indexing",
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

    console.print(f"  Corpus: {corpus['corpus_name']}")
    console.print(f"  Theorems: {corpus['total_theorems']}")
    console.print(f"  Files: {len(corpus['files'])}")

    # Prepare data for indexing
    declarations = []
    for file_data in corpus["files"]:
        for thm in file_data["theorems"]:
            # Extract namespace from full_name
            full_name = thm["full_name"]
            namespace = ".".join(full_name.split(".")[:-1]) if "." in full_name else ""

            # Get tactic names
            tactics = [t["tactic"].split()[0] for t in thm.get("tactics", [])]

            declarations.append({
                "name": thm["name"],
                "full_name": full_name,
                "decl_type": "theorem",  # LeanDojo extracts theorems
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

    console.print(f"  Prepared {len(declarations)} declarations for indexing")
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

    # Index into Weaviate (with embeddings)
    if not args.skip_weaviate:
        console.print("[bold]Indexing to Weaviate (with embeddings)...[/bold]")
        try:
            import os
            import weaviate
            from weaviate.classes.config import Property, DataType

            from rag.shared.embeddings import BGEEmbeddings, create_embedding_text

            # Connect to Weaviate Cloud
            url = os.getenv("WEAVIATE_URL")
            api_key = os.getenv("WEAVIATE_API_KEY")

            if not url or not api_key:
                console.print("[yellow]  WEAVIATE_URL or WEAVIATE_API_KEY not set[/yellow]")
                console.print("[yellow]  Skipping Weaviate indexing[/yellow]")
            else:
                client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=url,
                    auth_credentials=weaviate.auth.AuthApiKey(api_key),
                )

                # Create collection if needed
                collection_name = "LeanProofs"
                if not client.collections.exists(collection_name):
                    client.collections.create(
                        name=collection_name,
                        properties=[
                            Property(name="declaration_name", data_type=DataType.TEXT),
                            Property(name="full_name", data_type=DataType.TEXT),
                            Property(name="type_signature", data_type=DataType.TEXT),
                            Property(name="namespace", data_type=DataType.TEXT),
                            Property(name="file_path", data_type=DataType.TEXT),
                            Property(name="proof_preview", data_type=DataType.TEXT),
                            Property(name="doc_string", data_type=DataType.TEXT),
                        ],
                    )
                    console.print(f"  Created collection: {collection_name}")

                collection = client.collections.get(collection_name)

                # Initialize embeddings
                embedder = BGEEmbeddings()

                # Index with progress
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Embedding and indexing...", total=len(declarations))

                    batch_size = 32
                    for i in range(0, len(declarations), batch_size):
                        batch = declarations[i:i + batch_size]

                        # Create embedding texts
                        texts = [
                            create_embedding_text(
                                name=d["name"],
                                type_signature=d["type_signature"],
                                proof=d.get("proof_preview"),
                                doc_string=d.get("doc_string"),
                                tactics=d.get("tactics"),
                            )
                            for d in batch
                        ]

                        # Generate embeddings
                        embeddings = await embedder.embed_batch(texts)

                        # Insert into Weaviate
                        with collection.batch.dynamic() as weaviate_batch:
                            for j, decl in enumerate(batch):
                                weaviate_batch.add_object(
                                    properties={
                                        "declaration_name": decl["name"],
                                        "full_name": decl["full_name"],
                                        "type_signature": decl["type_signature"],
                                        "namespace": decl["namespace"],
                                        "file_path": decl["file_path"],
                                        "proof_preview": decl.get("proof_preview") or "",
                                        "doc_string": decl.get("doc_string") or "",
                                    },
                                    vector=embeddings[j],
                                )

                        progress.update(task, advance=len(batch))

                console.print(f"[green]  Indexed {len(declarations)} to Weaviate[/green]")
                client.close()

        except Exception as e:
            console.print(f"[red]  Weaviate indexing failed: {e}[/red]")
            import traceback
            traceback.print_exc()

    console.print()
    console.print("[bold green]Indexing complete![/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
