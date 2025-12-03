#!/usr/bin/env python3
"""
RAG Quality Evaluation Script

Measures retrieval quality metrics:
- Precision@K: % of retrieved results that are relevant
- Recall@K: % of relevant results that were retrieved
- MRR (Mean Reciprocal Rank): 1/rank of first relevant result
- NDCG@K: Normalized Discounted Cumulative Gain

Usage:
    python scripts/evaluate_rag.py --queries queries.json
    python scripts/evaluate_rag.py --interactive
"""
import argparse
import asyncio
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

console = Console()


# Default test queries with ground truth
DEFAULT_QUERIES = [
    {
        "query": "Nat.add_comm",
        "relevant": ["Nat.add_comm"],
        "description": "Exact name match - commutativity of addition",
        "type": "exact_name"
    },
    {
        "query": "commutativity of natural number addition",
        "relevant": ["Nat.add_comm", "Nat.comm"],
        "description": "Natural language query for commutative property",
        "type": "semantic"
    },
    {
        "query": "List.map",
        "relevant": ["List.map"],
        "description": "Exact name match - list map function",
        "type": "exact_name"
    },
    {
        "query": "how to transform each element of a list",
        "relevant": ["List.map", "List.traverse"],
        "description": "Semantic query for list transformation",
        "type": "semantic"
    },
    {
        "query": "induction tactic",
        "relevant": ["induction", "Nat.rec"],
        "description": "Tactic keyword search",
        "type": "keyword"
    },
    {
        "query": "prove equality by rewriting",
        "relevant": ["rw", "rewrite", "Eq.trans"],
        "description": "Semantic tactic query",
        "type": "semantic"
    },
    {
        "query": "Scalar",
        "relevant": ["Scalar"],
        "description": "Type signature pattern",
        "type": "type_signature"
    },
    {
        "query": "field element operations",
        "relevant": ["Field", "DivisionRing"],
        "description": "Algebraic structure semantic query",
        "type": "semantic"
    },
]


class RAGEvaluator:
    """Evaluates RAG retrieval quality."""

    def __init__(self, rag_system):
        self.rag = rag_system

    async def evaluate_query(
        self,
        query: str,
        relevant: List[str],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Evaluate a single query."""
        # Retrieve results
        results = await self.rag.retrieve(query, top_k=top_k)

        # Extract retrieved names (handle partial matches)
        retrieved_names = [r.full_name for r in results]

        # Calculate metrics
        metrics = self._calculate_metrics(
            retrieved_names,
            relevant,
            top_k
        )

        return {
            "query": query,
            "retrieved_count": len(retrieved_names),
            "relevant_count": len(relevant),
            "retrieved": retrieved_names[:5],  # Top 5 for display
            "metrics": metrics,
        }

    def _calculate_metrics(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> Dict[str, float]:
        """Calculate all retrieval metrics."""
        # Handle empty cases
        if not retrieved:
            return {
                "precision@k": 0.0,
                "recall@k": 0.0,
                "mrr": 0.0,
                "ndcg@k": 0.0,
            }

        # Precision@K: % of retrieved that are relevant
        retrieved_k = retrieved[:k]
        num_relevant_retrieved = self._count_relevant(retrieved_k, relevant)
        precision = num_relevant_retrieved / len(retrieved_k) if retrieved_k else 0.0

        # Recall@K: % of ground truth items that were found
        num_ground_truth_found = self._count_ground_truth_found(retrieved_k, relevant)
        recall = num_ground_truth_found / len(relevant) if relevant else 0.0

        # MRR: 1/rank of first relevant result
        mrr = self._calculate_mrr(retrieved, relevant)

        # NDCG@K: Normalized ranking quality
        ndcg = self._calculate_ndcg(retrieved_k, relevant)

        return {
            "precision@k": precision,
            "recall@k": recall,
            "mrr": mrr,
            "ndcg@k": ndcg,
        }

    def _count_relevant(
        self,
        retrieved: List[str],
        relevant: List[str]
    ) -> int:
        """Count how many retrieved results are relevant (for Precision)."""
        count = 0
        for ret in retrieved:
            for rel in relevant:
                # Exact match or partial match
                if rel.lower() in ret.lower() or ret.lower() in rel.lower():
                    count += 1
                    break
        return count

    def _count_ground_truth_found(
        self,
        retrieved: List[str],
        relevant: List[str]
    ) -> int:
        """Count how many ground truth items were found (for Recall)."""
        found = 0
        for rel in relevant:
            for ret in retrieved:
                # Check if this ground truth item appears in retrieved
                if rel.lower() in ret.lower() or ret.lower() in rel.lower():
                    found += 1
                    break  # Found this ground truth, move to next
        return found

    def _calculate_mrr(
        self,
        retrieved: List[str],
        relevant: List[str]
    ) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, ret in enumerate(retrieved):
            for rel in relevant:
                if rel.lower() in ret.lower() or ret.lower() in rel.lower():
                    return 1.0 / (i + 1)
        return 0.0

    def _calculate_ndcg(
        self,
        retrieved: List[str],
        relevant: List[str]
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        # DCG: sum of rel / log2(rank+1)
        dcg = 0.0
        for i, ret in enumerate(retrieved):
            relevance = 1.0 if self._is_relevant(ret, relevant) else 0.0
            dcg += relevance / math.log2(i + 2)  # i+2 because rank starts at 1

        # IDCG: perfect ranking
        idcg = 0.0
        for i in range(min(len(relevant), len(retrieved))):
            idcg += 1.0 / math.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def _is_relevant(self, retrieved: str, relevant: List[str]) -> bool:
        """Check if retrieved result is relevant (fuzzy match)."""
        for rel in relevant:
            if rel.lower() in retrieved.lower() or retrieved.lower() in rel.lower():
                return True
        return False

    async def evaluate_all(
        self,
        queries: List[Dict[str, Any]],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Evaluate all queries and aggregate metrics."""
        results = []

        console.print(f"[blue]Evaluating {len(queries)} queries...[/blue]\n")

        for i, q in enumerate(queries, 1):
            console.print(f"[{i}/{len(queries)}] {q['query']}")

            result = await self.evaluate_query(
                q["query"],
                q["relevant"],
                top_k=top_k
            )
            result["description"] = q.get("description", "")
            result["type"] = q.get("type", "unknown")
            results.append(result)

            # Display individual result
            metrics = result["metrics"]
            console.print(
                f"  P@{top_k}: {metrics['precision@k']:.2%} | "
                f"R@{top_k}: {metrics['recall@k']:.2%} | "
                f"MRR: {metrics['mrr']:.3f} | "
                f"NDCG: {metrics['ndcg@k']:.3f}"
            )
            console.print()

        # Aggregate metrics
        aggregate = self._aggregate_metrics(results)

        return {
            "individual_results": results,
            "aggregate": aggregate,
            "query_count": len(queries),
            "top_k": top_k,
        }

    def _aggregate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics across all queries."""
        if not results:
            return {}

        metrics = ["precision@k", "recall@k", "mrr", "ndcg@k"]
        aggregate = {}

        for metric in metrics:
            values = [r["metrics"][metric] for r in results]
            aggregate[f"mean_{metric}"] = sum(values) / len(values)
            aggregate[f"min_{metric}"] = min(values)
            aggregate[f"max_{metric}"] = max(values)

        return aggregate


def print_results(eval_results: Dict[str, Any]):
    """Pretty print evaluation results."""
    console.print("\n[bold green]RAG Evaluation Results[/bold green]\n")

    # Aggregate table
    console.print("[bold]Aggregate Metrics[/bold]")
    table = Table(show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    agg = eval_results["aggregate"]
    metrics_display = [
        ("Precision@K", "precision@k"),
        ("Recall@K", "recall@k"),
        ("MRR", "mrr"),
        ("NDCG@K", "ndcg@k"),
    ]

    for display_name, metric_key in metrics_display:
        is_percentage = "precision" in metric_key or "recall" in metric_key
        mean_val = (
            f"{agg[f'mean_{metric_key}']:.2%}"
            if is_percentage
            else f"{agg[f'mean_{metric_key}']:.3f}"
        )
        min_val = (
            f"{agg[f'min_{metric_key}']:.2%}"
            if is_percentage
            else f"{agg[f'min_{metric_key}']:.3f}"
        )
        max_val = (
            f"{agg[f'max_{metric_key}']:.2%}"
            if is_percentage
            else f"{agg[f'max_{metric_key}']:.3f}"
        )
        table.add_row(display_name, mean_val, min_val, max_val)

    console.print(table)
    console.print()

    # Individual results table
    console.print("[bold]Individual Query Results[/bold]")
    table = Table(show_header=True)
    table.add_column("Query", style="cyan", max_width=40)
    table.add_column("Type", style="yellow")
    table.add_column("P@K", justify="right")
    table.add_column("R@K", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("NDCG", justify="right")

    for result in eval_results["individual_results"]:
        m = result["metrics"]
        table.add_row(
            result["query"][:40],
            result["type"],
            f"{m['precision@k']:.2%}",
            f"{m['recall@k']:.2%}",
            f"{m['mrr']:.3f}",
            f"{m['ndcg@k']:.3f}",
        )

    console.print(table)
    console.print()

    # Summary
    console.print("[bold]Summary[/bold]")
    console.print(f"  Total queries: {eval_results['query_count']}")
    console.print(f"  Top-K: {eval_results['top_k']}")
    console.print(f"  Mean Precision@{eval_results['top_k']}: {agg['mean_precision@k']:.2%}")
    console.print(f"  Mean Recall@{eval_results['top_k']}: {agg['mean_recall@k']:.2%}")
    console.print(f"  Mean MRR: {agg['mean_mrr']:.3f}")
    console.print(f"  Mean NDCG@{eval_results['top_k']}: {agg['mean_ndcg@k']:.3f}")


async def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument(
        "--queries",
        help="JSON file with test queries",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve (default: 10)",
    )
    parser.add_argument(
        "--output",
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode - enter queries manually",
    )
    args = parser.parse_args()

    # Load queries
    if args.queries:
        with open(args.queries) as f:
            queries = json.load(f)
    elif args.interactive:
        console.print("[yellow]Interactive mode not yet implemented[/yellow]")
        console.print("[yellow]Using default queries instead[/yellow]\n")
        queries = DEFAULT_QUERIES
    else:
        console.print("[blue]Using default test queries[/blue]\n")
        queries = DEFAULT_QUERIES

    # Initialize RAG system
    console.print("[blue]Initializing RAG system...[/blue]")
    from rag.lean.llamaindex_lean import LeanRAG

    rag = LeanRAG()
    await rag.initialize()

    try:
        # Run evaluation
        evaluator = RAGEvaluator(rag)
        results = await evaluator.evaluate_all(queries, top_k=args.top_k)

        # Display results
        print_results(results)

        # Save to file if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            console.print(f"\n[green]Results saved to {args.output}[/green]")

    finally:
        await rag.close()


if __name__ == "__main__":
    asyncio.run(main())
