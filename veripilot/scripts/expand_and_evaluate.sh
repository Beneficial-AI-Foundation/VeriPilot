#!/bin/bash
# Corpus Expansion and Re-evaluation Script
# Adds Batteries, Aesop, and mathlib4 Init/ to corpus, then re-evaluates RAG quality

set -e  # Exit on error

echo "========================================="
echo "VeriPilot Corpus Expansion"
echo "========================================="
echo ""

# Change to veripilot directory
cd "$(dirname "$0")/.."

# Activate virtual environment
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "ERROR: Virtual environment not found at .venv/"
    echo "Please run: python -m venv .venv && pip install -e ."
    exit 1
fi

# Step 1: Download new sources
echo "Step 1/4: Downloading new corpus sources (Batteries, Aesop, expanded mathlib4)..."
python scripts/download_sources.py --priority 1

# Step 2: Extract corpus (incremental mode - only processes new sources)
echo ""
echo "Step 2/4: Extracting corpus (incremental mode - skips already-extracted)..."
python scripts/extract_corpus.py --incremental

# Step 3: Re-index databases
echo ""
echo "Step 3/4: Re-indexing databases (DuckDB + Qdrant + Neo4j)..."
python scripts/index_corpus.py

# Step 4: Run evaluation
echo ""
echo "Step 4/4: Running RAG quality evaluation..."
python scripts/evaluate_rag.py --output results_expanded.json

echo ""
echo "========================================="
echo "Corpus Expansion Complete!"
echo "========================================="
echo ""
echo "Check results_expanded.json for detailed metrics."
echo "Expected improvements:"
echo "  - Precision@10: 25% → 40-50%"
echo "  - Recall@10: 33% → 60-80%"
echo "  - Nat.add_comm and List.map queries should now succeed"
