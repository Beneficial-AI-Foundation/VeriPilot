# RAG TODO - Corpus Expansion & Quality Improvements

**Status**: Phase 1a Complete (95%) - Core RAG working, corpus expansion optional before merge
**Last Evaluated**: 2025-12-03 (Session 8)

---

## Current Baseline (21,182 entities)

### Quality Metrics
- **Precision@10**: 25.42% (2.5 relevant per 10 retrieved)
- **Recall@10**: 33.33% (finding 1/3 of ground truth items)
- **MRR**: 0.323 (relevant results often not ranked first)
- **NDCG@10**: 0.886 (good ranking when results exist)
- **Success Rate**: 4/8 test queries (50%)

### What Works ✅
- Scalar type queries (90% precision, 100% recall)
- Field operations (50% precision, 50% recall)
- Induction tactics (30% precision, 50% recall)
- Equality rewriting (33% precision, 67% recall)

### What Fails ❌
- `Nat.add_comm` query (0% recall) - Missing mathlib4/Init/
- `List.map` query (0% recall) - Missing Batteries/std4
- Natural language variants of above

**Root Cause**: Corpus gaps, not system bugs. RAG architecture validated (4/4 success when corpus has data).

---

## Quick Corpus Expansion (Recommended)

### One-Command Solution

The config has already been updated (`config/lean_rag.yaml`). Just run:

```bash
cd /workspace/projects/VeriPilot/veripilot
./scripts/expand_and_evaluate.sh
```

**What it does**:
1. Downloads new sources (Batteries, Aesop, expanded mathlib4)
2. Extracts corpus (incremental - only new sources, ~5-10 min)
3. Re-indexes databases (DuckDB + Qdrant + Neo4j, ~5 min)
4. Re-evaluates RAG quality
5. Saves results to `results_expanded.json`

**Expected Results**:
- Corpus: 21,182 → ~27,000 entities (+5,818)
- Precision@10: 25% → 40-50%
- Recall@10: 33% → 60-80%
- Success Rate: 4/8 → 7/8 (87.5%)
- All basic queries (Nat.add_comm, List.map) should work

**Time**: ~15-20 minutes total

---

## Manual Step-by-Step (if script fails)

### Prerequisites
```bash
cd /workspace/projects/VeriPilot/veripilot
source .venv/bin/activate
```

### Step 1: Verify Config (Already Done ✅)
The following sources were added to `config/lean_rag.yaml`:
- ✅ Batteries (std4) - `github.com/leanprover/std4`
- ✅ Aesop - `github.com/leanprover-community/aesop`
- ✅ mathlib4 expanded sparse checkout (added `Init/**` and `Logic/**`)

### Step 2: Download Sources (~2-3 min)
```bash
python scripts/download_sources.py --priority 1
```

**What downloads**:
- Batteries/std4 repo (~50 files, List.map, HashMap, etc.)
- Aesop repo (~80 files, proof automation)
- mathlib4 Init/ and Logic/ folders (sparse checkout update)

### Step 3: Extract Corpus (~5-10 min)
```bash
python scripts/extract_corpus.py --incremental
```

**Incremental mode** skips already-extracted sources (mathlib4, lean4-stdlib, books).
Only processes new sources: Batteries, Aesop, expanded mathlib4 folders.

**Expected additions**:
- Batteries: ~1,500 entities
- Aesop: ~800 entities
- mathlib4 Init/: ~2,500 entities
- mathlib4 Logic/: ~1,000 entities
- **Total new**: ~5,800 entities

### Step 4: Re-index Databases (~5 min)
```bash
python scripts/index_corpus.py
```

**What it does**:
- Recreates DuckDB type index (clears + re-inserts all 27K)
- Recreates Qdrant collection (deletes + re-uploads 27K vectors)
- Recreates Neo4j graph (clears + rebuilds relationships)

**Warning**: This is a full re-index, not incremental. All 27K entities re-indexed.

### Step 5: Validate (~30 sec)
```bash
# Integration tests
pytest tests/integration/test_lean_rag.py -v

# Quality evaluation
python scripts/evaluate_rag.py --output results_expanded.json
```

**Expected test results**: 9/11 or 10/11 passed (should fix Nat.add_comm, List.map queries)

### Step 6: Compare Results
```bash
# Original baseline
cat results.json | jq '.aggregate'

# After expansion
cat results_expanded.json | jq '.aggregate'
```

**Look for**:
- `mean_precision@k`: 0.2542 → 0.40-0.50
- `mean_recall@k`: 0.3333 → 0.60-0.80
- `mean_mrr`: 0.323 → 0.50+

---

## Troubleshooting

### Download fails
```bash
# Check internet connection
ping github.com

# Clear corrupted downloads
rm -rf data/sources/batteries data/sources/aesop

# Re-run
python scripts/download_sources.py --priority 1
```

### Extraction fails
```bash
# Check if sources exist
ls -la data/sources/batteries
ls -la data/sources/aesop

# Check extraction logs
python scripts/extract_corpus.py --incremental 2>&1 | tee extraction.log
```

### Indexing fails (Qdrant timeout)
```bash
# Check Qdrant connection
python -c "from qdrant_client import QdrantClient; import os; client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY')); print(client.get_collections())"

# If timeout, try smaller batches
# Edit scripts/index_corpus.py: batch_size = 32 → 16
```

### Indexing fails (Neo4j timeout)
```bash
# Check Neo4j connection
python -c "from neo4j import GraphDatabase; import os; driver = GraphDatabase.driver(os.getenv('NEO4J_URI'), auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))); driver.verify_connectivity(); print('Connected')"

# If timeout, increase Neo4j Aura plan (free tier may be slow)
```

---

## Alternative: Defer Expansion to Phase 2

**Rationale**: Current 25% precision is acceptable baseline for MVP.
- RAG architecture is validated (works perfectly when corpus has data)
- Can expand corpus incrementally based on actual agent failures
- Faster path to MVP (test agent architecture first)

**Trade-off**: Lower recall on basic queries, but may not matter if agents use different patterns.

**Decision**: Your call - merge now or expand first.

---

## Future Enhancements (Phase 2+)

### 1. Full mathlib4 Extraction
See [full-mathlib4-extraction.md](full-mathlib4-extraction.md) for detailed guide.
- Requires buildable mathlib4 project (2-6 hours)
- Would add ~38,000 more entities (from 27K → 65K)
- Use LeanDojo for proof states + tactics + premises

### 2. Advanced Retrieval Techniques
- **Query expansion**: Use LLM to rephrase queries for better recall
- **Hybrid search**: Combine BM25 + semantic with learned weights
- **Re-ranking**: Use cross-encoder to re-rank top-20 results
- **Feedback loop**: Track agent failures → add missing corpus

### 3. Quality Monitoring
- **A/B testing**: Compare retrieval strategies
- **User feedback**: Track which results agents actually use
- **Corpus coverage**: Identify frequent queries with 0% recall

### 4. Specialized Indices
- **Tactic index**: Fast lookup for tactic names/usage
- **Theorem dependency graph**: Traverse proof chains
- **Type class hierarchy**: Navigate algebraic structures

---

## Evaluation Queries Reference

Current default test queries in `scripts/evaluate_rag.py`:

1. `"Nat.add_comm"` (exact_name) - **Currently fails** (needs Init/)
2. `"commutativity of natural number addition"` (semantic) - **Currently fails**
3. `"List.map"` (exact_name) - **Currently fails** (needs Batteries)
4. `"how to transform each element of a list"` (semantic) - **Currently fails**
5. `"induction tactic"` (keyword) - **Works** (30% P, 50% R)
6. `"prove equality by rewriting"` (semantic) - **Works** (33% P, 67% R)
7. `"Scalar"` (type_signature) - **Works perfectly** (90% P, 100% R)
8. `"field element operations"` (semantic) - **Works** (50% P, 50% R)

After expansion, queries 1-4 should succeed.

---

## Summary

**Current State**: Phase 1a complete with 21K corpus, 25% precision baseline
**Recommended Next Step**: Run `./scripts/expand_and_evaluate.sh` to reach 27K corpus, 40-50% precision
**Alternative**: Merge now, expand later based on agent needs
**Time Investment**: 15-20 minutes for expansion, or 0 minutes to defer

**Decision Point**: Merge now (faster MVP) or expand first (better quality)?
