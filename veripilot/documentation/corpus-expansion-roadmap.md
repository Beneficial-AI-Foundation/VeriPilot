# Corpus Expansion Roadmap

This document outlines high-value corpus sources to expand the Lean RAG knowledge base beyond the current Priority 1 sources.

## Current Corpus (Phase 1a Complete)

**21,182 entities indexed:**
- mathlib4: 11,588 entities (selective extraction via sparse checkout)
- lean4-stdlib: 9,432 entities
- Books: 3,250 entities (TPIL4, fp-lean, MIL, metaprogramming-book)

## Recommended Expansions

### 1. Aesop Tactic Library ⭐ HIGH PRIORITY

**Repository**: https://github.com/leanprover-community/aesop
**Why Important**: Aesop is the most powerful automation tactic in Lean 4, essential for proof search
**Estimated Size**: ~50 files, ~5,000 lines
**Extraction Method**: Lightweight extractor (fast)

**Value**:
- Aesop rule definitions and tactics
- Proof automation patterns
- Integration with mathlib4 lemmas
- Essential for agent tactic generation

**Add to `config/lean_rag.yaml`**:
```yaml
- name: "aesop"
  type: "git"
  url: "https://github.com/leanprover-community/aesop.git"
  target_dir: "aesop"
  priority: 1
  extraction_mode: "lightweight"
```

### 2. Lean 4 Batteries (std4) ⭐ HIGH PRIORITY

**Repository**: https://github.com/leanprover/std4
**Why Important**: Standard library extensions (data structures, algorithms, tactics)
**Estimated Size**: ~200 files
**Extraction Method**: Lightweight extractor

**Value**:
- HashMap, RBTree, BinTree data structures
- Additional list/array utilities
- Tactic extensions
- Core library patterns

**Add to `config/lean_rag.yaml`**:
```yaml
- name: "batteries"
  type: "git"
  url: "https://github.com/leanprover/std4.git"
  target_dir: "batteries"
  priority: 1
  extraction_mode: "lightweight"
```

### 3. Full mathlib4 Extraction (Future)

**Current Status**: Using selective sparse checkout (5 folders: Tactic, Algebra, LinearAlgebra, AlgebraicGeometry, Data)
**Full Extraction**: See [full-mathlib4-extraction.md](full-mathlib4-extraction.md) for detailed guide

**Why Defer**:
- Requires 2-6 hours and buildable project
- Current 11,588 entities already provide strong coverage
- Can be done incrementally when needed

### 4. Sphere Eversion Project (Optional)

**Repository**: https://github.com/leanprover-community/sphere-eversion
**Why Interesting**: Modern Lean 4 practices, large formalized project
**Estimated Size**: ~10,000 lines
**Priority**: Lower (specialized mathematics)

### 5. Carleson Project (Optional)

**Repository**: https://github.com/fpvandoorn/carleson
**Why Interesting**: Recent formalization, harmonic analysis
**Estimated Size**: Large
**Priority**: Lower (very specialized)

## Implementation Steps

### Adding Aesop + Batteries (Recommended Next)

1. **Update `config/lean_rag.yaml`**: Add both sources to corpus list

2. **Download sources**:
   ```bash
   cd /workspace/projects/VeriPilot/veripilot
   python scripts/download_sources.py --priority 1
   ```

3. **Extract corpus** (incremental mode - skips already-extracted):
   ```bash
   python scripts/extract_corpus.py --incremental
   ```

4. **Re-index databases**:
   ```bash
   python scripts/index_corpus.py
   ```

5. **Validate**:
   ```bash
   pytest tests/integration/test_lean_rag.py -v
   python scripts/evaluate_rag.py
   ```

**Expected Result**: ~26,000-28,000 total entities

## Corpus Quality vs Quantity

### Current Challenges (from evaluation)

**Evaluation Results (Session 7)**:
- Mean Precision@10: 25.42%
- Mean Recall@10: ~50% (after fixing percentage bug)
- Issues: Nat.add_comm, List.map not found (0% recall)

**Root Cause**: These are likely in:
- `Nat.add_comm` → Full mathlib4 Nat section (not in sparse checkout)
- `List.map` → Batteries/std4 (not currently in corpus)

### Priority 1: Add Missing Core Items

Before expanding to niche projects, ensure core Lean 4 items are covered:

1. ✅ mathlib4 Tactic folder (done)
2. ✅ lean4-stdlib (done)
3. ⏳ **Batteries/std4** (contains List.map, core utilities)
4. ⏳ **Aesop** (proof automation)
5. ⏳ **Expand mathlib4 sparse checkout** to include:
   - `Init/` (contains Nat.add_comm and other Init items)
   - `Logic/` (basic logic lemmas)
   - `Data/List/` (list operations)

### Sparse Checkout Expansion (Quick Win)

Update `config/lean_rag.yaml` mathlib4 sparse checkout:

```yaml
sparse_checkout:
  - "Mathlib/Tactic/*"
  - "Mathlib/Algebra/*"
  - "Mathlib/LinearAlgebra/*"
  - "Mathlib/AlgebraicGeometry/*"
  - "Mathlib/Data/*"
  - "Mathlib/Init/*"        # ADD: Contains Nat.add_comm
  - "Mathlib/Logic/*"       # ADD: Basic logic
  - "Mathlib/Data/List/*"   # ADD: List operations (already covered by Data/*)
```

Then re-run:
```bash
python scripts/download_sources.py --priority 1
python scripts/extract_corpus.py --incremental
python scripts/index_corpus.py
```

## Evaluation-Driven Corpus Expansion

**Recommended Workflow**:
1. Run `python scripts/evaluate_rag.py` with test queries
2. Identify missing entities (0% recall queries)
3. Search for those entities in Lean 4 repos (GitHub search)
4. Add the repo/folder to corpus sources
5. Re-extract and re-index
6. Re-evaluate

**Example**:
- Query: "Nat.add_comm" → 0% recall
- GitHub search: Found in `mathlib4/Mathlib/Init/Data/Nat/Basic.lean`
- Solution: Add `Mathlib/Init/*` to sparse checkout
- Result: Query should now succeed

## Neo4j Node Count Discrepancy

**Observation**: 20,791 Neo4j nodes vs 21,182 entities
**Explanation**: ~391 entities lack premises for graph relationships (definitions, axioms)
**Impact**: Minimal - graph traversal still works for entities with dependencies

## Summary

**Immediate Recommendations** (to improve RAG quality before Phase 2):
1. ⭐ Add Batteries/std4 (contains List.map, core utilities)
2. ⭐ Add Aesop (proof automation)
3. ⭐ Expand mathlib4 sparse checkout to include Init/ and Logic/
4. Re-evaluate with `evaluate_rag.py` to validate improvements

**Target Metrics After Expansion**:
- Precision@10: 40-50% (currently 25.42%)
- Recall@10: 60-80% (currently ~50%)
- MRR: 0.5+ (currently 0.323)
- All "exact_name" queries should succeed

**Defer to Later**:
- Full mathlib4 LeanDojo extraction (time-intensive)
- Specialized projects (Sphere Eversion, Carleson)
- ProseSection entity extraction (architectural change)
