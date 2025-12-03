# Full Mathlib4 Extraction Guide (Future)

**Status**: Deferred - Use lightweight extraction for now
**Created**: 2025-11-30 (Session 3)
**Purpose**: Guide for running LeanDojo on full mathlib4 when ready to scale up

---

## Overview

This guide documents how to run full LeanDojo extraction on mathlib4 to get proof states, tactic sequences, and premise relationships. This is **deferred** because:

1. LeanDojo requires complete, buildable Lean projects
2. Sparse checkouts may cause build failures
3. Full extraction takes 2-6 hours
4. Initial corpus uses lightweight regex extraction (faster, sufficient for type/signature indexing)

**When to use this**: After validating the pipeline on smaller repos (sphere-eversion, metaprogramming-book) and confirming the need for proof state data from mathlib4.

---

## Prerequisites

### System Requirements
- **Disk space**: ~10GB minimum
  - mathlib4 repo: ~1.5GB
  - Lake cache: ~3-5GB
  - LeanDojo cache: ~2-4GB
  - Build artifacts: ~1-2GB
- **RAM**: 16GB+ recommended (8GB minimum)
- **CPU**: Multi-core recommended (uses `NUM_PROCS` parallel processes)
- **Time**: 2-6 hours depending on hardware

### Software Requirements
- Lean 4 installed (`elan` toolchain manager)
- Python 3.10+
- LeanDojo: `pip install lean-dojo`
- Git with LFS support

---

## Step 1: Clone Full Mathlib4

```bash
cd /workspace/data/sources

# Clone full mathlib4 repository
git clone https://github.com/leanprover-community/mathlib4.git
cd mathlib4

# Check commit hash (for reproducibility)
git rev-parse HEAD

# Check Lean version
cat lean-toolchain
```

**Expected output**:
- Repo size: ~1.5GB
- ~15,000+ `.lean` files
- Lean version: `leanprover/lean4:v4.x.x`

---

## Step 2: Download Pre-Built Cache (CRITICAL)

**This step saves hours** - mathlib4 takes 4-6 hours to build from scratch. Use pre-compiled cache:

```bash
cd mathlib4

# Download pre-built cache (requires ~3-5GB disk space)
lake exe cache get

# Verify cache downloaded
ls -lh .lake/build/lib/
```

If `lake exe cache get` fails:
```bash
# Alternative: use curl to download cache directly
curl -L "https://github.com/leanprover-community/mathlib4/releases/latest/download/cache.tar.gz" -o cache.tar.gz
tar -xzf cache.tar.gz -C .lake/
```

---

## Step 3: Run LeanDojo Tracing

### Option A: Trace Entire Repository

```bash
cd /workspace/projects/VeriPilot/veripilot
source .venv/bin/activate

python -c "
from lean_dojo import LeanGitRepo, trace
import os

# Set cache directory
os.environ['CACHE_DIR'] = '/workspace/data/lean_dojo_cache'
os.environ['NUM_PROCS'] = '4'  # Adjust based on CPU cores

# Trace mathlib4
repo = LeanGitRepo('/workspace/data/sources/mathlib4', commit='HEAD')
traced_repo = trace(repo)

print(f'Traced {len(list(traced_repo.traced_files))} files')
print(f'Cache location: /workspace/data/lean_dojo_cache')
"
```

**Expected duration**: 2-6 hours
**Expected output**: ~15,000 traced files

### Option B: Incremental Extraction (Recommended)

Start with high-value modules to test the pipeline:

```bash
# Extract only Tactic module first (~2000 files, ~30 min)
python scripts/extract_mathlib_module.py --module Mathlib/Tactic --output data/extracted/mathlib_tactic.json

# Then add more modules incrementally:
# - Mathlib/Algebra (~3000 files)
# - Mathlib/Data (~2000 files)
# - Mathlib/LinearAlgebra (~1500 files)
```

---

## Step 4: Extract Using VeriPilot Pipeline

Once LeanDojo tracing completes, use the extraction script:

```bash
cd /workspace/projects/VeriPilot/veripilot
source .venv/bin/activate

# Run extraction with LeanDojo mode
python scripts/extract_corpus.py \
    --mode lean_dojo \
    --repo /workspace/data/sources/mathlib4 \
    --output data/extracted/mathlib4_full.json
```

---

## Expected Output

### Extraction Statistics
- **Theorems**: ~50,000+
- **Lemmas**: ~40,000+
- **Definitions**: ~20,000+
- **Tactic steps**: ~500,000+
- **Proof states**: ~300,000+
- **Premise relationships**: ~1M+ edges

### Output Files
```
data/extracted/
├── mathlib4_full.json          # Complete extraction
├── mathlib4_theorems.json      # Theorems only
├── mathlib4_tactics.json       # Tactic compendium
└── mathlib4_premises.json      # Dependency graph
```

### Data Schema
Each extracted theorem includes:
```json
{
  "name": "List.map_id",
  "full_name": "List.map_id",
  "type_signature": "{α : Type u} → List.map (fun x => x) = id",
  "proof": "by simp [List.map_id']",
  "file_path": "Mathlib/Data/List/Basic.lean",
  "line_start": 1234,
  "line_end": 1236,
  "premises": ["List.map_id'", "simp"],
  "tactics": [
    {
      "index": 0,
      "tactic": "simp [List.map_id']",
      "state_before": "α : Type u\n⊢ List.map (fun x => x) = id",
      "state_after": "no goals",
      "success": true
    }
  ],
  "doc_string": "Mapping the identity function over a list is the identity."
}
```

---

## Troubleshooting

### Issue: Out of Memory

**Symptoms**: Python killed, `MemoryError`

**Solutions**:
1. Reduce parallel processes:
   ```bash
   export NUM_PROCS=2  # Default is 4
   ```

2. Process modules incrementally instead of all at once

3. Increase swap space (if using VM):
   ```bash
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### Issue: Lake Build Timeout

**Symptoms**: `lake build` hangs or times out

**Solutions**:
1. Use pre-built cache: `lake exe cache get`
2. Check internet connection (downloads large files)
3. Verify Lean toolchain version matches mathlib4:
   ```bash
   elan show
   cat lean-toolchain
   ```

### Issue: LeanDojo Extraction Fails

**Symptoms**: `trace()` raises exceptions, files missing

**Solutions**:
1. Verify mathlib4 builds successfully:
   ```bash
   cd mathlib4
   lake build  # Should complete without errors
   ```

2. Check LeanDojo cache permissions:
   ```bash
   ls -la /workspace/data/lean_dojo_cache
   chmod -R 755 /workspace/data/lean_dojo_cache
   ```

3. Use older mathlib4 commit if latest is unstable:
   ```bash
   git checkout v4.x.0  # Known stable release
   ```

### Issue: Extraction Too Slow

**Symptoms**: Taking longer than 6 hours

**Solutions**:
1. Check if cache was downloaded (Step 2)
2. Increase `NUM_PROCS` if you have spare CPU cores
3. Use SSD instead of HDD for `/workspace/data/`
4. Extract high-priority modules only (Tactic, Data, Algebra)

---

## Incremental Extraction Strategy

Instead of extracting all mathlib4 at once, extract by priority:

### Priority 1: Tactics (~2000 files, ~30 min)
```bash
python scripts/extract_mathlib_module.py \
    --module Mathlib/Tactic \
    --output data/extracted/mathlib_tactic.json
```

**Why first**: Tactic proofs are most useful for proof automation

### Priority 2: Core Data Structures (~2000 files, ~30 min)
```bash
python scripts/extract_mathlib_module.py \
    --module Mathlib/Data \
    --output data/extracted/mathlib_data.json
```

**Why second**: Fundamental types (List, Array, Nat, etc.)

### Priority 3: Algebra (~3000 files, ~45 min)
```bash
python scripts/extract_mathlib_module.py \
    --module Mathlib/Algebra \
    --output data/extracted/mathlib_algebra.json
```

**Why third**: Common proof patterns in algebraic structures

### Priority 4: Remaining Modules (~8000 files, ~2-3 hours)
```bash
# Extract everything else
python scripts/extract_corpus.py \
    --mode lean_dojo \
    --repo /workspace/data/sources/mathlib4 \
    --exclude Mathlib/Tactic Mathlib/Data Mathlib/Algebra \
    --output data/extracted/mathlib4_remaining.json
```

---

## Caching and Resumption

LeanDojo caches traced results. If extraction is interrupted:

```bash
# Check cache
ls -lh /workspace/data/lean_dojo_cache/

# Resume extraction (automatically uses cache)
python scripts/extract_corpus.py --mode lean_dojo --repo /workspace/data/sources/mathlib4
```

LeanDojo will skip already-traced files and continue from where it stopped.

---

## Validation

After extraction completes, validate the output:

```bash
python scripts/validate_extraction.py data/extracted/mathlib4_full.json

# Expected output:
# ✓ 50,000+ theorems extracted
# ✓ 500,000+ tactic steps
# ✓ 1M+ premise relationships
# ✓ All file paths valid
# ✓ Type signatures non-empty
```

---

## Performance Benchmarks

### Hardware Configurations

| Hardware | Extraction Time | Notes |
|----------|----------------|-------|
| 4 CPU cores, 16GB RAM, SSD | 2-3 hours | Recommended minimum |
| 8 CPU cores, 32GB RAM, SSD | 1.5-2 hours | Optimal for full extraction |
| 2 CPU cores, 8GB RAM, HDD | 6-8 hours | Minimum viable (use incremental) |

### Extraction Modes Comparison

| Mode | Time | Proof States | Tactics | Premises |
|------|------|--------------|---------|----------|
| Lightweight (regex) | 10-15 min | ❌ | ❌ | ❌ |
| LeanDojo (full) | 2-6 hours | ✅ | ✅ | ✅ |

---

## When to Use Full Extraction

**Use lightweight extraction** when:
- You only need type signatures and declarations
- Building proof search index for name/type matching
- Rapid iteration during development

**Use LeanDojo full extraction** when:
- You need proof state data for tactic generation
- Building premise selection models
- Analyzing tactic usage patterns
- Creating proof step prediction datasets

---

## Related Documents

- [../progress.md](../progress.md) - Current phase status
- [../resources/Lean-RAG-Resources.md](../resources/Lean-RAG-Resources.md) - Corpus source list
- [../resources/Perplexity_LeanDojo_DataExtraction.md](../resources/Perplexity_LeanDojo_DataExtraction.md) - LeanDojo API reference

---

**Last Updated**: 2025-11-30 (Session 3)
**Status**: Documentation complete, extraction deferred until lightweight pipeline validated
