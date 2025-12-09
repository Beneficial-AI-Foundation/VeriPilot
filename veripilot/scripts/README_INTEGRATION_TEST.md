# Integration Test for VeriPilot MVP

Tests the full pipeline (Parser → Agent → Verifier) against real Lean files.

## What it does

1. **Creates a VP_ copy** of the target file (never modifies the original)
2. **Finds sorries** using the parser
3. **Generates proofs** using the agent (with optional RAG)
4. **Verifies with lake build** (actual Lean compilation)
5. **Cleans up** VP_ copies and backups

## Usage

### Test a single sorry

```bash
cd veripilot
source .venv/bin/activate

python scripts/test_integration.py \
  --file ../lean-projects/dalek-verify-lean/ai-benchmark/tests/input.lean \
  --line 56 \
  --model gemini
```

### Test all sorries in a file

```bash
python scripts/test_integration.py \
  --file ../lean-projects/dalek-verify-lean/ai-benchmark/tests/input.lean \
  --all \
  --model gemini
```

### Options

- `--file PATH` - Path to .lean file (required)
- `--line N` - Line number of sorry to test
- `--all` - Test all sorries in file
- `--model {gemini|claude|claude-opus|aristotle}` - LLM model (default: gemini)
- `--no-rag` - Disable RAG (test agent only)
- `--project PATH` - Lean project directory (default: ../lean-projects/dalek-verify-lean)

## Expected Behavior

### File Safety
- Original file: **NEVER modified**
- Working copy: `VP_input.lean` (created in same directory)
- Backup: `VP_input.lean.bak` (created during verification)
- All temp files cleaned up after test

### Timeline
- Parser: <1s
- Agent (proof generation): 5-10s
- Verifier (lake build): 2-5 minutes per attempt (up to 4 attempts)
- **Total**: 2-20 minutes per sorry

### Output Example

```
============================================================
Testing sorry at input.lean:56
============================================================

✓ Created test copy: VP_input.lean

[1/3] Parsing VP_input.lean...
✓ Found sorry: sub_spec at line 56
  Signature: theorem sub_spec (a b : Array U64 5#usize)...

[2/3] Generating proof with gemini...
✓ Generated proof (42 chars):
  unfold sub
  progress*
  simp [Scalar52_as_Nat]
  Used 3 RAG results

[3/3] Verifying with lake build...
  (This may take 2-5 minutes...)

============================================================
✅ SUCCESS! Proof verified in 2 attempt(s)
  Time: 183.4s
  Final proof:
    unfold sub
    progress*
    simp [Scalar52_as_Nat]
    omega
============================================================

✓ Cleaned up: VP_input.lean
✓ Cleaned up: VP_input.lean.bak
```

## Benchmark File

**Default**: `lean-projects/dalek-verify-lean/ai-benchmark/tests/input.lean`

Contains 6 sorries:
- Line 21, 22, 23, 24: Inside `sub_loop_spec` progress block
- Line 33: `sub_loop_spec` arithmetic property
- Line 56: `sub_spec` main theorem

## Troubleshooting

### "RAG initialization failed"
- The test will continue without RAG (agent only)
- Check RAG config in `config/lean_rag.yaml`
- Verify Qdrant/Neo4j/DuckDB connections

### "lake build timed out"
- Default timeout: 5 minutes
- First build takes longer (compiling dependencies)
- Consider testing simpler theorems first

### "Lake build failed"
- Check that Lean/Lake are installed: `lake --version`
- Verify project builds: `cd lean-projects/dalek-verify-lean && lake build`
- Check proof is syntactically valid

## Next Steps

After successful integration test:
1. Proceed with Phase 4 (CLI implementation)
2. Run full evaluation on all benchmark files (Phase 5)
3. Compare with `dalek-lean-ai` baseline

## Notes

- **This test costs API credits** (LLM calls for proof generation)
- Start with `--line 56` (simplest sorry) before `--all`
- The integration test is **slower** than unit tests but validates real functionality
