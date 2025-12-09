# Integration Test Ready! 🚀

Phase 3 (Verification) is complete and ready for real-world testing.

## What's Ready

✅ **All 3 MVP phases implemented** (70/70 unit tests passing):
- Phase 1: Parser (11 tests)
- Phase 2: Prover Agent (30 tests)
- Phase 3: Verifier (29 tests)

✅ **Integration test script created**: `scripts/test_integration.py`

✅ **File safety**: Creates `VP_` copies, never modifies originals

## Quick Start

### Test a single sorry (recommended first test)

```bash
cd veripilot
source .venv/bin/activate

# Test the simplest sorry (line 56)
python scripts/test_integration.py \
  --file ../lean-projects/dalek-verify-lean/ai-benchmark/tests/input.lean \
  --line 56 \
  --model gemini
```

**Expected time**: 2-5 minutes (mostly lake build)

### Test all 6 sorries

```bash
python scripts/test_integration.py \
  --file ../lean-projects/dalek-verify-lean/ai-benchmark/tests/input.lean \
  --all \
  --model gemini
```

**Expected time**: 12-30 minutes (6 sorries × 2-5 min each)

## What to Expect

1. **VP_input.lean created** - Working copy (original untouched)
2. **Proof generated** - Agent uses RAG + LLM
3. **Lake build runs** - Real Lean compilation (slow but validates everything)
4. **Cleanup** - All temp files removed

## Success Criteria

✅ Parser finds the sorry correctly
✅ Agent generates valid Lean tactics
✅ Verifier replaces sorry with proof
✅ Lake build succeeds (proof compiles)
✅ Original file unchanged

## If Something Fails

This is **expected** for the first test! The integration will reveal:

- **Agent issues**: Proof generation not quite right for this domain
- **RAG issues**: Context not sufficient
- **Prompt issues**: Need to tune prompts for Aeneas code
- **Verifier issues**: Edge cases in file modification or error parsing

Each failure gives us concrete feedback to improve the system.

## Cost Warning

- Each test uses LLM API credits (up to 4 attempts per sorry)
- Estimated: $0.10-0.50 per sorry with Gemini
- Budget: $50 total for MVP

## After Integration Test

### If it works:
1. ✅ Validate pipeline works end-to-end
2. → Proceed to Phase 4 (CLI)
3. → Phase 5 (Full evaluation)

### If it fails (expected):
1. 🔍 Analyze failure mode
2. 🛠️ Fix specific issue (agent, RAG, verifier)
3. 🔄 Retry integration test
4. → Iterate until working

## Documentation

- Full details: `scripts/README_INTEGRATION_TEST.md`
- Progress tracker: `docs/claude-helpers/MVP/mvp-progress-tracker.md`

## Commands Reference

```bash
# Single sorry test
python scripts/test_integration.py --file ../lean-projects/dalek-verify-lean/ai-benchmark/tests/input.lean --line 56

# All sorries
python scripts/test_integration.py --file ../lean-projects/dalek-verify-lean/ai-benchmark/tests/input.lean --all

# Without RAG (agent only)
python scripts/test_integration.py --file ../lean-projects/dalek-verify-lean/ai-benchmark/tests/input.lean --line 56 --no-rag

# Different model
python scripts/test_integration.py --file ../lean-projects/dalek-verify-lean/ai-benchmark/tests/input.lean --line 56 --model claude
```

---

**Ready when you are!** 🎯
