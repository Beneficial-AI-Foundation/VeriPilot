# VeriPilot

Agentic Rust code verification autopilot using Lean 4 and Verus.

**Current Focus:** Lean 4 verifier agent for completing formal proofs in cryptographic code.

---

## Quick Start

```bash
# Navigate to veripilot
cd veripilot

# Activate virtual environment
source .venv/bin/activate

# Install dependencies (first time only)
pip install -e .

# Run VeriPilot
veripilot
```

Follow the interactive menu to:
1. Select a Lean file with `sorry` placeholders
2. Choose verification mode (ReAct, OpenManus, or ROMA)
3. Select model and temperature
4. Watch the agent attempt to complete the proofs

---

## Verification Modes

| Mode | Description |
|------|-------------|
| **Just Retry** | Simple retry loop with LLM proof generation |
| **ReAct** | Reasoning + Acting loop with thought/action/observation |
| **OpenManus (OM)** | ReAct with typed error recovery strategies |
| **ROMA** | Hierarchical goal decomposition for complex proofs |

---

## Project Structure

```
VeriPilot/
├── veripilot/                 # Main Python package
│   ├── src/                   # Source code
│   │   ├── agent/             # LLM agents (ReAct, ROMA)
│   │   ├── cli/               # Interactive CLI
│   │   ├── parser/            # Lean file parsing
│   │   ├── rag/               # RAG retrieval
│   │   └── verifier/          # LSP/Lake verification
│   ├── prompts/               # Agent prompts
│   └── docs/                  # Internal documentation
│
├── lean-projects/             # Lean 4 verification targets
│   └── dalek-verify-lean/     # curve25519-dalek formalization
│
└── verus-projects/            # Verus verification targets (future)
    └── dalek-lite/            # curve25519 Verus proofs
```

---

## Requirements

- Python 3.10+
- Lean 4 (with lake)
- OpenRouter API key (for LLM access)

---

## Configuration

Set your API key in `.env`:
```bash
OPENROUTER_API_KEY=your_key_here
```

---

## Current Status

- ✅ Lean 4 parser with sorry detection
- ✅ RAG with 21k+ declarations indexed
- ✅ LSP verification via MCP lean-lsp
- ✅ ReAct/OpenManus/ROMA agents implemented
- ⏳ Agent output quality improvements in progress

---

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

---

## Related Repositories

- [curve25519-dalek-lean-verify](https://github.com/Beneficial-Ai-Foundation/curve25519-dalek-lean-verify) - Lean 4 formalization
- [dalek-lite](https://github.com/Beneficial-AI-Foundation/dalek-lite) - Verus proofs (future)
