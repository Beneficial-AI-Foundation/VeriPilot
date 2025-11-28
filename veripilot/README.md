# VeriPilot

**Dual-Language Rust Verification Copilot**

VeriPilot is an intelligent formalization copilot that unifies Lean 4 theorem proving and Verus Rust verification through multi-agent orchestration, RAG-enhanced context retrieval, and LLM council validation.

---

## Overview

VeriPilot provides automated proof completion for:
- **Lean 4**: Mathematical theorem proving with tactic generation
- **Verus**: Rust code verification with annotation synthesis

The system intelligently routes requests based on file type and intent, retrieves relevant knowledge from curated indices, and orchestrates multiple AI agents to complete proofs at the best of its capabilities.

---

## Key Features

- **Language-Aware Routing**: Automatic detection of Lean vs. Verus intent with confidence scoring
- **RAG Infrastructure**: Semantic graphs, type indices, and proof exemplars for both languages
- **Multi-Agent Orchestration**: LangGraph state machine with CrewAI agents for modular task execution
- **LLM Council**: Multi-model validation with debate mechanism for high-stakes decisions
- **Attempt Escalation**: Six-phase retry logic from basic prover to council to human escalation
- **Interactive CLI**: Persistent session with real-time feedback and progress streaming

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourorg/veripilot.git
cd veripilot

# Run with Docker (recommended)
docker-compose up

# Or install locally
pip install -e .
veripilot --help
```

### Basic Commands

```bash
# Verify a Rust file with Verus
veripilot verify src/crypto.rs --lines 42-100

# Prove a Lean theorem
veripilot prove Curve25519.lean --theorem add_assoc

# Interactive chat mode
veripilot chat
```

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                 CLI Interface                    │
│  Commands: /verify, /prove, /chat, /escalate    │
├─────────────────────────────────────────────────┤
│             Agent Orchestration                  │
│  ┌─────────────┐  ┌──────────────────────────┐  │
│  │  LangGraph  │  │       CrewAI Agents      │  │
│  │ State Graph │  │ Router, Prover, Debugger │  │
│  └─────────────┘  └──────────────────────────┘  │
├─────────────────────────────────────────────────┤
│              Knowledge Layer (RAG)               │
│  Lean Index │ Verus Index │ Bridge Patterns     │
├─────────────────────────────────────────────────┤
│            Verification Backends                 │
│     Lean MCP Server    │    Verus + Z3          │
└─────────────────────────────────────────────────┘
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [REQUIREMENTS.md](REQUIREMENTS.md) | Sacred project specification |
| [docs/architecture/overview.md](docs/architecture/overview.md) | System architecture |
| [docs/phases/](docs/phases/) | Phase-by-phase implementation docs |
| [docs/claude-helpers/README.md](docs/claude-helpers/README.md) | Navigation guide for development |

---

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | Not Started | RAG Infrastructure |
| Phase 2 | Not Started | Agent Framework |
| Phase 3 | Not Started | Advanced Orchestration |
| Phase 4 | Not Started | CLI Interface |
| Phase 5 | Not Started | Integration & Hardening |

See [docs/claude-helpers/global-progress-tracker.md](docs/claude-helpers/global-progress-tracker.md) for detailed progress.

---

## Technology Stack

- **Orchestration**: LangGraph + CrewAI
- **LLMs**: Gemini Pro (primary), Claude 3.5 (ensemble)
- **RAG**: LlamaIndex + Neo4j + Weaviate
- **Verification**: Lean MCP Server, Verus Compiler + Z3
- **Observability**: LangFuse
- **CLI**: Typer + Rich

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
ruff check src/
```

---

## Project Structure

```
veripilot/
├── REQUIREMENTS.md          # Sacred specification
├── README.md                # This file
├── docs/
│   ├── architecture/        # Static system design docs
│   ├── phases/              # Phase implementation docs
│   └── claude-helpers/      # Session management for Claude Code
├── src/
│   ├── agents/              # CrewAI agent implementations
│   ├── orchestration/       # LangGraph state machine
│   ├── rag/                 # RAG infrastructure
│   ├── feedback/            # Verification backend integrations
│   ├── council/             # LLM council implementation
│   ├── gating/              # Confidence gating
│   └── cli/                 # CLI interface
├── tests/                   # Test suites
├── config/                  # YAML configuration files
└── docker/                  # Docker deployment files
```

---

## License

See [LICENSE](../LICENSE) for details.

---

## Related Projects

- [dalek-verify-lean](../lean-projects/dalek-verify-lean) - Lean 4 curve25519 formalization
- [dalek-lite](../verus-projects/dalek-lite) - Verus Rust verification for curve25519
