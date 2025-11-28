# VeriPilot: Dual-Language Rust Verification Copilot

**Purpose**: Foundational project specification - NEVER MODIFY

**Created**: 2025-11-28

---

## Mission Statement

VeriPilot is an intelligent formalization copilot that unifies two verification paradigms:
1. **Lean 4**: Mathematical theorem proving (formal proofs, mathematical correctness)
2. **Verus**: Rust code verification (systems correctness, invariant proof)

The system provides a language-aware agent orchestration framework that detects intent, routes intelligently to appropriate verification backends, and orchestrates multi-agent workflows for automated proof completion.

---

## Core Requirements

### 1. Dual-Language Verification
- Detect language intent (prove theorem vs. verify Rust function) with confidence scoring
- Route to appropriate verification backend (Lean MCP or Verus + Z3)
- Support file extension-based routing (`.lean` → Lean, `.rs` → Verus)
- No mixed-language files; single decision point per request

### 2. RAG Infrastructure
- Lean Knowledge Index: Semantic graphs, type index, proof exemplars, tactic compendium
- Verus Knowledge Index: Annotation patterns, SMT strategies, Rust verification patterns
- Bridge Knowledge: Rust ↔ Lean translation patterns
- Cascading retrieval: Type lookup → BM25 → Embeddings → Graph traversal

### 3. Multi-Agent Orchestration
- LangGraph state machine for workflow management
- CrewAI agents for role-based task execution
- Eight primary agents: Language Router, Decomposer, Lean Prover, Verus Prover, Translator, Formalizer, Debugger, CLI Handler
- Stateful attempt tracking with escalation logic

### 4. LLM Council & Confidence Gating
- HallBayes confidence scoring for proposal quality assessment
- Three-reviewer council (Proposer, Critic, Validator) with voting mechanism
- Debate phase for disagreement resolution
- Gating thresholds: >0.75 proceed, 0.5-0.75 council review, <0.5 escalate

### 5. CLI Interface
- Interactive chatbot with persistent session
- Commands: `/verify`, `/prove`, `/escalate-to-lean`, `/model`, `/smt-strategy`
- Real-time feedback, progress streaming, syntax highlighting
- Docker packaging for portable deployment

---

## Architecture Principles

### Modularity
- Language backends are pluggable; new verifiers add without core changes
- Abstract interfaces for RAG providers, agents, and gating strategies
- Factory pattern for component instantiation from YAML config

### Production-Readiness
- Circuit breakers and fallback mechanisms
- Observability instrumentation (LangFuse)
- Reproducibility through context snapshots

### Safety by Default
- Confidence gating filters low-quality proposals
- Council review validates high-stakes decisions
- Six-attempt escalation with human escalation fallback

### Developer-Friendly
- Persistent CLI session
- Incremental verification
- Non-blocking suggestions

---

## Scope & Constraints

### In Scope
- Lean 4 proof generation and completion
- Verus annotation generation for Rust code
- RAG-enhanced context retrieval
- Multi-agent orchestration with attempt loops
- CLI chatbot interface
- Docker deployment

### Out of Scope
- Other theorem provers (Coq, Isabelle, etc.)
- Other verification systems (Dafny, F*, etc.)
- GUI/web interface (CLI only for MVP)
- Direct IDE integration (future phase)

### Technical Constraints
- **Budget**: $300 total API costs across all phases
- **Timeline**: 10-12 weeks
- **Primary LLM**: Gemini 2.0 / Gemini Pro (cost-effective)
- **Backup LLM**: Claude 3.5 (ensemble diversity)
- **Orchestration**: LangGraph + CrewAI (no Autogen)
- **RAG Abstraction**: LlamaIndex (single framework)
- **Observability**: LangFuse (single platform)

---

## Success Criteria

### Demo-Ready
- [ ] VeriPilot completes at least 5 real curve25519 proofs (Lean + Verus)
- [ ] CLI feels polished (comparable to claude-code)
- [ ] Docker deployment works on any machine

### Quality Metrics
- [ ] 90%+ simple goal success rate
- [ ] 60%+ medium complexity goal success rate
- [ ] <10 seconds for simple goals
- [ ] <60 seconds for complex goals
- [ ] RAG retrieval relevance: 80%+ (Lean), 75%+ (Verus)

### Portfolio Quality
- [ ] Clean, minimal code
- [ ] Comprehensive documentation
- [ ] Professional CLI experience

---

## Phase Overview

| Phase | Name | Duration | Budget | Dependencies |
|-------|------|----------|--------|--------------|
| 1 | RAG Infrastructure | 2-3 weeks | $50 | None |
| 2 | Agent Framework | 3-4 weeks | $100 | Phase 1 |
| 3 | Advanced Orchestration | 2 weeks | $80 | Phase 2 |
| 4 | CLI Interface | 1.5-2 weeks | $40 | Phase 2 basics |
| 5 | Integration & Hardening | 1.5-2 weeks | $30 | All phases |

---

## Technology Stack

| Layer | Component | Technology |
|-------|-----------|------------|
| Orchestration | State Machine | LangGraph |
| Agent Coordination | Multi-agent Framework | CrewAI |
| LLM Primary | Main Generation | Gemini 2.0 / Gemini Pro |
| LLM Backup | Ensemble / Critique | Claude 3.5 |
| Knowledge Retrieval | RAG Abstraction | LlamaIndex |
| Semantic Graph | Knowledge Store | Neo4j |
| Type/Tactic Index | Fast Lookup | DuckDB |
| Vector Store | Embeddings | Weaviate / Milvus |
| Lean Integration | Native Verification | Lean MCP Server |
| Verus Integration | Rust Verification | Verus Compiler + Z3 |
| Observability | Telemetry & Tracing | LangFuse |
| CLI Framework | Interactive Shell | Typer + Rich |

---

## Document Governance

**This document is SACRED and must NEVER be modified after initial creation.**

- Always consult when planning features
- Serves as final arbiter for scope questions
- All scope changes require new project, not modification of this spec
