"""
ReAct Agent package for VeriPilot proof verification.

This package implements a LangGraph-based ReAct (Reasoning + Acting) agent
for Lean 4 proof verification. It transforms VeriPilot's simple retry loop
into a reasoning agent that learns from failures.

Components:
- state.py: LangGraph TypedDict state definitions
- nodes.py: Graph nodes (reasoning, execution, observation, router)
- graph.py: StateGraph definition and compilation
- agent.py: Main entry point for agent invocation

Agent Modes (selectable via CLI):
- JUST_RETRY: Simple retry loop (baseline, uses existing retry_handler)
- REACT: Thought → Action → Observation loop
- OM_REACT: ReAct + OpenManus error recovery patterns
- ROMA: Hierarchical decomposition for complex goals

Usage:
    from agent.react import ReActAgent, AgentMode

    agent = ReActAgent(mode=AgentMode.REACT)
    result = await agent.verify(sorry, initial_proof, rag=rag)
"""

from .state import (
    # Enums
    AgentMode,
    ProofStatus,
    # TypedDicts
    ProofState,
    ThoughtRecord,
    ActionRecord,
    ObservationRecord,
    # Factory functions
    create_initial_state,
    sorry_to_dict,
    dict_to_sorry,
    # Helper functions
    add_thought,
    add_action,
    add_observation,
    get_trace_summary,
    is_terminal,
    should_backtrack,
)

# Lazy imports for modules that require langgraph
# This allows importing state types even if langgraph isn't installed


def __getattr__(name: str):
    """Lazy loading of submodule classes that require langgraph."""
    if name == "ReActAgent":
        from .agent import ReActAgent
        return ReActAgent
    elif name == "ReActResult":
        from .agent import ReActResult
        return ReActResult
    elif name == "verify_with_react":
        from .agent import verify_with_react
        return verify_with_react
    elif name == "get_available_modes":
        from .agent import get_available_modes
        return get_available_modes
    elif name == "create_react_graph":
        from .graph import create_react_graph
        return create_react_graph
    elif name == "run_react_verification":
        from .graph import run_react_verification
        return run_react_verification
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Enums
    "AgentMode",
    "ProofStatus",
    # TypedDicts
    "ProofState",
    "ThoughtRecord",
    "ActionRecord",
    "ObservationRecord",
    # Factory functions
    "create_initial_state",
    "sorry_to_dict",
    "dict_to_sorry",
    # Helper functions
    "add_thought",
    "add_action",
    "add_observation",
    "get_trace_summary",
    "is_terminal",
    "should_backtrack",
    # Agent (lazy loaded)
    "ReActAgent",
    "ReActResult",
    "verify_with_react",
    "get_available_modes",
    # Graph (lazy loaded)
    "create_react_graph",
    "run_react_verification",
]
