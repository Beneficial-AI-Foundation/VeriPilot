"""
LangGraph StateGraph definition for ReAct proof verification.

Implements the core ReAct loop as a LangGraph StateGraph:

    START → reasoning → execution → observation → router
                ↑                                    │
                └────────── (continue) ─────────────┘
                                                     │
                              (success) → success_node → END
                              (failed)  → failed_node  → END

The graph uses conditional edges from the router node to determine
whether to continue iterating or terminate.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Callable, Any

from langgraph.graph import StateGraph, START, END

from .state import ProofState, ProofStatus
from .nodes import (
    reasoning_node,
    execution_node,
    observation_node,
    router_node,
    om_router_node,
    recovery_node,
    reset_recovery_stage,
    increment_attempt_node,
    success_node,
    failed_node,
    # Phase 4.0: Goal parsing
    goal_parser_node,
    # ROMA nodes
    complexity_analysis_node,
    atomizer_node,
    roma_router_node,
    planner_node,
    subtask_executor_node,
    subtask_router_node,
    aggregator_node,
)

if TYPE_CHECKING:
    from langgraph.graph.graph import CompiledGraph
    from verifier.verifier_service import VerifierService

logger = logging.getLogger(__name__)


def create_react_graph(
    verifier_service: Optional["VerifierService"] = None,
) -> "CompiledGraph":
    """
    Create and compile the ReAct proof verification graph.

    The graph implements a Thought → Action → Observation loop with
    automatic routing based on verification results.

    Args:
        verifier_service: VerifierService for LSP verification.
                         If None, verification will be simulated (for testing).

    Returns:
        Compiled LangGraph ready for invocation.

    Example:
        graph = create_react_graph(verifier_service)
        result = await graph.ainvoke(initial_state)
    """
    # Create graph with ProofState schema
    graph = StateGraph(ProofState)

    # =========================================================================
    # Add Nodes
    # =========================================================================

    # Goal parser node (Phase 4.0) - parses goal state for structured analysis
    graph.add_node("goal_parser", goal_parser_node)

    # Reasoning node - async, generates thought and plans tactic
    graph.add_node("reasoning", reasoning_node)

    # Execution node - records the action being taken
    # We wrap it to inject the verifier_service
    async def execution_with_service(state: ProofState) -> dict:
        return await execution_node(state, verifier_service)

    graph.add_node("execution", execution_with_service)

    # Observation node - async, runs verification and observes result
    async def observation_with_service(state: ProofState) -> dict:
        return await observation_node(state, verifier_service)

    graph.add_node("observation", observation_with_service)

    # Increment attempt counter for continue case
    graph.add_node("increment_attempt", increment_attempt_node)

    # Terminal nodes
    graph.add_node("success_terminal", success_node)
    graph.add_node("failed_terminal", failed_node)

    # =========================================================================
    # Add Edges
    # =========================================================================

    # Linear flow: START → goal_parser → reasoning → execution → observation
    graph.add_edge(START, "goal_parser")
    graph.add_edge("goal_parser", "reasoning")
    graph.add_edge("reasoning", "execution")
    graph.add_edge("execution", "observation")

    # Conditional edges from observation based on router
    graph.add_conditional_edges(
        "observation",
        router_node,
        {
            "continue": "increment_attempt",
            "success": "success_terminal",
            "failed": "failed_terminal",
        },
    )

    # Continue loop: increment → reasoning
    graph.add_edge("increment_attempt", "reasoning")

    # Terminal nodes go to END
    graph.add_edge("success_terminal", END)
    graph.add_edge("failed_terminal", END)

    # =========================================================================
    # Compile
    # =========================================================================

    compiled = graph.compile()
    logger.debug("ReAct graph compiled successfully")

    return compiled


def create_simple_graph() -> "CompiledGraph":
    """
    Create a minimal ReAct graph for testing without verification.

    This graph skips actual LSP verification and is useful for:
    - Unit testing the graph structure
    - Testing reasoning without a Lean project

    Returns:
        Compiled LangGraph for testing
    """
    return create_react_graph(verifier_service=None)


def create_om_react_graph(
    verifier_service: Optional["VerifierService"] = None,
) -> "CompiledGraph":
    """
    Create OpenManus-enhanced ReAct graph with error recovery.

    This graph extends the basic ReAct loop with:
    - Error recovery node for intelligent tactic modification
    - OpenManus-style error classification and strategy selection
    - Multi-stage recovery (primary → fallback → backtrack)

    Graph topology:
        START → reasoning → execution → observation
                     ↑                         │
                     │                  (om_router_node)
                     │                    /    │    \
              increment_attempt      recover  success  failed
                     ↑                   │       │       │
                     │                   v       v       v
                     +── (continue) ────-+      END     END

    Args:
        verifier_service: VerifierService for LSP verification.
                         If None, verification will be simulated.

    Returns:
        Compiled LangGraph with OpenManus error recovery.

    Example:
        graph = create_om_react_graph(verifier_service)
        result = await graph.ainvoke(initial_state)
    """
    # Create graph with ProofState schema
    graph = StateGraph(ProofState)

    # =========================================================================
    # Add Nodes
    # =========================================================================

    # Goal parser node (Phase 4.0) - parses goal state for structured analysis
    graph.add_node("goal_parser", goal_parser_node)

    # Reasoning node - async, generates thought and plans tactic
    graph.add_node("reasoning", reasoning_node)

    # Execution node - records the action being taken
    async def execution_with_service(state: ProofState) -> dict:
        return await execution_node(state, verifier_service)

    graph.add_node("execution", execution_with_service)

    # Observation node - async, runs verification and observes result
    async def observation_with_service(state: ProofState) -> dict:
        return await observation_node(state, verifier_service)

    graph.add_node("observation", observation_with_service)

    # Recovery node - OpenManus error analysis and tactic modification
    graph.add_node("recovery", recovery_node)

    # Reset recovery stage after successful verification
    graph.add_node("reset_recovery", reset_recovery_stage)

    # Increment attempt counter for continue case
    graph.add_node("increment_attempt", increment_attempt_node)

    # Terminal nodes
    graph.add_node("success_terminal", success_node)
    graph.add_node("failed_terminal", failed_node)

    # =========================================================================
    # Add Edges
    # =========================================================================

    # Linear flow: START → goal_parser → reasoning → execution → observation
    graph.add_edge(START, "goal_parser")
    graph.add_edge("goal_parser", "reasoning")
    graph.add_edge("reasoning", "execution")
    graph.add_edge("execution", "observation")

    # Conditional edges from observation based on OpenManus router
    graph.add_conditional_edges(
        "observation",
        om_router_node,
        {
            "continue": "increment_attempt",
            "recover": "recovery",
            "success": "success_terminal",
            "failed": "failed_terminal",
        },
    )

    # Recovery leads back to execution with modified tactic
    # (skip reasoning since recovery already modified the tactic)
    graph.add_edge("recovery", "execution")

    # Continue loop: increment → reset_recovery → reasoning
    graph.add_edge("increment_attempt", "reset_recovery")
    graph.add_edge("reset_recovery", "reasoning")

    # Terminal nodes go to END
    graph.add_edge("success_terminal", END)
    graph.add_edge("failed_terminal", END)

    # =========================================================================
    # Compile
    # =========================================================================

    compiled = graph.compile()
    logger.debug("OpenManus ReAct graph compiled successfully")

    return compiled


async def run_om_react_verification(
    initial_state: ProofState,
    verifier_service: Optional["VerifierService"] = None,
    on_step: Optional[Callable[[ProofState], None]] = None,
) -> ProofState:
    """
    Run the OpenManus-enhanced ReAct verification loop.

    Convenience function that creates the OM graph and runs it.

    Args:
        initial_state: Initial ProofState from create_initial_state()
        verifier_service: VerifierService for LSP verification
        on_step: Optional callback called after each step (for UI updates)

    Returns:
        Final ProofState with results including recovery trace
    """
    graph = create_om_react_graph(verifier_service)

    if on_step:
        final_state = initial_state
        async for event in graph.astream(initial_state):
            for node_name, node_output in event.items():
                if isinstance(node_output, dict):
                    for key, value in node_output.items():
                        if key in final_state:
                            if isinstance(final_state[key], list) and isinstance(value, list):
                                final_state[key] = final_state[key] + value
                            else:
                                final_state[key] = value
            on_step(final_state)
        return final_state
    else:
        return await graph.ainvoke(initial_state)


async def run_react_verification(
    initial_state: ProofState,
    verifier_service: Optional["VerifierService"] = None,
    on_step: Optional[Callable[[ProofState], None]] = None,
) -> ProofState:
    """
    Run the ReAct verification loop.

    Convenience function that creates the graph and runs it.

    Args:
        initial_state: Initial ProofState from create_initial_state()
        verifier_service: VerifierService for LSP verification
        on_step: Optional callback called after each step (for UI updates)

    Returns:
        Final ProofState with results

    Example:
        from agent.react import create_initial_state, run_react_verification

        state = create_initial_state(sorry, proof, file_content)
        result = await run_react_verification(state, verifier)

        if result["status"] == "success":
            print(f"Proof verified: {result['current_proof']}")
    """
    graph = create_react_graph(verifier_service)

    # Run with streaming if callback provided
    if on_step:
        final_state = initial_state
        async for event in graph.astream(initial_state):
            # Extract state from event
            for node_name, node_output in event.items():
                if isinstance(node_output, dict):
                    # Merge update into state (simplified - real merge uses reducers)
                    for key, value in node_output.items():
                        if key in final_state:
                            if isinstance(final_state[key], list) and isinstance(value, list):
                                final_state[key] = final_state[key] + value
                            else:
                                final_state[key] = value
            on_step(final_state)
        return final_state
    else:
        # Simple invoke
        return await graph.ainvoke(initial_state)


# ==============================================================================
# ROMA Graph (Hierarchical Decomposition)
# ==============================================================================

def create_roma_graph(
    verifier_service: Optional["VerifierService"] = None,
) -> "CompiledGraph":
    """
    Create and compile the ROMA hierarchical decomposition graph.

    The graph implements intelligent goal decomposition:
    - Complexity analysis to assess goal difficulty
    - Atomizer decides: solve directly or decompose
    - Planner creates subtask plan for complex goals
    - Subtask executor runs each subtask
    - Aggregator synthesizes sub-proofs into final proof

    Graph topology:
        START -> complexity_analysis -> atomizer -> roma_router
                                                        |
                                               (direct/decompose)
                                                /            \\
                                         direct_loop       planner
                                              |                |
                                            END          subtask_executor
                                                               |
                                                        subtask_router
                                                         /    |    \\
                                                   next   aggregate  failed
                                                     |        |        |
                                                     +---> aggregator  |
                                                              |        |
                                                            END      END

    Args:
        verifier_service: VerifierService for LSP verification.
                         If None, verification will be simulated.

    Returns:
        Compiled LangGraph ready for invocation.
    """
    graph = StateGraph(ProofState)

    # =========================================================================
    # Add Nodes
    # =========================================================================

    # Goal parser node (Phase 4.0) - parses goal state for structured analysis
    graph.add_node("goal_parser", goal_parser_node)

    # Complexity analysis node
    graph.add_node("complexity_analysis", complexity_analysis_node)

    # Atomizer decision node
    graph.add_node("atomizer", atomizer_node)

    # Planner node for decomposition
    graph.add_node("planner", planner_node)

    # Subtask executor with verifier service
    async def subtask_with_service(state: ProofState) -> dict:
        return await subtask_executor_node(state, verifier_service)

    graph.add_node("subtask_executor", subtask_with_service)

    # Aggregator node
    graph.add_node("aggregator", aggregator_node)

    # Direct path nodes (for atomic goals) - reuse ReAct nodes
    graph.add_node("direct_reasoning", reasoning_node)

    async def direct_execution(state: ProofState) -> dict:
        return await execution_node(state, verifier_service)

    graph.add_node("direct_execution", direct_execution)

    async def direct_observation(state: ProofState) -> dict:
        return await observation_node(state, verifier_service)

    graph.add_node("direct_observation", direct_observation)

    graph.add_node("direct_increment", increment_attempt_node)

    # Terminal nodes
    graph.add_node("success_terminal", success_node)
    graph.add_node("failed_terminal", failed_node)

    # =========================================================================
    # Add Edges
    # =========================================================================

    # Entry: START -> goal_parser -> complexity_analysis -> atomizer
    graph.add_edge(START, "goal_parser")
    graph.add_edge("goal_parser", "complexity_analysis")
    graph.add_edge("complexity_analysis", "atomizer")

    # Atomizer decision routing
    graph.add_conditional_edges(
        "atomizer",
        roma_router_node,
        {
            "direct": "direct_reasoning",
            "decompose": "planner",
            "success": "success_terminal",
            "failed": "failed_terminal",
        },
    )

    # Direct path (for atomic goals): reasoning -> execution -> observation
    graph.add_edge("direct_reasoning", "direct_execution")
    graph.add_edge("direct_execution", "direct_observation")

    # Direct path routing after observation
    graph.add_conditional_edges(
        "direct_observation",
        router_node,
        {
            "continue": "direct_increment",
            "success": "success_terminal",
            "failed": "failed_terminal",
        },
    )

    # Loop back from increment
    graph.add_edge("direct_increment", "direct_reasoning")

    # Decomposition path: planner -> subtask_executor -> subtask_router
    graph.add_edge("planner", "subtask_executor")

    graph.add_conditional_edges(
        "subtask_executor",
        subtask_router_node,
        {
            "next_subtask": "subtask_executor",  # Loop for more subtasks
            "aggregate": "aggregator",
            "failed": "failed_terminal",
        },
    )

    # Aggregator routing
    def aggregator_router(state: ProofState) -> str:
        if state["status"] == ProofStatus.SUCCESS.value:
            return "success"
        return "failed"

    graph.add_conditional_edges(
        "aggregator",
        aggregator_router,
        {
            "success": "success_terminal",
            "failed": "failed_terminal",
        },
    )

    # Terminal nodes go to END
    graph.add_edge("success_terminal", END)
    graph.add_edge("failed_terminal", END)

    # =========================================================================
    # Compile
    # =========================================================================

    compiled = graph.compile()
    logger.debug("ROMA graph compiled successfully")

    return compiled


async def run_roma_verification(
    initial_state: ProofState,
    verifier_service: Optional["VerifierService"] = None,
    on_step: Optional[Callable[[ProofState], None]] = None,
) -> ProofState:
    """
    Run the ROMA hierarchical decomposition verification loop.

    Convenience function that creates the ROMA graph and runs it.

    Args:
        initial_state: Initial ProofState from create_initial_state()
        verifier_service: VerifierService for LSP verification
        on_step: Optional callback called after each step (for UI updates)

    Returns:
        Final ProofState with results including decomposition trace
    """
    graph = create_roma_graph(verifier_service)

    if on_step:
        final_state = initial_state
        async for event in graph.astream(initial_state):
            for node_name, node_output in event.items():
                if isinstance(node_output, dict):
                    for key, value in node_output.items():
                        if key in final_state:
                            if isinstance(final_state[key], list) and isinstance(value, list):
                                final_state[key] = final_state[key] + value
                            else:
                                final_state[key] = value
            on_step(final_state)
        return final_state
    else:
        return await graph.ainvoke(initial_state)


# ==============================================================================
# Graph Visualization (for debugging)
# ==============================================================================

def get_graph_mermaid(
    verifier_service: Optional["VerifierService"] = None,
) -> str:
    """
    Get Mermaid diagram representation of the graph.

    Useful for documentation and debugging.

    Returns:
        Mermaid diagram string
    """
    graph = create_react_graph(verifier_service)

    try:
        return graph.get_graph().draw_mermaid()
    except Exception:
        # Fallback to manual diagram
        return """
graph TD
    START([START]) --> reasoning
    reasoning[Reasoning Node] --> execution
    execution[Execution Node] --> observation
    observation[Observation Node] --> router{Router}
    router -->|continue| increment_attempt
    increment_attempt[Increment Attempt] --> reasoning
    router -->|success| success_terminal
    router -->|failed| failed_terminal
    success_terminal[Success] --> END([END])
    failed_terminal[Failed] --> END
"""
