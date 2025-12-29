"""
ROMA (Recursive Orchestrated Multi-Agent) hierarchical decomposition for VeriPilot.

This package implements ROMA-style goal decomposition for complex proof goals:
- Complexity analysis to decide atomic vs decompose
- Goal decomposition into subtasks with dependencies
- Aggregation of sub-proofs into final proof
- Sub-agent spawning for parallel execution

Architecture:
    Atomizer → decides if goal is atomic or needs decomposition
    Planner → breaks complex goals into ordered subtasks
    Aggregator → synthesizes sub-proofs into final proof

Reference: docs/claude-helpers/resources/ROMA_et_al_veriplot.md
"""

from .complexity import GoalComplexity, ComplexityScore, GoalComplexityAnalyzer
from .atomizer import Atomizer
from .planner import RomaPlanner, SubTask, DecompositionPlan
from .aggregator import RomaAggregator, SubProof

__all__ = [
    # Complexity
    "GoalComplexity",
    "ComplexityScore",
    "GoalComplexityAnalyzer",
    # Atomizer
    "Atomizer",
    # Planner
    "RomaPlanner",
    "SubTask",
    "DecompositionPlan",
    # Aggregator
    "RomaAggregator",
    "SubProof",
]
