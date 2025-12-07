"""Parser module for extracting sorry locations and goal context from Lean files."""

from dataclasses import dataclass, field


@dataclass
class SorryLocation:
    """Represents a sorry placeholder in a Lean file with its context."""

    file_path: str
    line: int  # 1-indexed
    column: int  # 1-indexed
    theorem_name: str
    theorem_signature: str
    proof_prefix: str  # tactics before sorry
    namespace: str = ""
    imports: list[str] = field(default_factory=list)


@dataclass
class LeanGoal:
    """Represents a proof goal at a specific location."""

    target_type: str
    hypotheses: list[dict] = field(default_factory=list)  # [{name, type}, ...]


from .sorry_finder import find_sorries
from .goal_extractor import get_goal_at_sorry

__all__ = ["SorryLocation", "LeanGoal", "find_sorries", "get_goal_at_sorry"]
