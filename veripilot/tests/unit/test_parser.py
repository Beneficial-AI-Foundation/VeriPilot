"""Unit tests for the parser module."""

import pytest
from pathlib import Path

from src.parser import SorryLocation, LeanGoal, find_sorries
from src.parser.goal_extractor import parse_goal_response, format_goal_for_prompt


def _find_benchmark_file() -> str:
    """Find the benchmark file in either BAIF or legacy path."""
    candidates = [
        "/workspace/projects/BAIF/VeriPilot/lean-projects/dalek-verify-lean/ai-benchmark/tests/input.lean",
        "/workspace/projects/VeriPilot/lean-projects/dalek-verify-lean/ai-benchmark/tests/input.lean",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    raise FileNotFoundError(f"Benchmark file not found in: {candidates}")


@pytest.fixture
def benchmark_file():
    """Path to the dalek benchmark input.lean file."""
    return _find_benchmark_file()


class TestSorryFinder:
    """Tests for the sorry_finder module."""

    def test_find_all_sorries(self, benchmark_file):
        """Test that all 6 sorries are found in the benchmark file."""
        sorries = find_sorries(benchmark_file)
        assert len(sorries) == 6

    def test_sorry_locations(self, benchmark_file):
        """Test that sorry locations are correct."""
        sorries = find_sorries(benchmark_file)

        # Expected lines: 21, 22, 23, 24, 33, 56
        expected_lines = [21, 22, 23, 24, 33, 56]
        actual_lines = [s.line for s in sorries]
        assert actual_lines == expected_lines

    def test_theorem_names(self, benchmark_file):
        """Test that theorem names are correctly extracted."""
        sorries = find_sorries(benchmark_file)

        # First 5 sorries are in sub_loop_spec, last one in sub_spec
        assert all(s.theorem_name == "sub_loop_spec" for s in sorries[:5])
        assert sorries[5].theorem_name == "sub_spec"

    def test_line_range_filtering(self, benchmark_file):
        """Test that line range filtering works correctly."""
        # Only the sub_spec sorry on line 56
        sorries = find_sorries(benchmark_file, line_range=(50, 60))
        assert len(sorries) == 1
        assert sorries[0].line == 56
        assert sorries[0].theorem_name == "sub_spec"

        # First 4 sorries (lines 21-24)
        sorries = find_sorries(benchmark_file, line_range=(20, 25))
        assert len(sorries) == 4
        assert all(20 <= s.line <= 25 for s in sorries)

    def test_proof_prefix_extraction(self, benchmark_file):
        """Test that proof prefixes are extracted."""
        sorries = find_sorries(benchmark_file)

        # All sorries should have some proof context
        for sorry in sorries:
            assert sorry.proof_prefix is not None
            # First 5 should have unfold tactics
            if sorry.line <= 33:
                assert "unfold" in sorry.proof_prefix

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        with pytest.raises(FileNotFoundError):
            find_sorries("/nonexistent/file.lean")


class TestGoalExtractor:
    """Tests for the goal_extractor module."""

    def test_parse_simple_goal(self):
        """Test parsing a simple goal with no hypotheses."""
        goal_text = "⊢ 2 + 2 = 4"
        goal = parse_goal_response(goal_text)

        assert goal is not None
        assert goal.target_type == "2 + 2 = 4"
        assert goal.hypotheses == []

    def test_parse_goal_with_hypotheses(self):
        """Test parsing a goal with hypotheses."""
        goal_text = """n : ℕ
m : ℕ
h : n < m
⊢ n + 1 ≤ m"""
        goal = parse_goal_response(goal_text)

        assert goal is not None
        assert goal.target_type == "n + 1 ≤ m"
        assert len(goal.hypotheses) == 3
        assert goal.hypotheses[0] == {"name": "n", "type": "ℕ"}
        assert goal.hypotheses[1] == {"name": "m", "type": "ℕ"}
        assert goal.hypotheses[2] == {"name": "h", "type": "n < m"}

    def test_parse_no_goals(self):
        """Test parsing 'no goals' response."""
        goal_text = "no goals"
        goal = parse_goal_response(goal_text)
        assert goal is None

    def test_parse_empty_string(self):
        """Test parsing empty response."""
        goal = parse_goal_response("")
        assert goal is None

    def test_format_goal_for_prompt(self):
        """Test formatting a goal for LLM prompt."""
        goal = LeanGoal(
            target_type="n + 1 ≤ m",
            hypotheses=[
                {"name": "n", "type": "ℕ"},
                {"name": "h", "type": "n < m"},
            ],
        )

        formatted = format_goal_for_prompt(goal)
        assert "Hypotheses:" in formatted
        assert "n : ℕ" in formatted
        assert "h : n < m" in formatted
        assert "Goal: ⊢ n + 1 ≤ m" in formatted
