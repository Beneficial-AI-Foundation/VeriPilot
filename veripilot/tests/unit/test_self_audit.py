"""
Tests for self-auditing functionality.

Tests Poetiq pattern implementation: divergence detection,
oscillation detection, and autonomous termination decisions.
"""

import pytest

from verifier.self_audit import (
    AuditConfig,
    AuditState,
    SelfAuditingController,
    estimate_goal_complexity,
    estimate_tokens,
)


class TestAuditConfig:
    """Tests for AuditConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AuditConfig()

        assert config.max_iterations == 4
        assert config.max_tokens == 50000
        assert config.complexity_growth_threshold == 1.5
        assert config.oscillation_window == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = AuditConfig(
            max_iterations=6,
            max_tokens=100000,
            complexity_growth_threshold=2.0,
            oscillation_window=5,
        )

        assert config.max_iterations == 6
        assert config.max_tokens == 100000


class TestSelfAuditingController:
    """Tests for SelfAuditingController."""

    def test_should_continue_first_attempt(self):
        """Test that first attempt is always allowed."""
        controller = SelfAuditingController()

        should_continue, reason = controller.should_continue()

        assert should_continue is True
        assert reason == ""

    def test_should_stop_at_max_iterations(self):
        """Test stopping at max iterations."""
        config = AuditConfig(max_iterations=3)
        controller = SelfAuditingController(config)

        # Record 3 attempts
        for i in range(3):
            controller.record_attempt(
                error=f"error {i}",
                goal_complexity=10,
            )

        should_continue, reason = controller.should_continue()

        assert should_continue is False
        assert reason == "max_iterations_reached"

    def test_should_stop_at_token_budget(self):
        """Test stopping when token budget exhausted."""
        config = AuditConfig(max_tokens=1000)
        controller = SelfAuditingController(config)

        # Use up tokens
        controller.record_attempt(
            error="error",
            goal_complexity=10,
            tokens=500,
        )
        controller.record_attempt(
            error="error",
            goal_complexity=10,
            tokens=600,  # Total: 1100 > 1000
        )

        should_continue, reason = controller.should_continue()

        assert should_continue is False
        assert reason == "token_budget_exhausted"

    def test_detect_divergence(self):
        """Test divergence detection when complexity grows."""
        config = AuditConfig(
            complexity_growth_threshold=1.5,
            min_iterations_before_divergence_check=2,
        )
        controller = SelfAuditingController(config)

        # First attempt: complexity 10
        controller.record_attempt(error="e1", goal_complexity=10)
        # Second attempt: complexity 20 (2x growth > 1.5 threshold)
        controller.record_attempt(error="e2", goal_complexity=20)

        should_continue, reason = controller.should_continue()

        assert should_continue is False
        assert reason == "divergence_detected"

    def test_no_divergence_when_stable(self):
        """Test no divergence when complexity is stable."""
        config = AuditConfig(complexity_growth_threshold=1.5)
        controller = SelfAuditingController(config)

        # Stable complexity
        controller.record_attempt(error="e1", goal_complexity=10)
        controller.record_attempt(error="e2", goal_complexity=12)  # 1.2x < 1.5

        should_continue, reason = controller.should_continue()

        assert should_continue is True

    def test_detect_oscillation(self):
        """Test oscillation detection when same error repeats."""
        config = AuditConfig(oscillation_window=3)
        controller = SelfAuditingController(config)

        # Same error 3 times
        for _ in range(3):
            controller.record_attempt(
                error="type mismatch: expected Nat, got Int",
                goal_complexity=10,
            )

        should_continue, reason = controller.should_continue()

        assert should_continue is False
        assert reason == "oscillation_detected"

    def test_no_oscillation_with_different_errors(self):
        """Test no oscillation with different errors."""
        config = AuditConfig(oscillation_window=3)
        controller = SelfAuditingController(config)

        # Different errors
        controller.record_attempt(error="error 1", goal_complexity=10)
        controller.record_attempt(error="error 2", goal_complexity=10)
        controller.record_attempt(error="error 3", goal_complexity=10)

        should_continue, reason = controller.should_continue()

        assert should_continue is True

    def test_error_normalization_for_oscillation(self):
        """Test that error normalization works for oscillation detection."""
        config = AuditConfig(oscillation_window=3)
        controller = SelfAuditingController(config)

        # Same error with different line numbers (should normalize to same)
        controller.record_attempt(
            error="/path/to/file.lean:10:5: type mismatch",
            goal_complexity=10,
        )
        controller.record_attempt(
            error="/other/path/file.lean:20:10: type mismatch",
            goal_complexity=10,
        )
        controller.record_attempt(
            error="/third/file.lean:30:15: type mismatch",
            goal_complexity=10,
        )

        should_continue, reason = controller.should_continue()

        # Should detect oscillation because normalized errors are the same
        assert should_continue is False
        assert reason == "oscillation_detected"

    def test_record_successful_tactic(self):
        """Test recording a successful tactic."""
        controller = SelfAuditingController()

        controller.record_attempt(
            error=None,
            goal_complexity=5,
            tactic="simp",
            success=True,
        )

        assert "simp" in controller.state.successful_tactics
        assert len(controller.state.failed_tactics) == 0

    def test_record_failed_tactic(self):
        """Test recording a failed tactic."""
        controller = SelfAuditingController()

        controller.record_attempt(
            error="tactic failed",
            goal_complexity=10,
            tactic="grind",
            success=False,
        )

        assert "grind" in controller.state.failed_tactics
        assert len(controller.state.successful_tactics) == 0

    def test_get_summary(self):
        """Test getting audit summary."""
        controller = SelfAuditingController()

        controller.record_attempt(error="e1", goal_complexity=10, tokens=100)
        controller.record_attempt(error=None, goal_complexity=8, tokens=200)

        summary = controller.get_summary()

        assert summary["attempts"] == 2
        assert summary["tokens_used"] == 300
        assert summary["error_count"] == 1

    def test_reset(self):
        """Test resetting the controller."""
        controller = SelfAuditingController()

        controller.record_attempt(error="e1", goal_complexity=10)
        controller.record_attempt(error="e2", goal_complexity=20)

        controller.reset()

        assert controller.state.attempt == 0
        assert len(controller.state.error_history) == 0


class TestEstimateGoalComplexity:
    """Tests for estimate_goal_complexity function."""

    def test_empty_string(self):
        """Test empty string returns 0."""
        assert estimate_goal_complexity("") == 0

    def test_simple_goal(self):
        """Test simple goal has low complexity."""
        complexity = estimate_goal_complexity("Nat")
        assert complexity > 0
        assert complexity < 10

    def test_complex_goal(self):
        """Test complex goal has higher complexity."""
        simple = estimate_goal_complexity("Nat")
        complex_goal = estimate_goal_complexity(
            "∀ (x : Nat) (y : Int), f x y → g (h x) (k y)"
        )

        assert complex_goal > simple

    def test_nesting_increases_complexity(self):
        """Test that nesting increases complexity."""
        flat = estimate_goal_complexity("Nat Int Bool")
        nested = estimate_goal_complexity("(Nat, (Int, Bool))")

        assert nested > flat

    def test_arrows_increase_complexity(self):
        """Test that function arrows increase complexity."""
        no_arrows = estimate_goal_complexity("Nat Int")
        with_arrows = estimate_goal_complexity("Nat → Int → Bool")

        assert with_arrows > no_arrows


class TestEstimateTokens:
    """Tests for estimate_tokens function."""

    def test_empty_string(self):
        """Test empty string has 0 tokens."""
        assert estimate_tokens("") == 0

    def test_short_string(self):
        """Test short string token estimate."""
        # "hello" is 5 chars, ~1 token
        tokens = estimate_tokens("hello")
        assert tokens >= 1

    def test_longer_string(self):
        """Test longer string has more tokens."""
        short = estimate_tokens("hello")
        long_text = estimate_tokens("hello world this is a much longer string")

        assert long_text > short
