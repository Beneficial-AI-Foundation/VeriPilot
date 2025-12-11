"""
Tests for error message normalization.

Tests mapping Lean errors to LLM-friendly forms.
"""

import pytest

from verifier.error_normalizer import (
    ErrorMessageNormalizer,
    NormalizedError,
    normalize_error,
    format_error_for_prompt,
)


class TestErrorMessageNormalizer:
    """Tests for ErrorMessageNormalizer class."""

    def setup_method(self):
        """Create normalizer for each test."""
        self.normalizer = ErrorMessageNormalizer()

    def test_type_mismatch_error(self):
        """Test normalizing type mismatch errors."""
        error = """
        type mismatch
        expected
          Nat
        got
          Int
        """

        result = self.normalizer.normalize(error)

        assert result.error_type == "type_mismatch"
        assert "Nat" in result.normalized
        assert "Int" in result.normalized
        assert result.suggestion

    def test_unknown_identifier_error(self):
        """Test normalizing unknown identifier errors."""
        error = "unknown identifier 'my_lemma'"

        result = self.normalizer.normalize(error)

        assert result.error_type == "unknown_identifier"
        assert "my_lemma" in result.normalized
        assert "not in scope" in result.normalized.lower() or "not found" in result.suggestion.lower()

    def test_unknown_tactic_error(self):
        """Test normalizing unknown tactic errors."""
        error = "unknown tactic 'my_custom_tactic'"

        result = self.normalizer.normalize(error)

        assert result.error_type == "unknown_tactic"
        assert "my_custom_tactic" in result.normalized
        assert "simp" in result.suggestion or "standard" in result.suggestion.lower()

    def test_tactic_failed_error(self):
        """Test normalizing tactic failed errors."""
        error = "tactic 'simp' failed, no goals to close"

        result = self.normalizer.normalize(error)

        assert result.error_type == "tactic_failed"
        assert "simp" in result.normalized

    def test_tactic_failed_grind(self):
        """Test grind-specific suggestion."""
        error = "tactic 'grind' failed"

        result = self.normalizer.normalize(error)

        assert "grind" in result.normalized
        # Should have grind-specific advice
        assert "complex" in result.suggestion.lower() or "break" in result.suggestion.lower()

    def test_sorry_error(self):
        """Test normalizing sorry errors."""
        error = "declaration uses 'sorry'"

        result = self.normalizer.normalize(error)

        assert result.error_type == "sorry"
        assert "sorry" in result.normalized.lower()

    def test_motive_error(self):
        """Test normalizing motive type errors."""
        error = "motive is not type correct"

        result = self.normalizer.normalize(error)

        assert result.error_type == "motive_error"
        assert "motive" in result.normalized.lower()
        assert "rcases" in result.suggestion or "induction" in result.suggestion.lower()

    def test_application_mismatch_error(self):
        """Test normalizing application mismatch errors."""
        error = "application type mismatch: f x"

        result = self.normalizer.normalize(error)

        assert result.error_type == "application_mismatch"

    def test_unknown_error(self):
        """Test handling unknown error types."""
        error = "some completely unknown error format"

        result = self.normalizer.normalize(error)

        assert result.error_type == "unknown"
        assert result.normalized  # Should have some content
        assert result.suggestion  # Should have default suggestion

    def test_clean_error_removes_paths(self):
        """Test that error cleaning removes file paths."""
        error = "/workspace/projects/file.lean:10:5: type error"

        result = self.normalizer.normalize(error)

        # Path should be removed or cleaned
        assert "/workspace" not in result.normalized

    def test_format_for_prompt(self):
        """Test formatting error for LLM prompt."""
        error = "tactic 'simp' failed"
        previous_proof = "simp"

        formatted = self.normalizer.format_for_prompt(
            error=error,
            previous_proof=previous_proof,
        )

        assert "Previous Attempt Failed" in formatted
        assert "simp" in formatted
        assert "Suggestion" in formatted

    def test_format_for_prompt_with_context(self):
        """Test formatting error with context."""
        error = "unknown identifier 'my_lemma'"

        formatted = self.normalizer.format_for_prompt(
            error=error,
            previous_proof="exact my_lemma",
            context="Available lemmas: useful_lemma, another_lemma",
        )

        assert "Previous" in formatted
        assert "Suggestion" in formatted


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_normalize_error(self):
        """Test normalize_error convenience function."""
        result = normalize_error("type mismatch expected Nat got Int")

        assert isinstance(result, NormalizedError)
        assert result.error_type == "type_mismatch"

    def test_format_error_for_prompt(self):
        """Test format_error_for_prompt convenience function."""
        formatted = format_error_for_prompt(
            error="tactic 'grind' failed",
            previous_proof="grind",
        )

        assert isinstance(formatted, str)
        assert "grind" in formatted


class TestNormalizedError:
    """Tests for NormalizedError dataclass."""

    def test_normalized_error_creation(self):
        """Test creating NormalizedError."""
        error = NormalizedError(
            original="original error",
            normalized="Normalized: clean error",
            error_type="type_mismatch",
            suggestion="Try something else",
        )

        assert error.original == "original error"
        assert error.normalized == "Normalized: clean error"
        assert error.error_type == "type_mismatch"
        assert error.suggestion == "Try something else"

    def test_normalized_error_default_suggestion(self):
        """Test NormalizedError with default suggestion."""
        error = NormalizedError(
            original="error",
            normalized="error",
            error_type="unknown",
        )

        assert error.suggestion == ""  # Default is empty string
