"""
Error message normalization for LLM-friendly feedback.

Maps Lean's sometimes cryptic error messages to clearer forms
that help the LLM understand what went wrong and how to fix it.

Reference: docs/claude-helpers/resources/POETIQ_deep_dive.md Section 4.4
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class NormalizedError:
    """Normalized error with additional context."""

    original: str
    normalized: str
    error_type: str  # type_mismatch, unknown_identifier, tactic_failed, etc.
    suggestion: str = ""


class ErrorMessageNormalizer:
    """
    Normalize Lean error messages for LLM consumption.

    Poetiq pattern: Binary feedback with clear error messages improves
    the LLM's ability to self-correct.
    """

    # Patterns for common Lean errors
    TYPE_MISMATCH_PATTERN = re.compile(
        r"type mismatch.*expected\s+(.+?)\s+.*got\s+(.+?)(?:\n|$)",
        re.IGNORECASE | re.DOTALL
    )

    UNKNOWN_IDENTIFIER_PATTERN = re.compile(
        r"unknown identifier '([^']+)'",
        re.IGNORECASE
    )

    UNKNOWN_TACTIC_PATTERN = re.compile(
        r"unknown tactic '([^']+)'",
        re.IGNORECASE
    )

    TACTIC_FAILED_PATTERN = re.compile(
        r"tactic '([^']+)' failed",
        re.IGNORECASE
    )

    SORRY_PATTERN = re.compile(
        r"declaration uses 'sorry'",
        re.IGNORECASE
    )

    MOTIVE_PATTERN = re.compile(
        r"motive is not type correct",
        re.IGNORECASE
    )

    APPLICATION_MISMATCH_PATTERN = re.compile(
        r"application type mismatch",
        re.IGNORECASE
    )

    def normalize(self, error: str, context: Optional[str] = None) -> NormalizedError:
        """
        Normalize a Lean error message.

        Args:
            error: Raw Lean error message
            context: Optional goal context for better suggestions

        Returns:
            NormalizedError with clear message and type
        """
        error = error.strip()

        # Try each pattern
        if match := self.TYPE_MISMATCH_PATTERN.search(error):
            return self._handle_type_mismatch(error, match)

        if match := self.UNKNOWN_IDENTIFIER_PATTERN.search(error):
            return self._handle_unknown_identifier(error, match, context)

        if match := self.UNKNOWN_TACTIC_PATTERN.search(error):
            return self._handle_unknown_tactic(error, match)

        if match := self.TACTIC_FAILED_PATTERN.search(error):
            return self._handle_tactic_failed(error, match)

        if self.SORRY_PATTERN.search(error):
            return self._handle_sorry(error)

        if self.MOTIVE_PATTERN.search(error):
            return self._handle_motive_error(error)

        if self.APPLICATION_MISMATCH_PATTERN.search(error):
            return self._handle_application_mismatch(error)

        # Default: clean up the error but don't transform it
        return NormalizedError(
            original=error,
            normalized=self._clean_error(error),
            error_type="unknown",
            suggestion="Review the error message and try a different approach.",
        )

    def _handle_type_mismatch(
        self, error: str, match: re.Match
    ) -> NormalizedError:
        """Handle type mismatch errors."""
        expected = match.group(1).strip()[:50]
        got = match.group(2).strip()[:50]

        normalized = f"Type mismatch: expected `{expected}` but got `{got}`"
        suggestion = (
            "Check that the types align. You may need to use a type conversion "
            "or apply a different lemma."
        )

        return NormalizedError(
            original=error,
            normalized=normalized,
            error_type="type_mismatch",
            suggestion=suggestion,
        )

    def _handle_unknown_identifier(
        self, error: str, match: re.Match, context: Optional[str]
    ) -> NormalizedError:
        """Handle unknown identifier errors."""
        identifier = match.group(1)

        normalized = f"Unknown identifier: `{identifier}` is not in scope"
        suggestion = (
            f"The identifier `{identifier}` was not found. "
            "Check spelling, ensure it's imported, or use a different lemma."
        )

        # If we have context, suggest looking at available lemmas
        if context and "Available" in context:
            suggestion += " See the available lemmas in the context."

        return NormalizedError(
            original=error,
            normalized=normalized,
            error_type="unknown_identifier",
            suggestion=suggestion,
        )

    def _handle_unknown_tactic(
        self, error: str, match: re.Match
    ) -> NormalizedError:
        """Handle unknown tactic errors."""
        tactic = match.group(1)

        normalized = f"Unknown tactic: `{tactic}` is not a valid tactic"
        suggestion = (
            f"The tactic `{tactic}` is not recognized. "
            "Use standard Lean 4 tactics: simp, rfl, exact, apply, "
            "intro, cases, induction, rw, unfold, have, grind, omega, decide."
        )

        return NormalizedError(
            original=error,
            normalized=normalized,
            error_type="unknown_tactic",
            suggestion=suggestion,
        )

    def _handle_tactic_failed(
        self, error: str, match: re.Match
    ) -> NormalizedError:
        """Handle tactic failed errors."""
        tactic = match.group(1)

        normalized = f"Tactic `{tactic}` failed to make progress"
        suggestion = (
            f"The tactic `{tactic}` did not work on this goal. "
            "Try a different tactic or unfold more definitions first."
        )

        # Special suggestions for common tactics
        if tactic == "simp":
            suggestion = (
                "`simp` failed. Try `simp only [...]` with specific lemmas, "
                "or use `simp?` to see what lemmas are available."
            )
        elif tactic == "grind":
            suggestion = (
                "`grind` failed. The goal may be too complex. "
                "Try breaking it down with `have` or using more specific tactics."
            )
        elif tactic in ("omega", "decide"):
            suggestion = (
                f"`{tactic}` failed. Ensure the goal is in the right form. "
                "For omega: arithmetic over integers. For decide: decidable propositions."
            )

        return NormalizedError(
            original=error,
            normalized=normalized,
            error_type="tactic_failed",
            suggestion=suggestion,
        )

    def _handle_sorry(self, error: str) -> NormalizedError:
        """Handle sorry-related errors."""
        return NormalizedError(
            original=error,
            normalized="Proof incomplete: `sorry` placeholder remains",
            error_type="sorry",
            suggestion="The proof still contains `sorry`. Complete the proof with valid tactics.",
        )

    def _handle_motive_error(self, error: str) -> NormalizedError:
        """Handle motive type errors (common in induction/cases)."""
        return NormalizedError(
            original=error,
            normalized="Motive error: the induction/match motive is not type correct",
            error_type="motive_error",
            suggestion=(
                "This often happens with `cases` or `induction`. "
                "Try using `rcases` instead, or provide an explicit motive with `induction ... with`."
            ),
        )

    def _handle_application_mismatch(self, error: str) -> NormalizedError:
        """Handle application type mismatch errors."""
        return NormalizedError(
            original=error,
            normalized="Application error: function applied to wrong type",
            error_type="application_mismatch",
            suggestion=(
                "A function or lemma was applied to an argument of the wrong type. "
                "Check the types match, or use a conversion."
            ),
        )

    def _clean_error(self, error: str) -> str:
        """
        Clean up an error message by removing noise.

        Removes file paths, line numbers, and other details that
        don't help the LLM understand the error.
        """
        # Remove file paths
        cleaned = re.sub(r"/[^\s:]+\.lean:\d+:\d+:", "", error)
        # Remove excessive whitespace
        cleaned = re.sub(r"\s+", " ", cleaned)
        # Truncate if too long
        if len(cleaned) > 500:
            cleaned = cleaned[:500] + "..."
        return cleaned.strip()

    def format_for_prompt(
        self,
        error: str,
        previous_proof: str = "",
        context: Optional[str] = None,
    ) -> str:
        """
        Format an error for inclusion in an LLM retry prompt.

        Args:
            error: Raw Lean error
            previous_proof: The proof that caused the error
            context: Optional goal context

        Returns:
            Formatted string for prompt
        """
        normalized = self.normalize(error, context)

        lines = ["## Previous Attempt Failed", ""]

        if previous_proof:
            lines.extend([
                "**Your previous proof:**",
                "```lean",
                previous_proof.strip(),
                "```",
                "",
            ])

        lines.extend([
            f"**Error:** {normalized.normalized}",
            "",
            f"**Suggestion:** {normalized.suggestion}",
        ])

        return "\n".join(lines)


# Module-level instance for convenience
_normalizer = ErrorMessageNormalizer()


def normalize_error(error: str, context: Optional[str] = None) -> NormalizedError:
    """Convenience function to normalize an error."""
    return _normalizer.normalize(error, context)


def format_error_for_prompt(
    error: str,
    previous_proof: str = "",
    context: Optional[str] = None,
) -> str:
    """Convenience function to format an error for a prompt."""
    return _normalizer.format_for_prompt(error, previous_proof, context)
