"""
Unit tests for the Prover Agent module.

Tests:
- RAG query formulation
- Context formatting
- Prompt generation
- LLM client (with mocks)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field

# Test fixtures
@dataclass
class MockSorryLocation:
    """Mock SorryLocation for testing."""
    file_path: str = "/test/file.lean"
    line: int = 10
    column: int = 5
    theorem_name: str = "test_theorem"
    theorem_signature: str = "theorem test_theorem (x : Nat) : x + 0 = x := by"
    proof_prefix: str = "  intro h"
    namespace: str = "Test"
    imports: list = field(default_factory=lambda: ["import Mathlib.Algebra.Basic"])


@dataclass
class MockRetrievalResult:
    """Mock RetrievalResult for testing."""
    name: str = "add_zero"
    full_name: str = "Nat.add_zero"
    type_signature: str = "(n : Nat) : n + 0 = n"
    proof: str = "rfl"
    doc_string: str = "Addition with zero"
    namespace: str = "Nat"
    score: float = 0.9


class TestRagQuery:
    """Tests for rag_query.py functions."""

    def test_extract_keywords_basic(self):
        """Test keyword extraction from Lean code."""
        from agent.rag_query import extract_keywords

        text = "theorem foo : Nat → Bool := by simp"
        keywords = extract_keywords(text)

        assert "simp" in keywords
        assert "Nat" in keywords
        assert "Bool" in keywords

    def test_extract_keywords_tactics(self):
        """Test extraction of known tactics."""
        from agent.rag_query import extract_keywords

        text = "unfold f; progress; grind; omega"
        keywords = extract_keywords(text)

        assert "unfold" in keywords
        assert "progress" in keywords
        assert "grind" in keywords
        assert "omega" in keywords

    def test_extract_keywords_aeneas(self):
        """Test extraction of Aeneas-specific terms."""
        from agent.rag_query import extract_keywords

        text = "Scalar U64 Array Result ok err"
        keywords = extract_keywords(text)

        assert "Scalar" in keywords
        assert "U64" in keywords
        assert "Array" in keywords
        assert "Result" in keywords

    def test_build_query(self):
        """Test query building from SorryLocation."""
        from agent.rag_query import build_query

        sorry = MockSorryLocation()
        query = build_query(sorry)

        assert sorry.theorem_name in query.text
        assert query.top_k == 6

    def test_build_query_with_goal(self):
        """Test query building with goal state."""
        from agent.rag_query import build_query

        sorry = MockSorryLocation()
        goal = "⊢ x + 0 = x"
        query = build_query(sorry, goal=goal)

        assert query.goal_state == goal

    def test_prioritize_results(self):
        """Test result prioritization."""
        from agent.rag_query import prioritize_results

        sorry = MockSorryLocation(theorem_name="add_spec")
        results = [
            MockRetrievalResult(name="unrelated", score=0.8),
            MockRetrievalResult(name="add_spec_helper", score=0.5),
        ]

        prioritized = prioritize_results(results, sorry)

        # add_spec_helper should be boosted due to name similarity
        assert prioritized[0].name == "add_spec_helper"


class TestContextFormatter:
    """Tests for context_formatter.py functions."""

    def test_format_rag_results_empty(self):
        """Test formatting empty RAG results."""
        from agent.context_formatter import format_rag_results

        result = format_rag_results([])
        assert result == ""

    def test_format_rag_results(self):
        """Test formatting RAG results."""
        from agent.context_formatter import format_rag_results

        results = [MockRetrievalResult()]
        formatted = format_rag_results(results)

        assert "Available Lemmas" in formatted
        assert "Nat.add_zero" in formatted
        assert "(n : Nat)" in formatted

    def test_format_file_context(self):
        """Test file context formatting."""
        from agent.context_formatter import format_file_context

        sorry = MockSorryLocation()
        file_content = "-- file content"
        formatted = format_file_context(sorry, file_content)

        assert "File Context" in formatted
        assert sorry.theorem_signature in formatted
        assert "sorry  -- FILL THIS" in formatted
        assert f"Line {sorry.line}" in formatted

    def test_format_file_context_with_imports(self):
        """Test file context includes imports."""
        from agent.context_formatter import format_file_context

        sorry = MockSorryLocation()
        formatted = format_file_context(sorry, "")

        assert "Imports" in formatted
        assert "Mathlib.Algebra.Basic" in formatted

    def test_format_proof_hints(self):
        """Test proof hints formatting."""
        from agent.context_formatter import format_proof_hints

        sorry = MockSorryLocation()
        hints = format_proof_hints(sorry)

        assert "Proof Strategy" in hints
        assert "unfold" in hints.lower()
        assert "progress" in hints.lower()
        assert "grind" in hints.lower()

    def test_format_proof_hints_spec_theorem(self):
        """Test proof hints for spec theorems."""
        from agent.context_formatter import format_proof_hints

        sorry = MockSorryLocation(theorem_name="my_function_spec")
        hints = format_proof_hints(sorry)

        assert "spec theorem" in hints.lower()

    def test_format_context(self):
        """Test full context formatting."""
        from agent.context_formatter import format_context

        sorry = MockSorryLocation()
        results = [MockRetrievalResult()]
        formatted = format_context(sorry, "-- content", results)

        # Should include all sections
        assert "Available Lemmas" in formatted
        assert "File Context" in formatted
        assert "Proof Strategy" in formatted


class TestPrompts:
    """Tests for prompts.py functions."""

    def test_build_system_prompt_default(self):
        """Test default system prompt."""
        from agent.prompts import build_system_prompt

        prompt = build_system_prompt()
        assert "Lean" in prompt
        assert "proof" in prompt.lower()

    def test_build_system_prompt_gemini(self):
        """Test Gemini-specific system prompt."""
        from agent.prompts import build_system_prompt

        prompt = build_system_prompt("gemini")
        assert "tactic" in prompt.lower()

    def test_build_system_prompt_claude(self):
        """Test Claude-specific system prompt."""
        from agent.prompts import build_system_prompt

        prompt = build_system_prompt("claude")
        assert "tactic" in prompt.lower()

    def test_build_user_prompt(self):
        """Test user prompt building."""
        from agent.prompts import build_user_prompt

        sorry = MockSorryLocation()
        context = "## Context\nSome context here"

        prompt = build_user_prompt(sorry, context)

        assert f"line {sorry.line}" in prompt
        assert sorry.theorem_name in prompt
        assert context in prompt

    def test_build_user_prompt_with_goal(self):
        """Test user prompt with goal state."""
        from agent.prompts import build_user_prompt

        sorry = MockSorryLocation()
        goal = "⊢ x + 0 = x"

        prompt = build_user_prompt(sorry, "", goal=goal)

        assert "Current Goal" in prompt
        assert goal in prompt

    def test_build_retry_prompt(self):
        """Test retry prompt building."""
        from agent.prompts import build_retry_prompt

        sorry = MockSorryLocation()
        prev_proof = "simp"
        error = "type mismatch"

        prompt = build_retry_prompt(sorry, "", prev_proof, error, attempt=2)

        assert "Attempt 2" in prompt
        assert prev_proof in prompt
        assert error in prompt
        assert "simpler" in prompt.lower()  # Attempt 2 guidance

    def test_extract_proof_plain(self):
        """Test extracting plain proof."""
        from agent.prompts import extract_proof_from_response

        response = "simp\nrfl"
        proof = extract_proof_from_response(response)

        assert proof == "simp\nrfl"

    def test_extract_proof_markdown(self):
        """Test extracting proof from markdown."""
        from agent.prompts import extract_proof_from_response

        response = "Here is the proof:\n```lean\nsimp\nrfl\n```\nExplanation..."
        proof = extract_proof_from_response(response)

        assert "simp" in proof
        assert "rfl" in proof
        assert "Explanation" not in proof

    def test_extract_proof_with_preamble(self):
        """Test extracting proof with preamble text."""
        from agent.prompts import extract_proof_from_response

        response = "Here is the proof:\nsimp\nrfl"
        proof = extract_proof_from_response(response)

        assert "simp" in proof
        assert "Here is" not in proof


class TestLLMClient:
    """Tests for llm_client.py."""

    def test_provider_config_exists(self):
        """Test that provider configs are defined."""
        from agent.llm_client import PROVIDERS

        assert "gemini" in PROVIDERS
        assert "claude" in PROVIDERS
        assert "aristotle" in PROVIDERS

    def test_gemini_config(self):
        """Test Gemini provider configuration (Direct Google API)."""
        from agent.llm_client import PROVIDERS

        gemini = PROVIDERS["gemini"]
        assert gemini.client_type == "google"
        assert gemini.model == "gemini-3-pro-preview"
        assert gemini.env_key == "GOOGLE_API_KEY"

    def test_gemini_openrouter_config(self):
        """Test Gemini OpenRouter fallback configuration."""
        from agent.llm_client import PROVIDERS

        gemini = PROVIDERS["gemini-openrouter"]
        assert gemini.client_type == "openai"
        assert "openrouter" in gemini.base_url
        assert gemini.env_key == "OPENROUTER_API_KEY"

    def test_claude_config(self):
        """Test Claude provider configuration."""
        from agent.llm_client import PROVIDERS

        claude = PROVIDERS["claude"]
        assert claude.client_type == "anthropic"
        assert claude.env_key == "ANTHROPIC_API_KEY"

    @pytest.mark.asyncio
    async def test_generate_unknown_model(self):
        """Test error handling for unknown model."""
        from agent.llm_client import LLMClient

        client = LLMClient()

        with pytest.raises(ValueError, match="Unknown model"):
            await client.generate("test", model="unknown")

    @pytest.mark.asyncio
    async def test_generate_aristotle_raises(self):
        """Test that Aristotle raises for prompt-based API."""
        from agent.llm_client import LLMClient

        client = LLMClient()

        with pytest.raises(ValueError, match="file-based API"):
            await client.generate("test", model="aristotle")


class TestProofResult:
    """Tests for ProofResult dataclass."""

    def test_proof_result_success(self):
        """Test successful ProofResult."""
        from agent import ProofResult

        result = ProofResult(
            success=True,
            proof_code="simp",
            model_used="gemini",
            rag_context=["lemma1"],
        )

        assert result.success
        assert result.proof_code == "simp"
        assert result.error is None

    def test_proof_result_failure(self):
        """Test failed ProofResult."""
        from agent import ProofResult

        result = ProofResult(
            success=False,
            proof_code="",
            model_used="gemini",
            error="API error",
        )

        assert not result.success
        assert result.error == "API error"


# Integration test (requires mocking)
class TestGenerateProofIntegration:
    """Integration tests for generate_proof function."""

    @pytest.mark.asyncio
    async def test_generate_proof_mock(self):
        """Test generate_proof with mocked LLM."""
        from agent.llm_client import generate_proof, LLMClient

        sorry = MockSorryLocation()
        file_content = "-- test file"

        # Mock the LLM client
        with patch.object(LLMClient, 'generate', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "simp\nrfl"

            result = await generate_proof(sorry, file_content, model="gemini")

            assert result.success
            assert "simp" in result.proof_code
            assert result.model_used == "gemini"
