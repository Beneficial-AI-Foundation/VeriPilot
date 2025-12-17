"""
Multi-provider LLM client for the Prover Agent.

Supports:
- Gemini 3 Pro (via Direct Google API or OpenRouter)
- Claude Sonnet/Opus 4.5 (via Anthropic API)
- Aristotle (via aristotlelib - file-based)
"""

import os
from typing import Optional
from dataclasses import dataclass

from parser import SorryLocation
from .prompts import (
    build_system_prompt,
    build_user_prompt,
    build_retry_prompt,
    extract_proof_from_response,
)
from .context_formatter import format_context
from .rag_query import retrieve_context


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    name: str
    client_type: str  # "google", "openai", "anthropic", "aristotle"
    model: str
    env_key: str
    base_url: Optional[str] = None


# Provider configurations
PROVIDERS = {
    # Primary: Direct Google API (preferred when GOOGLE_API_KEY is set)
    # Using Gemini 3.0 Pro Preview as default for VeriPilot verifier agent
    "gemini": ProviderConfig(
        name="Gemini 3.0 Pro",
        client_type="google",
        model="gemini-3-pro-preview",
        env_key="GOOGLE_API_KEY",
    ),
    # Fallback: OpenRouter (uses OPENROUTER_API_KEY)
    "gemini-openrouter": ProviderConfig(
        name="Gemini 3 Pro (OpenRouter)",
        client_type="openai",
        model="google/gemini-3-pro-preview",
        env_key="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
    ),
    "claude": ProviderConfig(
        name="Claude Sonnet 4.5",
        client_type="anthropic",
        model="claude-sonnet-4-5",
        env_key="ANTHROPIC_API_KEY",
    ),
    "claude-opus": ProviderConfig(
        name="Claude Opus 4.5",
        client_type="anthropic",
        model="claude-opus-4-5",
        env_key="ANTHROPIC_API_KEY",
    ),
    "aristotle": ProviderConfig(
        name="Aristotle",
        client_type="aristotle",
        model="aristotle",
        env_key="ARISTOTLE_API_KEY",
    ),
}


class LLMClient:
    """
    Unified async LLM client for multiple providers.

    Usage:
        client = LLMClient()
        response = await client.generate("prompt", model="gemini")
    """

    def __init__(self):
        """Initialize the LLM client."""
        self._openai_client = None
        self._anthropic_client = None
        self._google_client = None

    def _get_google_client(self):
        """Get or create Google GenAI client."""
        if self._google_client is None:
            try:
                from google import genai
            except ImportError:
                raise ImportError("google-genai package not installed. Run: pip install google-genai")

            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set in environment")

            self._google_client = genai.Client(api_key=api_key)
        return self._google_client

    def _get_openai_client(self, provider: ProviderConfig):
        """Get or create OpenAI-compatible client."""
        if self._openai_client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")

            api_key = os.getenv(provider.env_key)
            if not api_key:
                raise ValueError(f"{provider.env_key} not set in environment")

            self._openai_client = AsyncOpenAI(
                api_key=api_key,
                base_url=provider.base_url,
            )
        return self._openai_client

    def _get_anthropic_client(self):
        """Get or create Anthropic client."""
        if self._anthropic_client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set in environment")

            self._anthropic_client = AsyncAnthropic(api_key=api_key)
        return self._anthropic_client

    async def generate(
        self,
        user_prompt: str,
        model: str = "gemini",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """
        Generate a response from an LLM.

        Args:
            user_prompt: The user message/prompt
            model: Model key (gemini, claude, claude-opus, aristotle)
            system_prompt: Optional system prompt (uses default if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Generated text response
        """
        if model not in PROVIDERS:
            raise ValueError(f"Unknown model: {model}. Available: {list(PROVIDERS.keys())}")

        provider = PROVIDERS[model]

        if system_prompt is None:
            system_prompt = build_system_prompt(model)

        if provider.client_type == "google":
            return await self._generate_google(
                provider, user_prompt, system_prompt, temperature, max_tokens
            )
        elif provider.client_type == "openai":
            return await self._generate_openai(
                provider, user_prompt, system_prompt, temperature, max_tokens
            )
        elif provider.client_type == "anthropic":
            return await self._generate_anthropic(
                provider, user_prompt, system_prompt, temperature, max_tokens
            )
        elif provider.client_type == "aristotle":
            raise ValueError("Aristotle uses file-based API. Use generate_with_aristotle() instead.")
        else:
            raise ValueError(f"Unknown client type: {provider.client_type}")

    async def _generate_google(
        self,
        provider: ProviderConfig,
        user_prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate using Google GenAI API (direct)."""
        from google.genai import types
        from google.api_core import exceptions as google_exceptions
        import asyncio

        client = self._get_google_client()

        # Combine system prompt and user prompt for Google's format
        full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt

        # Google GenAI uses synchronous API, run in executor
        def _sync_generate():
            try:
                response = client.models.generate_content(
                    model=provider.model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                return response.text if response.text else ""
            except google_exceptions.NotFound as e:
                raise ValueError(f"Model '{provider.model}' not found. Check model name or API access.") from e
            except google_exceptions.PermissionDenied as e:
                raise ValueError(f"Permission denied for model '{provider.model}'. Check API key or quota.") from e
            except google_exceptions.ResourceExhausted as e:
                raise ValueError(f"Quota exceeded for model '{provider.model}'. {str(e)}") from e
            except google_exceptions.GoogleAPIError as e:
                raise ValueError(f"Google API error: {str(e)}") from e

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_generate)

    async def _generate_openai(
        self,
        provider: ProviderConfig,
        user_prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate using OpenAI-compatible API (OpenRouter)."""
        client = self._get_openai_client(provider)

        response = await client.chat.completions.create(
            model=provider.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content or ""

    async def _generate_anthropic(
        self,
        provider: ProviderConfig,
        user_prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate using Anthropic API."""
        from anthropic import APIError, NotFoundError, AuthenticationError

        client = self._get_anthropic_client()

        try:
            response = await client.messages.create(
                model=provider.model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Extract text from response
            if response.content and len(response.content) > 0:
                return response.content[0].text
            return ""
        except NotFoundError as e:
            raise ValueError(f"Model '{provider.model}' not found. Check model name in llm_client.py PROVIDERS config.") from e
        except AuthenticationError as e:
            raise ValueError(f"ANTHROPIC_API_KEY authentication failed: {str(e)}") from e
        except APIError as e:
            raise ValueError(f"Anthropic API error: {str(e)}") from e


async def generate_with_aristotle(
    sorry: SorryLocation,
    file_content: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate proof using Aristotle (file-based API).

    Aristotle works differently - it takes a full file and returns
    the solved version. We extract the proof for the specific sorry.

    Args:
        sorry: The sorry location
        file_content: Full file content
        output_path: Optional path for output file

    Returns:
        Generated proof tactics
    """
    try:
        import aristotlelib
    except ImportError:
        raise ImportError("aristotlelib not installed. Run: pip install aristotlelib")

    api_key = os.getenv("ARISTOTLE_API_KEY")
    if not api_key:
        raise ValueError("ARISTOTLE_API_KEY not set in environment")

    # Aristotle needs the actual file path
    solution_path = await aristotlelib.Project.prove_from_file(
        input_file_path=sorry.file_path,
        auto_add_imports=True,
        wait_for_completion=True,
        polling_interval_seconds=30,
        output_file_path=output_path,
    )

    # Read the solution and extract the proof
    if solution_path and os.path.exists(solution_path):
        with open(solution_path) as f:
            solution_content = f.read()
        return _extract_proof_from_solution(solution_content, sorry)

    return ""


def _extract_proof_from_solution(solution: str, sorry: SorryLocation) -> str:
    """
    Extract the proof for a specific theorem from Aristotle's solution.

    Args:
        solution: Full solved file content
        sorry: The original sorry location

    Returns:
        Extracted proof tactics
    """
    import re

    # Find the theorem in the solution
    # Pattern: theorem <name> ... := by
    pattern = rf"(theorem|lemma|def)\s+{re.escape(sorry.theorem_name)}[^:]*:=\s*by\s*\n((?:[ \t]+.*\n)*)"
    match = re.search(pattern, solution, re.MULTILINE)

    if match:
        proof_block = match.group(2)
        # Clean up the proof
        lines = proof_block.strip().split("\n")
        return "\n".join(line.strip() for line in lines if line.strip())

    return ""


async def generate_proof(
    sorry: SorryLocation,
    file_content: str,
    rag=None,  # LeanRAG instance
    model: str = "gemini",
    max_attempts: int = 4,
) -> "ProofResult":
    """
    Generate a proof for a sorry location.

    This is the main entry point for proof generation.

    Args:
        sorry: The sorry location to fill
        file_content: Full file content
        rag: Optional LeanRAG instance for context retrieval
        model: LLM model to use
        max_attempts: Maximum retry attempts

    Returns:
        ProofResult with success status and proof code
    """
    # Import here to avoid circular dependency
    from . import ProofResult

    # Retrieve RAG context if available
    rag_results = []
    if rag is not None:
        try:
            rag_results = await retrieve_context(sorry, rag)
        except Exception:
            pass  # Continue without RAG if it fails

    # Format context
    context = format_context(sorry, file_content, rag_results)
    rag_context = [r.full_name for r in rag_results]

    # Handle Aristotle specially
    if model == "aristotle":
        try:
            proof = await generate_with_aristotle(sorry, file_content)
            return ProofResult(
                success=bool(proof),
                proof_code=proof,
                model_used="aristotle",
                rag_context=rag_context,
                error=None if proof else "Aristotle returned empty solution",
            )
        except Exception as e:
            return ProofResult(
                success=False,
                proof_code="",
                model_used="aristotle",
                rag_context=rag_context,
                error=str(e),
            )

    # Standard LLM flow
    client = LLMClient()

    # Build initial prompt
    prompt = build_user_prompt(sorry, context)

    for attempt in range(1, max_attempts + 1):
        try:
            response = await client.generate(prompt, model=model)
            proof = extract_proof_from_response(response)

            if proof:
                return ProofResult(
                    success=True,  # Note: actual verification happens in Phase 3
                    proof_code=proof,
                    model_used=model,
                    rag_context=rag_context,
                    attempts=attempt,
                )

            # Empty response - build retry prompt
            if attempt < max_attempts:
                prompt = build_retry_prompt(
                    sorry, context, "", "Empty response from model", attempt + 1
                )

        except Exception as e:
            if attempt >= max_attempts:
                return ProofResult(
                    success=False,
                    proof_code="",
                    model_used=model,
                    rag_context=rag_context,
                    error=str(e),
                    attempts=attempt,
                )
            # Retry on error
            prompt = build_retry_prompt(
                sorry, context, "", f"Error: {e}", attempt + 1
            )

    return ProofResult(
        success=False,
        proof_code="",
        model_used=model,
        rag_context=rag_context,
        error="Max attempts reached",
        attempts=max_attempts,
    )
