"""
Multi-provider LLM client for the Prover Agent.

Supports:
- Gemini 3 Pro (via Direct Google API or OpenRouter)
- Claude Sonnet/Opus 4.5 (via Anthropic API)
"""

import os
import time
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

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

            # OpenRouter requires specific headers for proper API access
            # See: https://openrouter.ai/docs/api-reference
            default_headers = {}
            if provider.base_url and "openrouter" in provider.base_url:
                default_headers = {
                    "HTTP-Referer": "https://github.com/VeriPilot/veripilot",
                    "X-Title": "VeriPilot",
                }

            self._openai_client = AsyncOpenAI(
                api_key=api_key,
                base_url=provider.base_url,
                default_headers=default_headers if default_headers else None,
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
        max_tokens: int = 16384,  # Increased for complex proofs (IsZero, Add, etc.)
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
        from openai import APIError, APIConnectionError, RateLimitError, AuthenticationError

        client = self._get_openai_client(provider)

        # Log request details
        logger.info(
            f"OpenRouter request: model={provider.model}, "
            f"base_url={provider.base_url}, temp={temperature}, max_tokens={max_tokens}"
        )
        logger.debug(f"System prompt length: {len(system_prompt)} chars")
        logger.debug(f"User prompt length: {len(user_prompt)} chars")

        start_time = time.time()
        try:
            response = await client.chat.completions.create(
                model=provider.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            latency = time.time() - start_time

            # Get choice details for debugging
            choice = response.choices[0] if response.choices else None
            content = choice.message.content if choice and choice.message else ""
            content = content or ""

            # Get finish reason
            finish_reason = choice.finish_reason if choice else "no_choice"

            # Log response details
            logger.info(
                f"OpenRouter response: latency={latency:.2f}s, "
                f"response_length={len(content)} chars, finish_reason={finish_reason}"
            )
            if hasattr(response, 'usage') and response.usage:
                logger.info(
                    f"Token usage: prompt={response.usage.prompt_tokens}, "
                    f"completion={response.usage.completion_tokens}, "
                    f"total={response.usage.total_tokens}"
                )

            # Detailed diagnostics for empty response
            if not content:
                logger.warning(
                    f"OpenRouter returned empty response content! "
                    f"finish_reason={finish_reason}, "
                    f"num_choices={len(response.choices) if response.choices else 0}"
                )
                # Specific warning for length limit
                if finish_reason == "length":
                    logger.error(
                        f"Response cut off due to max_tokens limit ({max_tokens}). "
                        f"The model started generating but hit the token limit before producing content. "
                        f"Consider increasing max_tokens."
                    )
                # Check for refusal or other message fields
                if choice and choice.message:
                    msg = choice.message
                    refusal = getattr(msg, 'refusal', None)
                    if refusal:
                        logger.warning(f"Model refusal: {refusal}")
                    # Log all message attributes for debugging
                    logger.debug(f"Message attributes: {vars(msg) if hasattr(msg, '__dict__') else msg}")
                # Check for OpenRouter-specific error in response
                if hasattr(response, 'error') and response.error:
                    logger.error(f"OpenRouter error field: {response.error}")

            return content

        except AuthenticationError as e:
            logger.error(f"OpenRouter auth error: {e}")
            raise ValueError(f"OpenRouter authentication failed. Check {provider.env_key}.") from e
        except RateLimitError as e:
            logger.error(f"OpenRouter rate limit: {e}")
            raise ValueError(f"OpenRouter rate limit exceeded for {provider.model}.") from e
        except APIConnectionError as e:
            logger.error(f"OpenRouter connection error: {e}")
            raise ValueError(f"Failed to connect to OpenRouter: {e}") from e
        except APIError as e:
            # Log full error details for debugging
            logger.error(
                f"OpenRouter API error: status={getattr(e, 'status_code', 'N/A')}, "
                f"message={e.message if hasattr(e, 'message') else str(e)}"
            )
            # Check for common OpenRouter-specific errors
            error_msg = str(e).lower()
            if "model" in error_msg and ("not found" in error_msg or "not available" in error_msg):
                raise ValueError(
                    f"Model '{provider.model}' not available on OpenRouter. "
                    f"Check model name or account tier (some models require premium access)."
                ) from e
            if "quota" in error_msg or "credits" in error_msg:
                raise ValueError(
                    f"OpenRouter quota/credits exhausted for {provider.model}."
                ) from e
            raise ValueError(f"OpenRouter API error: {e}") from e
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Unexpected OpenRouter error after {latency:.2f}s: {type(e).__name__}: {e}")
            raise

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


async def generate_proof(
    sorry: SorryLocation,
    file_content: str,
    rag=None,  # LeanRAG instance
    model: str = "gemini",
    temperature: float = 0.2,
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
        temperature: Sampling temperature (0.2 recommended for proofs)
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

    # Standard LLM flow
    client = LLMClient()

    # Build initial prompt
    prompt = build_user_prompt(sorry, context)

    for attempt in range(1, max_attempts + 1):
        try:
            response = await client.generate(prompt, model=model, temperature=temperature)
            proof = extract_proof_from_response(response)

            if proof:
                return ProofResult(
                    success=True,  # Note: actual verification happens in Phase 3
                    proof_code=proof,
                    model_used=model,
                    rag_context=rag_context,
                    attempts=attempt,
                    temperature=temperature,
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
                    temperature=temperature,
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
        temperature=temperature,
    )
