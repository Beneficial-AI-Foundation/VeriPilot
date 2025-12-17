"""
Prover Agent module for generating Lean proofs using RAG + LLM.

This module provides the core proof generation functionality:
- Query RAG for relevant lemmas and tactics
- Format context for LLM prompts
- Call LLMs (Gemini, Claude, Aristotle) to generate proofs
"""

from dataclasses import dataclass, field
from typing import Optional

from parser import SorryLocation


@dataclass
class ProofResult:
    """Result of a proof generation attempt."""

    success: bool
    proof_code: str  # Generated proof tactics
    model_used: str  # Which LLM generated this
    rag_context: list[str] = field(default_factory=list)  # RAG results used
    error: Optional[str] = None  # Error message if failed
    attempts: int = 1  # Number of attempts made
    temperature: float = 0.2  # Temperature used for generation


from .rag_query import build_query, retrieve_context
from .context_formatter import format_context
from .prompts import build_system_prompt, build_user_prompt, build_retry_prompt
from .llm_client import LLMClient, generate_proof

__all__ = [
    "ProofResult",
    "build_query",
    "retrieve_context",
    "format_context",
    "build_system_prompt",
    "build_user_prompt",
    "build_retry_prompt",
    "LLMClient",
    "generate_proof",
]
