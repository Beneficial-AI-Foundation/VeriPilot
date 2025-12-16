"""
VeriPilot Agent Tools.

This module provides external tools for the prover agent:
- LeanSearch: Semantic search for Lean 4 theorems and tactics
- Kimina: Proof verification via Kimina Lean Server
"""

from .lean_search import (
    LeanSearchResult,
    KiminaVerifyResult,
    leansearch_cli,
    kimina_verify,
    kimina_search,
    search_lean_library,
    is_kimina_available,
    is_leansearch_installed,
    run_search_sync,
)

__all__ = [
    "LeanSearchResult",
    "KiminaVerifyResult",
    "leansearch_cli",
    "kimina_verify",
    "kimina_search",
    "search_lean_library",
    "is_kimina_available",
    "is_leansearch_installed",
    "run_search_sync",
]
