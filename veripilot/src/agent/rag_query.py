"""
RAG query formulation for the Prover Agent.

Converts SorryLocation context into effective RAG queries
to retrieve relevant lemmas, tactics, and proof patterns.
"""

import re
from typing import Optional

from interfaces.rag_provider import RetrievalQuery, RetrievalResult
from parser import SorryLocation

# Lean-specific terms to extract from code
LEAN_TACTICS = {
    "simp", "rfl", "exact", "apply", "intro", "intros", "cases", "induction",
    "constructor", "ext", "funext", "rw", "rewrite", "have", "let", "show",
    "ring", "omega", "decide", "native_decide", "norm_num", "linarith",
    "unfold", "progress", "grind", "scalar_tac", "aesop", "trivial",
}

AENEAS_TERMS = {
    "progress", "scalar_tac", "Scalar", "U8", "U16", "U32", "U64", "I8", "I16",
    "I32", "I64", "Usize", "Result", "Array", "Slice", "Vec", "ok", "err",
}


def build_query(
    sorry: SorryLocation,
    goal: Optional[str] = None,
    top_k: int = 6,
) -> RetrievalQuery:
    """
    Build a RAG query from a SorryLocation.

    Extracts keywords from:
    - Theorem name and signature
    - Proof prefix (existing tactics)
    - Optional goal state

    Args:
        sorry: The sorry location with context
        goal: Optional goal state text
        top_k: Number of results to retrieve

    Returns:
        RetrievalQuery configured for the sorry context
    """
    # Extract meaningful keywords
    keywords = extract_keywords(sorry.theorem_signature)
    keywords.extend(extract_keywords(sorry.proof_prefix))

    if goal:
        keywords.extend(extract_keywords(goal))

    # Build query text - combine theorem name with key terms
    query_parts = [sorry.theorem_name]
    query_parts.extend(keywords[:10])  # Limit keywords

    query_text = " ".join(query_parts)

    return RetrievalQuery(
        text=query_text,
        top_k=top_k,
        type_hint=_extract_type_hint(sorry.theorem_signature),
        goal_state=goal,
    )


def extract_keywords(text: str) -> list[str]:
    """
    Extract Lean-specific keywords from code text.

    Finds:
    - Tactic names
    - Type names (capitalized)
    - Aeneas-specific terms
    - Lemma references

    Args:
        text: Lean code text

    Returns:
        List of extracted keywords
    """
    keywords = []

    # Find all identifiers (alphanumeric + underscore)
    identifiers = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text)

    for ident in identifiers:
        # Skip common noise words
        if ident in {"by", "do", "if", "then", "else", "match", "with", "fun", "let", "in"}:
            continue

        # Include known tactics
        if ident.lower() in LEAN_TACTICS:
            keywords.append(ident.lower())
            continue

        # Include Aeneas terms
        if ident in AENEAS_TERMS:
            keywords.append(ident)
            continue

        # Include capitalized type names
        if ident[0].isupper() and len(ident) > 2:
            keywords.append(ident)
            continue

        # Include names with underscores (likely lemma names)
        if "_" in ident and len(ident) > 5:
            keywords.append(ident)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)

    return unique


def _extract_type_hint(signature: str) -> Optional[str]:
    """
    Extract a type hint from a theorem signature.

    Looks for the return type after the final ':'.

    Args:
        signature: Full theorem signature

    Returns:
        Type hint string or None
    """
    # Find return type (after final : and before :=)
    match = re.search(r":\s*([^:=]+?)\s*:=", signature)
    if match:
        return match.group(1).strip()
    return None


async def retrieve_context(
    sorry: SorryLocation,
    rag,  # LeanRAG instance
    goal: Optional[str] = None,
    top_k: int = 6,
) -> list[RetrievalResult]:
    """
    Retrieve relevant context from RAG for a sorry location.

    Args:
        sorry: The sorry location with context
        rag: Initialized LeanRAG instance
        goal: Optional goal state
        top_k: Number of results

    Returns:
        List of relevant RetrievalResults
    """
    query = build_query(sorry, goal, top_k)
    results = await rag.retrieve(query.text, top_k=query.top_k)
    return prioritize_results(results, sorry)


def prioritize_results(
    results: list[RetrievalResult],
    sorry: SorryLocation,
) -> list[RetrievalResult]:
    """
    Re-rank results by relevance to the specific sorry.

    Prioritizes:
    1. Results with similar names to theorem
    2. Results in same namespace
    3. Results with proof examples

    Args:
        results: Original RAG results
        sorry: Sorry location for context

    Returns:
        Re-ranked results
    """
    if not results:
        return results

    def score_result(r: RetrievalResult) -> float:
        score = r.score

        # Boost results with similar names
        if sorry.theorem_name.lower() in r.name.lower():
            score += 0.3
        elif any(word in r.name.lower() for word in sorry.theorem_name.lower().split("_")):
            score += 0.1

        # Boost results in same namespace
        if sorry.namespace and r.namespace == sorry.namespace:
            score += 0.2

        # Boost results with proofs (we can learn from them)
        if r.proof:
            score += 0.1

        # Boost spec theorems (common in Aeneas)
        if "_spec" in r.name.lower():
            score += 0.15

        return score

    # Re-sort by adjusted score
    return sorted(results, key=score_result, reverse=True)
