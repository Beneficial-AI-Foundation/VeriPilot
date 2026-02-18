"""
LangGraph nodes for ReAct proof verification agent.

Implements the core nodes of the ReAct loop:
- reasoning_node: Generate thought and plan next action
- execution_node: Execute tactic via LSP verification
- observation_node: Parse verification feedback
- router_node: Decide next step (continue/backtrack/terminate)

Each node takes ProofState as input and returns a partial state update.
LangGraph merges updates using Annotated reducers (operator.add for lists).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, TYPE_CHECKING

from .state import (
    ActionRecord,
    AttemptRecord,
    GoalAnalysis,
    HypothesisInfo,
    ObservationRecord,
    ProofState,
    ProofStatus,
    RecoveryRecord,
    SubTaskRecord,
    ThoughtRecord,
    add_action,
    add_observation,
    add_thought,
    dict_to_sorry,
    should_backtrack,
)
from .error_recovery import (
    ErrorRecoveryController,
    ErrorSeverity,
    RecoveryContext,
    RecoveryStrategy,
    TacticModifier,
)
from .roma import (
    Atomizer,
    GoalComplexity,
    GoalComplexityAnalyzer,
    RomaAggregator,
    RomaPlanner,
    SubProof,
)

if TYPE_CHECKING:
    from verifier.verifier_service import VerifierService

logger = logging.getLogger(__name__)

# Dedicated logger for LLM I/O
llm_logger = logging.getLogger("veripilot.llm_output")

# Strategy shift: after this many consecutive outer failures with zero
# successful candidates, inject a "try something fundamentally different"
# prompt to break out of repetitive failure patterns.
STRATEGY_SHIFT_THRESHOLD = 5

STRATEGY_SHIFT_PROMPT = """## Strategy Shift Required

The following {n} approaches have ALL FAILED on this goal:

{failed_summary}

These approaches are NOT WORKING. You MUST try something fundamentally different:
- If you were using `simp`/`rfl`/automation, try manual `unfold` + explicit rewriting with `rw`
- If you were applying lemmas directly, try `have` blocks with intermediate steps
- If you were doing case analysis, try a direct algebraic approach (`ring`, `omega`)
- If you were using `rw`, try `conv` or `simp only [specific_lemma]`
- If everything failed, try term-mode proof (`exact ...`) or a small helper lemma

Do NOT repeat any approach from the failed list above.
"""


def log_llm_request(goal_state: str, model: str, context_summary: str = "") -> None:
    """Log LLM request details."""
    llm_logger.info(f"=== LLM REQUEST ({model}) ===")
    llm_logger.info(f"Goal state:\n{goal_state[:500]}")
    if context_summary:
        llm_logger.info(f"Context: {context_summary}")


def log_llm_response(tactics: list[str], model: str) -> None:
    """Log LLM response (proposed tactics)."""
    llm_logger.info(f"=== LLM RESPONSE ({model}) ===")
    llm_logger.info(f"Proposed {len(tactics)} tactics:")
    for i, tactic in enumerate(tactics, 1):
        preview = tactic[:150] + "..." if len(tactic) > 150 else tactic
        llm_logger.info(f"  {i}. {preview}")


# ==============================================================================
# Prompt Loading
# ==============================================================================

_ITERATIVE_SYSTEM_PROMPT_CACHE: str | None = None


def _load_iterative_system_prompt() -> str:
    """
    Load the iterative system prompt from prompts/verifier/iterative_system_v1.md.

    Uses caching for performance. Falls back to minimal prompt if file not found.
    """
    global _ITERATIVE_SYSTEM_PROMPT_CACHE

    if _ITERATIVE_SYSTEM_PROMPT_CACHE is not None:
        return _ITERATIVE_SYSTEM_PROMPT_CACHE

    try:
        from agent.prompt_loader import load_latest_prompt
        _ITERATIVE_SYSTEM_PROMPT_CACHE = load_latest_prompt("iterative_system")
        logger.debug("Loaded iterative system prompt from file")
    except (ImportError, FileNotFoundError) as e:
        logger.warning(f"Could not load iterative_system prompt: {e}, using fallback")
        _ITERATIVE_SYSTEM_PROMPT_CACHE = (
            "You are a Lean 4 theorem prover. Generate 1-3 atomic proof snippets "
            "for the given goal state. Each snippet should be a small, self-contained "
            "block of proof code. Separate snippets with --- markers."
        )

    return _ITERATIVE_SYSTEM_PROMPT_CACHE


_REACT_SYSTEM_PROMPT_CACHE: str | None = None


def _load_react_system_prompt() -> str:
    """
    Load the ReAct system prompt from prompts/verifier/react_system_v1.md.

    Uses caching for performance. Falls back to minimal prompt if file not found.
    """
    global _REACT_SYSTEM_PROMPT_CACHE

    if _REACT_SYSTEM_PROMPT_CACHE is not None:
        return _REACT_SYSTEM_PROMPT_CACHE

    try:
        from agent.prompt_loader import load_latest_prompt
        _REACT_SYSTEM_PROMPT_CACHE = load_latest_prompt("react_system")
        logger.debug("Loaded ReAct system prompt from file")
    except (ImportError, FileNotFoundError) as e:
        logger.warning(f"Could not load react_system prompt: {e}, using fallback")
        _REACT_SYSTEM_PROMPT_CACHE = """You are a Lean 4 theorem prover using ReAct (Reasoning + Acting).
Output in this exact format:
THOUGHT: <reasoning about what to try>
TACTIC: <Lean 4 tactic code>
CONFIDENCE: <0.0-1.0>

Use `try grind`, `try omega`, `try ring` for safety. Use `rw [h]` to rewrite with hypotheses."""

    return _REACT_SYSTEM_PROMPT_CACHE


# ==============================================================================
# Goal State Parser (Phase 4.0)
# ==============================================================================

import re


def _extract_hypotheses(goal_state: str, max_hypotheses: int = 50) -> list[HypothesisInfo]:
    """
    Extract hypothesis information from a Lean goal state.

    Parses lines of the form:
        hypothesis_name : type_expression

    Args:
        goal_state: Raw goal state string from LSP
        max_hypotheses: Maximum number to parse (for performance)

    Returns:
        List of HypothesisInfo dictionaries
    """
    hypotheses = []

    # Split at ⊢ to separate hypotheses from goal
    parts = goal_state.split("⊢")
    if len(parts) < 2:
        # Try alternate formats
        parts = goal_state.split("|-")

    hyp_section = parts[0] if parts else goal_state

    # Pattern to match hypothesis declarations
    # Matches: name : type (possibly multi-line with indentation)
    hyp_pattern = re.compile(
        r'^([a-zA-Z_][a-zA-Z0-9_\']*)\s*:\s*(.+?)(?=\n[a-zA-Z_]|\n⊢|\Z)',
        re.MULTILINE | re.DOTALL
    )

    # Also try simpler line-by-line parsing
    lines = hyp_section.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('--'):
            continue

        # Match "name : type"
        match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_\']*)\s*:\s*(.+)$', line)
        if match:
            name = match.group(1)
            type_str = match.group(2).strip()

            # Detect if this is an equation (contains = but not in ≤ or ≥)
            is_equation = bool(re.search(r'(?<![<>≤≥!])=(?![=])', type_str))

            # Detect if this is a bound (contains < or > or ≤ or ≥)
            is_bound = bool(re.search(r'[<>≤≥]', type_str))

            hypotheses.append(HypothesisInfo(
                name=name,
                type_str=type_str[:200],  # Truncate long types
                is_equation=is_equation,
                is_bound=is_bound,
            ))

            if len(hypotheses) >= max_hypotheses:
                break

    return hypotheses


def _classify_goal_type(goal_state: str) -> tuple[str, str]:
    """
    Classify the type of goal and extract the goal expression.

    Returns:
        (goal_type, goal_expr) where goal_type is one of:
        - "equality": Goal is a = b
        - "implication": Goal is A → B
        - "forall": Goal is ∀ x, P x
        - "exists": Goal is ∃ x, P x
        - "conjunction": Goal is A ∧ B
        - "disjunction": Goal is A ∨ B
        - "other": Unknown structure
    """
    # Extract goal expression after ⊢
    goal_expr = ""
    if "⊢" in goal_state:
        goal_expr = goal_state.split("⊢", 1)[1].strip()
    elif "|-" in goal_state:
        goal_expr = goal_state.split("|-", 1)[1].strip()
    else:
        goal_expr = goal_state.strip()

    # Clean up multiline goals
    goal_expr = ' '.join(goal_expr.split())

    # Classify based on top-level structure
    goal_clean = goal_expr.strip()

    # Check for forall (∀ or unicode variants)
    if goal_clean.startswith("∀") or goal_clean.startswith("forall"):
        return "forall", goal_expr

    # Check for exists (∃)
    if goal_clean.startswith("∃") or goal_clean.startswith("exists"):
        return "exists", goal_expr

    # Check for implication at top level (→ not inside parentheses)
    # Simple heuristic: if → appears and is not deeply nested
    if "→" in goal_clean or "->" in goal_clean:
        # Count nesting depth
        depth = 0
        for i, c in enumerate(goal_clean):
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
            elif c == "→" and depth == 0:
                return "implication", goal_expr

    # Check for equality at top level
    if re.search(r'(?<![<>≤≥!])=(?![=])', goal_clean):
        return "equality", goal_expr

    # Check for conjunction/disjunction
    if "∧" in goal_clean or "/\\" in goal_clean:
        return "conjunction", goal_expr
    if "∨" in goal_clean or "\\/" in goal_clean:
        return "disjunction", goal_expr

    return "other", goal_expr


def _find_rewrite_candidates(hypotheses: list[HypothesisInfo]) -> list[str]:
    """
    Find hypotheses that can be used with `rw [h]`.

    Rewrite candidates are hypotheses of the form h : a = b.
    """
    return [h["name"] for h in hypotheses if h["is_equation"]]


def _find_bound_hypotheses(hypotheses: list[HypothesisInfo]) -> list[str]:
    """
    Find hypotheses that represent bounds (h : x < N).
    """
    return [h["name"] for h in hypotheses if h["is_bound"]]


def _find_definitions_in_scope(goal_state: str) -> list[str]:
    """
    Extract definition names that might need unfolding.

    Looks for CamelCase identifiers that are likely definitions.
    """
    definitions = set()

    # Pattern for CamelCase or snake_case_with_caps identifiers
    # These are likely user-defined definitions
    pattern = re.compile(r'\b([A-Z][a-zA-Z0-9_]*(?:_[a-zA-Z0-9]+)*)\b')

    for match in pattern.finditer(goal_state):
        name = match.group(1)
        # Filter out common non-definition patterns
        if name not in {"True", "False", "Type", "Prop", "Sort", "Nat", "Int", "Bool"}:
            definitions.add(name)

    # Also look for specific patterns like "as_Nat", "as_Int"
    pattern2 = re.compile(r'\b([a-zA-Z_]+_as_[A-Za-z]+)\b')
    for match in pattern2.finditer(goal_state):
        definitions.add(match.group(1))

    return sorted(list(definitions))[:20]  # Limit to 20


def _summarize_goal(goal_type: str, goal_expr: str, hyp_count: int) -> str:
    """
    Create a one-line summary of the goal for prompts.
    """
    goal_short = goal_expr[:100] + "..." if len(goal_expr) > 100 else goal_expr

    if goal_type == "equality":
        return f"Prove equality with {hyp_count} hypotheses: {goal_short}"
    elif goal_type == "implication":
        return f"Prove implication with {hyp_count} hypotheses: {goal_short}"
    elif goal_type == "forall":
        return f"Prove universal statement with {hyp_count} hypotheses: {goal_short}"
    elif goal_type == "exists":
        return f"Prove existential with {hyp_count} hypotheses: {goal_short}"
    else:
        return f"Prove goal ({goal_type}) with {hyp_count} hypotheses: {goal_short}"


def parse_goal_state(goal_state: str) -> GoalAnalysis:
    """
    Parse a Lean goal state into structured analysis.

    This is the main entry point for goal parsing.

    Args:
        goal_state: Raw goal state string from LSP

    Returns:
        GoalAnalysis with extracted information
    """
    if not goal_state or not goal_state.strip():
        return GoalAnalysis(
            hypotheses=[],
            goal_type="other",
            goal_expr="",
            rewrite_candidates=[],
            bound_hypotheses=[],
            definitions_in_scope=[],
            goal_summary="No goal state available",
            hypothesis_count=0,
        )

    hypotheses = _extract_hypotheses(goal_state)
    goal_type, goal_expr = _classify_goal_type(goal_state)
    rewrite_candidates = _find_rewrite_candidates(hypotheses)
    bound_hypotheses = _find_bound_hypotheses(hypotheses)
    definitions = _find_definitions_in_scope(goal_state)
    summary = _summarize_goal(goal_type, goal_expr, len(hypotheses))

    return GoalAnalysis(
        hypotheses=hypotheses,
        goal_type=goal_type,
        goal_expr=goal_expr,
        rewrite_candidates=rewrite_candidates,
        bound_hypotheses=bound_hypotheses,
        definitions_in_scope=definitions,
        goal_summary=summary,
        hypothesis_count=len(hypotheses),
    )


def goal_parser_node(state: ProofState) -> dict[str, Any]:
    """
    Parse the current goal state into structured analysis.

    This node runs at the start of each proof attempt to extract
    hypothesis information, goal type, and rewrite candidates.

    Returns:
        Partial state update with goal_analysis
    """
    goal_state = state.get("goal_state", "")

    # Parse the goal state
    analysis = parse_goal_state(goal_state)

    logger.debug(
        f"Goal parsed: type={analysis['goal_type']}, "
        f"hypotheses={analysis['hypothesis_count']}, "
        f"rewrite_candidates={len(analysis['rewrite_candidates'])}"
    )

    # Store previous goal state for progress tracking
    previous_goal = state.get("goal_state", "")

    return {
        "goal_analysis": dict(analysis),  # Convert TypedDict to dict for JSON serialization
        "previous_goal_state": previous_goal,
    }


# ==============================================================================
# Iterative Tactic Loop (Phase 4.4)
# ==============================================================================


def _load_single_shot_prompt() -> str:
    """
    Load the single-shot tactic generation prompt.

    Uses the prompt loader to get the latest version from
    prompts/verifier/single_shot_tactics_v*.md.

    Falls back to a simple inline prompt if file not found.
    """
    try:
        from agent.prompt_loader import load_latest_prompt
        return load_latest_prompt("single_shot_tactics")
    except (ImportError, FileNotFoundError) as e:
        logger.warning(f"Could not load single_shot_tactics prompt: {e}, using fallback")
        # Minimal fallback - just asks for plain tactics
        return """You are a Lean 4 proof assistant. Generate tactics to close the given proof goal.

Output ONLY tactics, one per line. No explanations, no markdown, no numbering.

Example output format:
simp [add_comm]
ring
omega
exact h
rw [h_eq]

Use `try grind`, `try omega`, `try ring` for safety (prevents crashes).
Use hypothesis names explicitly (rw [h]) rather than wildcards [*]."""


async def _single_shot_tactic_generation(
    goal_state: str,
    model: str,
    temperature: float,
    context: Optional[dict[str, Any]],
    llm_client: Any,
) -> tuple[list[str], bool, str]:
    """
    Fallback single-shot tactic generation when MCP is unavailable.

    Instead of the iterative loop (which requires MCP multi_attempt), this
    generates tactics in a single LLM call. Less effective but works without
    MCP connection.

    Returns:
        Tuple of (tactics, success, final_goal):
        - tactics: List of generated tactics (not verified)
        - success: Always False (can't verify without MCP)
        - final_goal: Original goal state (unchanged)
    """
    logger.info("Using single-shot tactic generation (MCP unavailable)")

    prompt = _build_single_shot_prompt(goal_state, context)
    # Use dedicated single-shot prompt (NOT ReAct format)
    system = _load_single_shot_prompt()

    logger.debug(f"Single-shot prompt (first 300 chars):\n{prompt[:300]}...")

    # Log LLM request
    context_summary = "single_shot"
    if context and "theorem_name" in context:
        context_summary += f", theorem={context['theorem_name']}"
    log_llm_request(goal_state, model, context_summary)

    try:
        # LLMClient.generate() returns a string directly, not a dict
        response = await llm_client.generate(
            user_prompt=prompt,
            model=model,
            temperature=temperature,
            system_prompt=system,
        )

        # Debug: Log raw LLM response to diagnose parsing issues
        logger.info(f"Single-shot raw response (first 500 chars):\n{response[:500]}...")

        tactics = _parse_tactic_list(response)
        log_llm_response(tactics, model)

        if tactics:
            logger.info(f"Single-shot generated {len(tactics)} tactics: {tactics[:5]}")
            # Note: We return success=False because we can't verify without MCP
            # The caller should use verifier_service.verify_proof_on_copy() to verify
            return tactics, False, goal_state
        else:
            logger.warning(f"Single-shot generation produced no tactics from response:\n{response[:300]}...")
            return [], False, goal_state

    except Exception as e:
        logger.error(f"Single-shot tactic generation failed: {e}")
        return [], False, goal_state


def _build_single_shot_prompt(goal_state: str, context: Optional[dict[str, Any]]) -> str:
    """Build prompt for single-shot tactic generation."""
    parts = [
        "Generate Lean 4 tactics to close this goal.",
        "",
        "## Goal State",
        "```",
        goal_state,
        "```",
    ]

    if context:
        if context.get("theorem_name"):
            parts.extend(["", f"**Theorem:** {context['theorem_name']}"])
        if context.get("hypothesis_names"):
            parts.extend(["", f"**Hypotheses available:** {', '.join(context['hypothesis_names'])}"])
        if context.get("rewrite_candidates"):
            parts.extend(["", f"**Rewrite candidates (h : a = b):** {', '.join(context['rewrite_candidates'])}"])
        if context.get("definitions_in_scope"):
            parts.extend(["", f"**Definitions to unfold:** {', '.join(context['definitions_in_scope'][:5])}"])
        if context.get("rag_tactics"):
            parts.extend(["", "**Similar proofs from codebase:**"])
            for tactic in context["rag_tactics"][:3]:
                parts.append(f"  - {tactic[:100]}...")

    parts.extend([
        "",
        "## Instructions",
        "Output one or more tactics, one per line.",
        "Start with simpler tactics (simp, rfl, ring) before complex ones.",
        "Use hypothesis names explicitly (rw [h]) rather than wildcards [*].",
        "Chain with semicolons only if necessary.",
        "",
        "## Output (tactics only, no explanation)",
    ])

    return "\n".join(parts)


async def iterative_tactic_loop(
    goal_state: str,
    file_path: str,
    line: int,
    model: str,
    temperature: float = 0.2,
    max_steps: int = 15,
    consecutive_failure_threshold: int = 3,
    context: Optional[dict[str, Any]] = None,
    mcp_client: Optional[Any] = None,
    attempt_number: int = 1,
    sorry_index: int = 1,
) -> tuple[list[str], bool, str, list[AttemptRecord]]:
    """
    Iteratively apply tactics using MCP edit_file_with_capture.

    Two-level attempt model:
    - OUTER attempts: controlled by --max-attempts (default 5),
      user-visible, VP files numbered by this
    - INNER steps: max_steps parameter (default 15),
      implementation detail, tactic candidates per outer attempt

    Args:
        goal_state: Current goal state from LSP
        file_path: Absolute path to the Lean file
        line: Line number where tactic should be applied
        model: LLM model to use for tactic generation
        temperature: Temperature for LLM generation
        max_steps: Maximum number of tactics to try
        consecutive_failure_threshold: Re-analyze after N failures
        context: Optional additional context
        mcp_client: Optional connected MCP client (LeanMCPClient)
        attempt_number: Outer attempt number (VP file naming)
        sorry_index: Sorry index (console output)

    Returns:
        (tactics_applied, success, final_goal_state, attempt_records)
    """
    from agent.llm_client import LLMClient
    from verifier.error_normalizer import ErrorMessageNormalizer
    from pathlib import Path as _Path

    # Console output (graceful degradation)
    try:
        from cli.console_output import (
            print_attempt_start,
            print_attempt_trying,
            print_attempt_success,
            print_attempt_failure,
        )
        _has_console = True
    except ImportError:
        _has_console = False

    tactics_applied: list[str] = []
    attempt_records: list[AttemptRecord] = []
    current_goal = goal_state
    consecutive_failures = 0
    # Track total consecutive outer failures across the whole loop
    # (never reset by re-analysis threshold, only by actual success).
    # Fed to _generate_tactic_candidates for strategy shift detection.
    total_consecutive_failures = 0
    llm_client = LLMClient()
    normalizer = ErrorMessageNormalizer()

    # Check if MCP is available for iterative testing
    if mcp_client is None:
        logger.warning(
            "MCP client not available - "
            "falling back to single-shot generation"
        )
        tactics, success, final_goal = (
            await _single_shot_tactic_generation(
                goal_state=goal_state,
                model=model,
                temperature=temperature,
                context=context,
                llm_client=llm_client,
            )
        )
        return tactics, success, final_goal, []

    logger.info(
        f"Starting iterative tactic loop: "
        f"max_steps={max_steps}, attempt={attempt_number}"
    )

    # Print attempt start
    if _has_console:
        print_attempt_start(
            sorry_index, attempt_number,
            max_steps, current_goal,
        )

    step_start_time = time.time()

    for step in range(max_steps):
        # Check if goal is already closed
        if _is_goal_closed(current_goal):
            elapsed = time.time() - step_start_time
            logger.info(f"Goal closed after {step} steps")
            if _has_console:
                print_attempt_success(
                    sorry_index, attempt_number,
                    max_steps, elapsed,
                )
            return (
                tactics_applied, True,
                current_goal, attempt_records,
            )

        # Check if we need full re-analysis
        needs_reanalysis = (
            consecutive_failures >= consecutive_failure_threshold
        )
        if needs_reanalysis:
            logger.info(
                f"Triggering re-analysis after "
                f"{consecutive_failures} failures"
            )
            consecutive_failures = 0

        # Generate candidates using structured sliding window
        candidates = await _generate_tactic_candidates(
            goal_state=current_goal,
            model=model,
            temperature=temperature,
            context=context,
            deep_analysis=needs_reanalysis,
            attempt_history=attempt_records,
            consecutive_failures=total_consecutive_failures,
        )

        if not candidates:
            logger.warning(
                f"No tactic candidates generated at step {step}"
            )
            consecutive_failures += 1
            total_consecutive_failures += 1
            continue

        # Try candidates using edit_file_with_capture
        from verifier.mcp_client import MCPWorkerCrashError

        success_found = False
        last_modified_content: Optional[str] = None
        last_error_msg: Optional[str] = None

        for candidate in candidates:
            # Print what we're trying
            if _has_console:
                print_attempt_trying(
                    sorry_index, attempt_number, candidate,
                )

            candidate_start = time.time()
            try:
                (
                    made_progress, new_goal_str,
                    error_msg, modified_content,
                ) = await mcp_client.edit_file_with_capture(
                    file_path=file_path,
                    line=line,
                    tactic=candidate,
                )
            except MCPWorkerCrashError:
                raise  # Let caller handle crash recovery
            except Exception as e:
                logger.warning(
                    f"edit_file_with_capture failed "
                    f"for '{candidate[:50]}': {e}"
                )
                continue

            # Track last modified content for VP file
            if modified_content is not None:
                last_modified_content = modified_content

            candidate_elapsed = time.time() - candidate_start

            if made_progress and new_goal_str:
                tactics_applied.append(candidate)
                current_goal = new_goal_str
                consecutive_failures = 0
                total_consecutive_failures = 0
                success_found = True
                logger.info(
                    f"Step {step}: "
                    f"'{candidate[:50]}...' succeeded"
                )

                if _is_goal_closed(new_goal_str):
                    elapsed = time.time() - step_start_time
                    logger.info(
                        f"Goal closed by tactic: "
                        f"{candidate[:50]}..."
                    )
                    if _has_console:
                        print_attempt_success(
                            sorry_index, attempt_number,
                            max_steps, elapsed,
                        )

                    # Write VP file with successful content
                    if last_modified_content is not None:
                        original = _Path(file_path)
                        vp_name = (
                            f"{original.stem}"
                            f"_VP{attempt_number}"
                            f"{original.suffix}"
                        )
                        vp_path = original.parent / vp_name
                        vp_path.write_text(last_modified_content)
                        logger.debug(
                            f"Created VP file: {vp_path}"
                        )

                    return (
                        tactics_applied, True,
                        current_goal, attempt_records,
                    )
                break
            elif error_msg:
                last_error_msg = error_msg
                logger.debug(
                    f"Step {step}: "
                    f"'{candidate[:50]}' -- {error_msg}"
                )

                # Normalize error and create AttemptRecord
                normalized = normalizer.normalize(error_msg)

                # RAG lookup for unknown identifiers
                rag_suggestions: list[str] = []
                if normalized.error_type == "unknown_identifier":
                    id_match = re.search(
                        r"unknown identifier '([^']+)'",
                        error_msg,
                        re.IGNORECASE,
                    )
                    if id_match:
                        rag_suggestions = (
                            await _suggest_identifier_corrections(
                                id_match.group(1)
                            )
                        )

                record = AttemptRecord(
                    number=len(attempt_records) + 1,
                    snippet=candidate,
                    normalized_error=normalized.normalized,
                    error_type=normalized.error_type,
                    goal_state_after=new_goal_str or "",
                    suggestion=normalized.suggestion,
                    rag_suggestions=rag_suggestions,
                    elapsed_seconds=candidate_elapsed,
                )
                attempt_records.append(record)

        # After trying all candidates, write VP file
        if last_modified_content is not None:
            original = _Path(file_path)
            vp_name = (
                f"{original.stem}"
                f"_VP{attempt_number}"
                f"{original.suffix}"
            )
            vp_path = original.parent / vp_name
            vp_path.write_text(last_modified_content)
            logger.debug(f"Created VP file: {vp_path}")

        if not success_found:
            consecutive_failures += 1
            total_consecutive_failures += 1
            logger.debug(
                f"Step {step}: No tactic succeeded, "
                f"failures={consecutive_failures}"
            )
            # Print failure for this step
            if _has_console and last_error_msg:
                error_summary = last_error_msg[:100]
                print_attempt_failure(
                    sorry_index, attempt_number,
                    max_steps, error_summary,
                )

    # Max steps reached
    logger.info(
        f"Max steps ({max_steps}) reached, "
        f"applied {len(tactics_applied)} tactics"
    )
    if _has_console:
        print_attempt_failure(
            sorry_index, attempt_number,
            max_steps, "max steps reached",
        )
    return (
        tactics_applied, False,
        current_goal, attempt_records,
    )


def _is_goal_closed(goal_state: str) -> bool:
    """Check if the goal state indicates no remaining goals."""
    if not goal_state:
        return False
    goal_lower = goal_state.lower().strip()
    return (
        "no goals" in goal_lower
        or goal_lower == "goals accomplished"
        or goal_lower == ""
    )


def _extract_goal_from_result(result: dict[str, Any]) -> Optional[str]:
    """Extract goal state from multi_attempt result."""
    if not result:
        return None
    # multi_attempt returns goal state for each tactic
    # Format may vary - try common keys
    for key in ["goal_state", "goals_after", "goal", "result"]:
        if key in result:
            return str(result[key])
    return None


def _made_progress(old_goal: str, new_goal: str) -> bool:
    """
    Check if the goal state changed (tactic made progress).

    Progress is defined as:
    - Goal state is different from before
    - Goal state is not an error message
    - Goal is closed (no goals) or changed meaningfully
    """
    if not new_goal:
        return False

    # Check for error indicators
    error_indicators = ["error", "unknown identifier", "type mismatch", "failed"]
    new_lower = new_goal.lower()
    if any(err in new_lower for err in error_indicators):
        return False

    # Goal closed = progress
    if _is_goal_closed(new_goal):
        return True

    # Different goal state = progress (but not if it's just whitespace changes)
    old_normalized = old_goal.strip()
    new_normalized = new_goal.strip()
    return old_normalized != new_normalized


async def _generate_tactic_candidates(
    goal_state: str,
    model: str,
    temperature: float,
    context: Optional[dict[str, Any]],
    deep_analysis: bool,
    attempt_history: list[AttemptRecord],
    consecutive_failures: int = 0,
) -> list[str]:
    """
    Generate candidate tactics for the current goal state.

    Uses a structured sliding window of the last 3 attempts
    (snippet + normalized error + goal state + suggestion)
    so the LLM can learn from previous failures.

    When consecutive_failures >= STRATEGY_SHIFT_THRESHOLD, injects a
    strategy shift prompt to force the LLM to try fundamentally
    different approaches.

    Args:
        goal_state: Current goal state
        model: LLM model to use
        temperature: Temperature for generation
        context: Additional context (theorem_name, etc.)
        deep_analysis: If True, do full re-analysis
        attempt_history: Structured records of previous attempts
        consecutive_failures: Count of consecutive outer failures
            (no candidate succeeded). Triggers strategy shift when
            >= STRATEGY_SHIFT_THRESHOLD.

    Returns:
        List of 3-5 candidate tactics to try
    """
    from agent.llm_client import LLMClient

    llm_client = LLMClient()

    # Load iterative system prompt (atomic snippet guidance)
    system_prompt = _load_iterative_system_prompt()

    # Build user prompt for snippet generation
    prompt_lines = []

    # Strategy shift: inject "try something different" prompt before goal
    if consecutive_failures >= STRATEGY_SHIFT_THRESHOLD:
        failed_summary = "\n".join(
            f"- `{a['snippet'][:100]}` -> "
            f"{a['error_type']}: {a['normalized_error'][:80]}"
            for a in attempt_history[-consecutive_failures:]
        )
        prompt_lines.append(
            STRATEGY_SHIFT_PROMPT.format(
                n=consecutive_failures,
                failed_summary=failed_summary,
            )
        )
        prompt_lines.append("")
        logger.info(
            f"Strategy shift triggered after "
            f"{consecutive_failures} consecutive failures"
        )

    prompt_lines.extend([
        "Generate 1-3 atomic proof snippets for this goal state.",
        "",
        "## Current Goal State",
        "```lean",
        goal_state,
        "```",
        "",
    ])

    # Add context if available
    if context:
        if "theorem_name" in context:
            prompt_lines.append(
                f"**Theorem:** {context['theorem_name']}"
            )
        if "hypotheses" in context:
            prompt_lines.append(
                f"**Key Hypotheses:** {context['hypotheses']}"
            )
        prompt_lines.append("")

    # Structured sliding window of recent attempts (last 3)
    if attempt_history:
        prompt_lines.extend([
            "## Recent Attempts (last 3 -- do NOT repeat these)",
        ])
        for attempt in attempt_history[-3:]:
            prompt_lines.extend([
                f"### Attempt {attempt['number']}",
                f"**Tried:** `{attempt['snippet'][:200]}`",
                f"**Error:** {attempt['normalized_error']}",
                f"**Suggestion:** {attempt['suggestion']}",
            ])
            if attempt['goal_state_after']:
                goal_preview = attempt['goal_state_after'][:300]
                prompt_lines.append(
                    f"**Goal after:** ```{goal_preview}```"
                )
            if attempt['rag_suggestions']:
                alts = ', '.join(
                    f'`{s}`'
                    for s in attempt['rag_suggestions'][:3]
                )
                prompt_lines.append(
                    f"**Did you mean:** {alts}"
                )
            prompt_lines.append("")

    # Add guidance based on analysis depth
    if deep_analysis:
        prompt_lines.extend([
            "## Re-analysis Mode",
            "Previous approaches failed repeatedly. "
            "Try something fundamentally different:",
            "- If automation failed, try manual rewriting",
            "- If rewriting failed, try case analysis "
            "or intermediate lemmas",
            "- If direct proof failed, try a `calc` block "
            "or term-mode proof",
            "",
        ])

    # Output format reminder
    prompt_lines.extend([
        "## Output",
        "Separate each snippet with `---`. No markdown fences.",
    ])

    prompt = "\n".join(prompt_lines)

    # Generate with slightly higher temperature for diversity
    gen_temp = min(0.7, temperature + 0.1)

    # Increase temperature further during strategy shift to encourage novelty
    if consecutive_failures >= STRATEGY_SHIFT_THRESHOLD:
        gen_temp = min(0.9, gen_temp + 0.2)

    # Log LLM request
    context_summary = f"deep_analysis={deep_analysis}"
    if context and "theorem_name" in context:
        context_summary += f", theorem={context['theorem_name']}"
    log_llm_request(goal_state, model, context_summary)

    try:
        response = await llm_client.generate(
            prompt,
            model=model,
            temperature=gen_temp,
            system_prompt=system_prompt,
        )

        # Parse snippets from response (handles --- separators)
        tactics = _parse_tactic_list(response)
        log_llm_response(tactics[:5], model)
        return tactics[:5]  # Max 5

    except Exception as e:
        logger.warning(f"Tactic generation failed: {e}")
        # Return some default tactics
        return ["simp", "ring", "omega"]


async def _suggest_identifier_corrections(identifier: str) -> list[str]:
    """Query RAG when Lean reports an unknown identifier.

    Strategy:
    1. Try exact name lookup via LeanTypeIndex.query_by_name()
    2. If no results, try fuzzy search via LeanTypeIndex.query()
    3. Limit to top 3 suggestions
    4. Gracefully degrade: return empty list if index unavailable
    """
    import asyncio

    try:
        from rag.lean.type_index import LeanTypeIndex

        index = LeanTypeIndex()  # Uses default db_path from env

        # Level 1: exact name (catches namespace issues like
        # add_comm -> Nat.add_comm)
        results = await asyncio.wait_for(
            index.query_by_name(identifier, limit=3),
            timeout=2.0,
        )
        if results:
            return [r.full_name for r in results]

        # Level 2: fuzzy search (catches typos)
        results = await asyncio.wait_for(
            index.query(identifier, limit=3),
            timeout=2.0,
        )
        if results:
            return [r.full_name for r in results]

        return []
    except Exception as e:
        logger.debug(
            f"RAG lookup for '{identifier}' failed "
            f"(graceful degradation): {e}"
        )
        return []


def _parse_tactic_list(response: str) -> list[str]:
    """Parse list of tactics/snippets from LLM response.

    Handles multiple formats:
    - --- separated blocks (atomic snippets, preferred)
    - Bulleted lists (-, *)
    - Numbered lists (1., 2.)
    - Code blocks (```lean ... ```)
    - Plain lines that look like Lean tactics
    """
    tactics = []

    # Try --- separated blocks first (atomic snippet format)
    if "---" in response:
        blocks = response.split("---")
        for block in blocks:
            # Strip each block and skip empty ones
            snippet = block.strip()
            # Skip blocks that are just markdown headers or explanatory text
            if not snippet or snippet.startswith("#") or snippet.startswith("```"):
                continue
            # Strip code fences if wrapping the whole block
            snippet = re.sub(r"^```(?:lean)?\s*\n?", "", snippet)
            snippet = re.sub(r"\n?```\s*$", "", snippet)
            snippet = snippet.strip()
            if snippet and not snippet.startswith("You ") and not snippet.startswith("The "):
                tactics.append(snippet)
        if tactics:
            return tactics

    # Extract code blocks
    code_block_pattern = r"```(?:lean)?\s*\n(.*?)```"
    code_blocks = re.findall(code_block_pattern, response, re.DOTALL)
    for block in code_blocks:
        for line in block.split("\n"):
            line = line.strip()
            if line and not line.startswith("--"):  # Skip comments
                tactics.append(line)

    # If we found tactics in code blocks, return them
    if tactics:
        return tactics

    # Otherwise, try other formats
    for line in response.split("\n"):
        line = line.strip()
        # Skip empty lines and markdown headers
        if not line or line.startswith("#"):
            continue

        # Look for lines starting with - or *
        if line.startswith("-") or line.startswith("*"):
            tactic = line[1:].strip()
            tactic = tactic.strip("`")
            if tactic and not tactic.startswith("This") and not tactic.startswith("The"):
                tactics.append(tactic)
        # Accept numbered lists
        elif line and line[0].isdigit() and "." in line[:3]:
            tactic = line.split(".", 1)[1].strip()
            tactic = tactic.strip("`")
            if tactic and not tactic.startswith("This") and not tactic.startswith("The"):
                tactics.append(tactic)
        # Accept inline code that looks like a tactic
        elif line.startswith("`") and line.endswith("`"):
            tactic = line.strip("`")
            if tactic:
                tactics.append(tactic)
        # Accept lines that look like Lean tactics (start with common tactic keywords)
        elif any(line.startswith(kw) for kw in [
            "simp", "rfl", "ring", "omega", "exact", "apply", "intro", "constructor",
            "rw", "have", "let", "unfold", "decide", "norm_num", "try", "sorry",
            "cases", "rcases", "calc", "conv", "progress",
        ]):
            tactics.append(line)

    return tactics


# ==============================================================================
# Reasoning Node
# ==============================================================================

async def reasoning_node(state: ProofState) -> dict[str, Any]:
    """
    Generate reasoning about the current proof state and plan next action.

    This node:
    1. Analyzes current goal state and error history
    2. Generates a thought about what to try next
    3. Plans the next tactic to attempt

    Uses LLM to generate ReAct-style reasoning.

    Returns:
        Partial state update with new thought and incremented step
    """
    from agent.llm_client import LLMClient
    from agent.prompts import extract_proof_from_response

    step = state["step"] + 1
    attempt = state["attempt_count"]

    # Build reasoning prompt (user message)
    prompt = _build_reasoning_prompt(state)

    # Load ReAct system prompt
    system_prompt = _load_react_system_prompt()

    # Generate thought and action plan
    try:
        client = LLMClient()

        # Use slightly lower temperature for reasoning
        temperature = max(0.1, state["base_temperature"] - 0.1)

        response = await client.generate(
            prompt,
            model=state["model_used"],
            system_prompt=system_prompt,
            temperature=temperature,
        )

        # Parse thought and action from response
        thought_content, tactic_plan, confidence = _parse_reasoning_response(response)

    except Exception as e:
        logger.warning(f"Reasoning failed: {e}")
        thought_content = f"Reasoning failed: {e}. Will retry with previous approach."
        tactic_plan = state["current_proof"]
        confidence = 0.3

    # Create thought record
    thought = ThoughtRecord(
        step=step,
        content=thought_content,
        tactic_plan=tactic_plan,
        confidence=confidence,
    )

    logger.debug(f"Step {step} thought: {thought_content[:100]}...")

    return {
        "step": step,
        "thoughts": [thought],
        "current_proof": tactic_plan if tactic_plan else state["current_proof"],
    }


def _build_reasoning_prompt(state: ProofState) -> str:
    """Build prompt for reasoning about next action."""
    sorry = dict_to_sorry(state["sorry_location"])

    lines = [
        "# ReAct Proof Reasoning",
        "",
        f"## Task",
        f"Prove the theorem `{sorry.theorem_name}` at line {sorry.line}.",
        "",
        f"## Current State",
        f"- Attempt: {state['attempt_count']}/{state['max_attempts']}",
        f"- Step: {state['step']}",
        "",
    ]

    # Add goal state if available
    if state["goal_state"]:
        lines.extend([
            "## Goal State",
            "```lean",
            state["goal_state"],
            "```",
            "",
        ])

    # Add structured goal analysis if available (Phase 4.0)
    goal_analysis = state.get("goal_analysis")
    if goal_analysis:
        lines.extend([
            "## Goal Analysis",
            f"- **Goal Type:** {goal_analysis.get('goal_type', 'unknown')}",
            f"- **Summary:** {goal_analysis.get('goal_summary', 'N/A')}",
            "",
        ])

        # Add rewrite candidates (hypotheses of form h : a = b)
        rewrite_candidates = goal_analysis.get("rewrite_candidates", [])
        if rewrite_candidates:
            lines.extend([
                "### Rewrite Candidates (h : a = b)",
                "These hypotheses can be used with `rw [h]`:",
            ])
            for h in rewrite_candidates[:10]:
                lines.append(f"- `{h}`")
            lines.append("")

        # Add bound hypotheses (h : x < N)
        bound_hyps = goal_analysis.get("bound_hypotheses", [])
        if bound_hyps:
            lines.extend([
                "### Bound Hypotheses (h : x < N)",
                "Useful for `omega`, `linarith`, or bounds reasoning:",
            ])
            for h in bound_hyps[:10]:
                lines.append(f"- `{h}`")
            lines.append("")

        # Add definitions that might need unfolding
        definitions = goal_analysis.get("definitions_in_scope", [])
        if definitions:
            lines.extend([
                "### Definitions in Scope",
                "Consider unfolding with `simp only [def]` or `unfold def`:",
            ])
            for d in definitions[:10]:
                lines.append(f"- `{d}`")
            lines.append("")

        # Add hypothesis names for reference
        hypotheses = goal_analysis.get("hypotheses", [])
        if hypotheses:
            lines.extend([
                "### Available Hypotheses",
                "All hypotheses by name (use specific names instead of `[*]`):",
            ])
            for hyp in hypotheses[:15]:
                name = hyp.get("name", "")
                type_str = hyp.get("type_str", "")[:60]
                lines.append(f"- `{name}` : {type_str}")
            if len(hypotheses) > 15:
                lines.append(f"  ... and {len(hypotheses) - 15} more")
            lines.append("")

    # Add error history (last 3)
    if state["error_history"]:
        lines.extend([
            "## Recent Errors",
        ])
        for err in state["error_history"][-3:]:
            lines.append(f"- {err[:150]}")
        lines.append("")

    # Add previous attempts
    if state["proof_history"]:
        lines.extend([
            "## Previous Attempts",
        ])
        for i, proof in enumerate(state["proof_history"][-3:], 1):
            lines.append(f"{i}. `{proof[:80]}`")
        lines.append("")

    # Add RAG context if available
    if state["rag_results"]:
        lines.extend([
            "## Relevant Lemmas (from RAG)",
        ])
        for r in state["rag_results"][:5]:
            if isinstance(r, dict):
                name = r.get("name", "")
                sig = r.get("signature", "")[:80]
                lines.append(f"- `{name}`: {sig}")
        lines.append("")

    # Add import file context if available
    if state.get("import_context"):
        lines.extend([
            state["import_context"],  # Already formatted as markdown
            "",
        ])

    # Instructions - enhanced with goal analysis guidance
    lines.extend([
        "## Instructions",
        "",
        "Think step-by-step about what tactic to try next.",
        "Consider:",
        "1. What went wrong in previous attempts?",
        "2. **Use specific hypothesis names** (e.g., `rw [h_eq]`) instead of wildcards (`rw [*]`)",
        "3. What automation tactics might help? (grind, simp, omega, scalar_tac)",
        "4. Do we need to unfold definitions listed above?",
        "5. Should we use progress* for Aeneas code?",
        "",
    ])

    # Add goal-type specific hints
    if goal_analysis:
        goal_type = goal_analysis.get("goal_type", "other")
        if goal_type == "equality":
            lines.extend([
                "**Hint (equality goal):** Try `rfl`, `ring`, `simp`, or rewrite with hypotheses.",
                "",
            ])
        elif goal_type == "implication":
            lines.extend([
                "**Hint (implication goal):** Consider `intro h` to introduce the assumption.",
                "",
            ])
        elif goal_type == "forall":
            lines.extend([
                "**Hint (universal goal):** Use `intro x` to introduce the variable.",
                "",
            ])
        elif goal_type == "exists":
            lines.extend([
                "**Hint (existential goal):** Use `use <witness>` to provide the witness.",
                "",
            ])

    lines.extend([
        "## Output Format",
        "",
        "THOUGHT: <your reasoning about what to try>",
        "TACTIC: <the exact tactic code to try>",
        "CONFIDENCE: <0.0-1.0 how confident you are>",
    ])

    return "\n".join(lines)


def _parse_reasoning_response(response: str) -> tuple[str, str, float]:
    """
    Parse reasoning response into thought, tactic, and confidence.

    Returns:
        (thought_content, tactic_plan, confidence)
    """
    thought = ""
    tactic = ""
    confidence = 0.5

    lines = response.strip().split("\n")
    current_section = None

    for line in lines:
        line_upper = line.upper().strip()
        if line_upper.startswith("THOUGHT:"):
            thought = line.split(":", 1)[1].strip()
            current_section = "thought"
        elif line_upper.startswith("TACTIC:"):
            tactic = line.split(":", 1)[1].strip()
            current_section = "tactic"
        elif line_upper.startswith("CONFIDENCE:"):
            try:
                conf_str = line.split(":", 1)[1].strip()
                confidence = float(conf_str)
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, IndexError):
                pass
            current_section = None
        elif current_section == "thought" and line.strip():
            thought += " " + line.strip()
        elif current_section == "tactic" and line.strip():
            tactic += "\n" + line.strip()

    # Clean up tactic (remove markdown fences if present)
    tactic = tactic.strip()
    if tactic.startswith("```"):
        tactic = tactic.split("```")[1] if "```" in tactic[3:] else tactic[3:]
        if tactic.startswith("lean"):
            tactic = tactic[4:]
        tactic = tactic.strip()

    return thought.strip(), tactic.strip(), confidence


# ==============================================================================
# Execution Node
# ==============================================================================

async def execution_node(
    state: ProofState,
    verifier_service: Optional["VerifierService"] = None,
) -> dict[str, Any]:
    """
    Execute the planned tactic via LSP verification.

    This node:
    1. Takes the current_proof from state
    2. Creates a verification copy of the file
    3. Runs LSP verification
    4. Returns action record with execution details

    Args:
        state: Current proof state
        verifier_service: VerifierService for LSP verification (optional)

    Returns:
        Partial state update with action record
    """
    sorry = dict_to_sorry(state["sorry_location"])
    proof_code = state["current_proof"]
    step = state["step"]

    # Create action record
    action = ActionRecord(
        step=step,
        action_type="apply_tactic",
        content=proof_code,
        model_used=state["model_used"],
        temperature=state["base_temperature"],
    )

    logger.debug(f"Step {step} executing: {proof_code[:80]}...")

    # Store execution metadata for observation node
    # (actual verification happens in observation_node for cleaner separation)

    return {
        "actions": [action],
        "proof_history": [proof_code],
    }


# ==============================================================================
# Observation Node
# ==============================================================================

async def observation_node(
    state: ProofState,
    verifier_service: Optional["VerifierService"] = None,
) -> dict[str, Any]:
    """
    Get observation from executing the last action.

    This node:
    1. Runs LSP verification on the current proof
    2. Parses the result (success/errors/goals)
    3. Creates observation record
    4. Updates status if proof succeeded

    Args:
        state: Current proof state
        verifier_service: VerifierService for LSP verification

    Returns:
        Partial state update with observation and status
    """
    sorry = dict_to_sorry(state["sorry_location"])
    proof_code = state["current_proof"]
    step = state["step"]
    attempt = state["attempt_count"]

    # Run verification
    success = False
    errors: list[str] = []
    goal_state = ""
    error_type = None

    if verifier_service:
        try:
            success, errors, copy_path = await verifier_service.verify_proof_on_copy(
                sorry=sorry,
                proof_code=proof_code,
                attempt=attempt,
                model_used=state["model_used"],
                temperature=state["base_temperature"],
            )

            # Get goal state if failed
            if not success and hasattr(verifier_service, 'get_goal_state'):
                goal_state = await verifier_service.get_goal_state(
                    sorry.file_path, sorry.line
                )

            # Classify error type
            if errors:
                error_type = _classify_error(errors[0])

        except Exception as e:
            logger.warning(f"Verification error: {e}")
            errors = [str(e)]
            error_type = "verification_error"
    else:
        # No verifier - simulate for testing
        errors = ["No verifier service available"]
        error_type = "no_verifier"

    # Create observation
    observation = ObservationRecord(
        step=step,
        success=success,
        content=errors[0] if errors else "Proof verified successfully",
        error_type=error_type,
        goals_remaining=None,  # TODO: parse from goal state
    )

    logger.debug(f"Step {step} observation: success={success}, errors={len(errors)}")

    # Prepare state updates
    updates: dict[str, Any] = {
        "observations": [observation],
        "goal_state": goal_state if goal_state else state["goal_state"],
    }

    # Update status on success
    if success:
        updates["status"] = ProofStatus.SUCCESS.value
    elif errors:
        updates["error_history"] = errors[:1]  # Add first error to history

    return updates


def _classify_error(error: str) -> str:
    """Classify error type from error message."""
    error_lower = error.lower()

    if "type mismatch" in error_lower:
        return "type_mismatch"
    elif "unknown identifier" in error_lower or "unknown constant" in error_lower:
        return "unknown_identifier"
    elif "unsolved goals" in error_lower:
        return "unsolved_goals"
    elif "expected" in error_lower and "got" in error_lower:
        return "type_mismatch"
    elif "syntax" in error_lower or "unexpected" in error_lower:
        return "syntax_error"
    elif "timeout" in error_lower:
        return "timeout"
    else:
        return "unknown"


# ==============================================================================
# Recovery Node (OpenManus Error Recovery)
# ==============================================================================

async def recovery_node(state: ProofState) -> dict[str, Any]:
    """
    OpenManus recovery decision node.

    Analyzes the latest error, determines recovery strategy,
    and modifies the tactic for the next attempt.

    Called when om_router_node routes to "recover" instead of "continue".

    This node:
    1. Gets latest error from observations
    2. Classifies error severity and selects recovery strategy
    3. Modifies tactic using TacticModifier
    4. Returns updated state with modified tactic and recovery record

    Returns:
        Partial state update with modified tactic and recovery metadata
    """
    controller = ErrorRecoveryController()
    modifier = TacticModifier()

    step = state["step"]
    current_tactic = state["current_proof"]
    recovery_stage = state["recovery_stage"]

    # Get latest error from observations
    latest_obs = state["observations"][-1] if state["observations"] else None

    if not latest_obs or latest_obs.get("success"):
        # No error to recover from - shouldn't happen but handle gracefully
        logger.debug("recovery_node called with no error - resetting recovery stage")
        return {
            "recovery_stage": 0,
        }

    error_type = latest_obs.get("error_type", "unknown")
    error_content = latest_obs.get("content", "")

    # Build recovery context from state
    context = RecoveryContext(
        tried_tactics=list(state["tried_tactics"]),
        definitions_to_unfold=state.get("definitions_to_unfold", []),
        successful_tactics=state.get("successful_tactics", []),
        error_content=error_content,
        goal_state=state.get("goal_state", ""),
        attempt_count=state["attempt_count"],
    )

    # Classify error and get strategy
    severity, primary_strategy = controller.classify_error(error_type)

    # Get actual strategy based on recovery stage
    strategy = controller.get_recovery_strategy(
        error_type=error_type,
        recovery_stage=recovery_stage,
        context=context,
    )

    # Check for fatal errors or abort
    if strategy == RecoveryStrategy.ABORT:
        logger.info(f"Recovery aborted for {error_type} (fatal)")
        return {
            "status": ProofStatus.FAILED.value,
            "current_error_type": error_type,
            "current_severity": severity.value,
            "active_strategy": strategy.value,
        }

    # Check for escalation (backtrack or human review)
    if strategy == RecoveryStrategy.BACKTRACK:
        logger.info(f"Recovery escalated to backtrack for {error_type}")
        # For now, just continue with a safe default tactic
        # Phase 4 will implement actual checkpoint restoration
        modified_tactic = "try grind <;> try simp"
    elif strategy == RecoveryStrategy.ESCALATE:
        logger.info(f"Recovery escalated for {error_type} - human review needed")
        modified_tactic = current_tactic  # Keep current, let router decide
    else:
        # Apply recovery strategy to modify tactic
        modified_tactic = modifier.apply_strategy(current_tactic, strategy, context)

    # Extract definitions from error for UNFOLD_MORE strategy
    definitions = modifier.extract_definitions_from_error(error_content)

    # Create recovery record
    recovery_record = RecoveryRecord(
        step=step,
        error_type=error_type,
        severity=severity.value,
        strategy=strategy.value,
        original_tactic=current_tactic,
        modified_tactic=modified_tactic,
        success=False,  # Will be updated by next observation
    )

    logger.info(
        f"Recovery: {error_type} ({severity.value}) -> {strategy.value}: "
        f"{modified_tactic[:50]}..."
    )

    return {
        "current_proof": modified_tactic,
        "current_error_type": error_type,
        "current_severity": severity.value,
        "active_strategy": strategy.value,
        "recovery_stage": recovery_stage + 1,
        "recovery_attempts": state["recovery_attempts"] + 1,
        "tried_tactics": [current_tactic],  # Appended via reducer
        "definitions_to_unfold": definitions if definitions else state.get("definitions_to_unfold", []),
        "recovery_records": [recovery_record],  # Appended via reducer
    }


def reset_recovery_stage(state: ProofState) -> dict[str, Any]:
    """
    Reset recovery stage after a successful verification or new attempt.

    Called when transitioning from recovery back to normal flow.
    """
    return {
        "recovery_stage": 0,
        "current_error_type": None,
        "active_strategy": None,
    }


# ==============================================================================
# Router Node
# ==============================================================================

def router_node(state: ProofState) -> str:
    """
    Decide the next step based on current state.

    Returns one of:
    - "continue": Go back to reasoning for another attempt
    - "backtrack": Restore to checkpoint (Phase 4)
    - "success": Proof verified, terminate
    - "failed": Max attempts or early termination

    This is a routing function used by LangGraph's conditional edges.
    """
    # Check terminal conditions first
    if state["status"] == ProofStatus.SUCCESS.value:
        return "success"

    if state["status"] == ProofStatus.FAILED.value:
        return "failed"

    # Check max attempts
    if state["attempt_count"] >= state["max_attempts"]:
        logger.info(f"Max attempts ({state['max_attempts']}) reached")
        return "failed"

    # Check for backtrack conditions
    if should_backtrack(state):
        logger.info("Backtrack condition detected")
        # For now, treat backtrack as continue with new attempt
        # Phase 4 will implement actual checkpoint restoration
        return "continue"

    # Check observations for early termination patterns
    if _should_terminate_early(state):
        return "failed"

    # Continue with next attempt
    return "continue"


def _should_terminate_early(state: ProofState) -> bool:
    """
    Check if we should terminate early based on patterns.

    Implements Poetiq self-auditing patterns:
    - Divergence: complexity increasing
    - Oscillation: same error repeating
    """
    # Check oscillation (same error 3 times)
    errors = state["error_history"]
    if len(errors) >= 3:
        if errors[-1] == errors[-2] == errors[-3]:
            logger.info("Oscillation detected: same error 3 times")
            return True

    # Check all observations failed with same error type
    observations = state["observations"]
    if len(observations) >= 3:
        recent = observations[-3:]
        if all(not o["success"] for o in recent):
            error_types = [o.get("error_type") for o in recent]
            if len(set(error_types)) == 1 and error_types[0] is not None:
                logger.info(f"Same error type ({error_types[0]}) 3 times")
                return True

    return False


# ==============================================================================
# OpenManus Router Node
# ==============================================================================

def om_router_node(state: ProofState) -> str:
    """
    OpenManus-aware router for OM_REACT mode.

    Routes to "recover" when:
    - Error is RECOVERABLE or TRANSIENT
    - Recovery stage < 2 (haven't exhausted recovery options)
    - Haven't exceeded max recovery attempts

    Returns one of:
    - "continue": Go back to reasoning for another attempt
    - "recover": Route to recovery_node for error analysis
    - "success": Proof verified, terminate
    - "failed": Max attempts or unrecoverable error
    """
    # Check terminal conditions first
    if state["status"] == ProofStatus.SUCCESS.value:
        return "success"

    if state["status"] == ProofStatus.FAILED.value:
        return "failed"

    # Check max attempts
    if state["attempt_count"] >= state["max_attempts"]:
        logger.info(f"Max attempts ({state['max_attempts']}) reached")
        return "failed"

    # Get latest observation
    latest_obs = state["observations"][-1] if state["observations"] else None

    # If no observation or success, continue normally
    if not latest_obs:
        return "continue"

    if latest_obs.get("success"):
        # Reset recovery stage on success
        return "success"

    # Error occurred - decide whether to recover or continue
    error_type = latest_obs.get("error_type", "unknown")
    recovery_stage = state["recovery_stage"]

    # Check if we should attempt recovery
    if recovery_stage < 2:  # Still have recovery stages available
        # Classify error to check severity
        controller = ErrorRecoveryController()
        severity, _ = controller.classify_error(error_type)

        # Recoverable and transient errors can be recovered
        if severity in (ErrorSeverity.RECOVERABLE, ErrorSeverity.TRANSIENT):
            logger.debug(f"Routing to recovery for {error_type} (stage {recovery_stage})")
            return "recover"

        # Fatal errors skip recovery
        if severity == ErrorSeverity.FATAL:
            logger.info(f"Fatal error {error_type} - skipping recovery")
            return "failed"

    # Recovery exhausted - check for early termination
    if _should_terminate_early(state):
        return "failed"

    # Check for backtrack conditions
    if should_backtrack(state):
        logger.info("Backtrack condition detected")
        # For now, continue (Phase 4 will implement checkpoints)
        return "continue"

    # Continue with next attempt
    return "continue"


# ==============================================================================
# Increment Attempt Node
# ==============================================================================

def increment_attempt_node(state: ProofState) -> dict[str, Any]:
    """
    Increment attempt counter when continuing to next iteration.

    This node is called when router returns "continue".
    """
    return {
        "attempt_count": state["attempt_count"] + 1,
    }


# ==============================================================================
# Termination Nodes
# ==============================================================================

def success_node(state: ProofState) -> dict[str, Any]:
    """Mark proof as successful."""
    return {
        "status": ProofStatus.SUCCESS.value,
    }


def failed_node(state: ProofState) -> dict[str, Any]:
    """Mark proof as failed."""
    return {
        "status": ProofStatus.FAILED.value,
    }


# ==============================================================================
# ROMA Nodes (Hierarchical Decomposition)
# ==============================================================================

async def complexity_analysis_node(state: ProofState) -> dict[str, Any]:
    """
    Analyze proof goal complexity for ROMA mode.

    This is the first node in the ROMA graph. It:
    1. Analyzes the goal state complexity
    2. Scores nesting, quantifiers, type complexity
    3. Estimates automation likelihood

    Returns:
        Partial state update with complexity analysis results
    """
    analyzer = GoalComplexityAnalyzer()

    goal_state = state["goal_state"]
    context = state["file_content"]
    attempt_count = state["attempt_count"]
    errors = list(state["error_history"])

    # Run complexity analysis
    score = analyzer.analyze(
        goal_state=goal_state,
        context=context,
        previous_attempts=attempt_count,
        previous_errors=errors,
    )

    logger.info(
        f"ROMA complexity: {score.complexity.value} "
        f"(score={score.overall_score:.2f}, auto={score.automation_likelihood:.2f})"
    )

    return {
        "roma_active": True,
        "roma_complexity": score.complexity.value,
        "roma_complexity_score": score.overall_score,
        "step": state["step"] + 1,
    }


async def atomizer_node(state: ProofState) -> dict[str, Any]:
    """
    Decide whether to solve goal directly or decompose.

    This node uses the Atomizer to make the atomic/decompose decision
    based on complexity analysis and attempt history.

    Returns:
        Partial state update with atomizer decision
    """
    atomizer = Atomizer()

    decision = await atomizer.should_decompose(
        goal_state=state["goal_state"],
        context=state["file_content"],
        previous_attempts=state["attempt_count"],
        previous_errors=list(state["error_history"]),
        tried_tactics=list(state["tried_tactics"]),
    )

    logger.info(
        f"ROMA atomizer: {'DECOMPOSE' if decision.should_decompose else 'DIRECT'} "
        f"(confidence={decision.confidence:.2f})"
    )

    return {
        "roma_strategy": decision.suggested_strategy if decision.should_decompose else None,
    }


def roma_router_node(state: ProofState) -> str:
    """
    Route based on atomizer decision.

    Returns one of:
    - "direct": Solve goal directly (atomic)
    - "decompose": Decompose into subtasks
    - "success": Already succeeded
    - "failed": Already failed
    """
    if state["status"] == ProofStatus.SUCCESS.value:
        return "success"

    if state["status"] == ProofStatus.FAILED.value:
        return "failed"

    # Check if decomposition was suggested
    if state.get("roma_strategy"):
        return "decompose"

    return "direct"


async def planner_node(state: ProofState) -> dict[str, Any]:
    """
    Create decomposition plan for complex goal.

    Uses RomaPlanner to break the goal into subtasks with dependencies.

    Returns:
        Partial state update with decomposition plan
    """
    planner = RomaPlanner()

    # Build complexity score from state (for enhanced planning)
    from .roma.complexity import ComplexityScore

    complexity_score = ComplexityScore(
        overall_score=state.get("roma_complexity_score", 0.5),
        complexity=GoalComplexity(state.get("roma_complexity", "simple")),
    )

    plan = await planner.decompose(
        goal_state=state["goal_state"],
        context=state["file_content"],
        complexity_score=complexity_score,
        suggested_strategy=state.get("roma_strategy"),
        rag_results=state.get("rag_results", []),
    )

    logger.info(
        f"ROMA planner: {plan.strategy.value} with {len(plan.subtasks)} subtasks"
    )

    # Serialize plan for state storage
    plan_dict = {
        "strategy": plan.strategy.value,
        "subtasks": [
            {
                "id": st.id,
                "description": st.description,
                "goal_hint": st.goal_hint,
                "suggested_tactics": st.suggested_tactics,
                "dependencies": st.dependencies,
                "is_critical": st.is_critical,
                "max_attempts": st.max_attempts,
            }
            for st in plan.subtasks
        ],
        "synthesis_strategy": plan.synthesis_strategy,
        "entry_tactic": plan.entry_tactic,
        "exit_tactic": plan.exit_tactic,
    }

    # Set first ready subtask
    first_subtask = plan.subtasks[0].id if plan.subtasks else None

    return {
        "roma_plan": plan_dict,
        "roma_current_subtask": first_subtask,
        "current_proof": plan.entry_tactic or state["current_proof"],
    }


async def subtask_executor_node(
    state: ProofState,
    verifier_service: Optional["VerifierService"] = None,
) -> dict[str, Any]:
    """
    Execute the current subtask.

    This node:
    1. Gets the current subtask from the plan
    2. Tries suggested tactics in order
    3. Records results and updates state

    Returns:
        Partial state update with subtask results
    """
    plan_dict = state.get("roma_plan")
    if not plan_dict:
        logger.warning("subtask_executor called without plan")
        return {"status": ProofStatus.FAILED.value}

    current_id = state.get("roma_current_subtask")
    if not current_id:
        logger.warning("No current subtask to execute")
        return {}

    # Find current subtask
    subtask_data = None
    for st in plan_dict.get("subtasks", []):
        if st["id"] == current_id:
            subtask_data = st
            break

    if not subtask_data:
        logger.warning(f"Subtask {current_id} not found in plan")
        return {}

    sorry = dict_to_sorry(state["sorry_location"])
    success = False
    tactics_used: list[str] = []
    error: Optional[str] = None
    final_goal = state.get("goal_state", "")

    # Get initial goal state for this subtask
    # Use current goal_state or subtask's goal_hint as starting point
    initial_goal = state.get("goal_state", subtask_data.get("goal_hint", ""))

    # Build context for tactic generation
    context = {
        "theorem_name": sorry.theorem_name,
        "subtask_description": subtask_data.get("description", ""),
        "goal_hint": subtask_data.get("goal_hint", ""),
        "suggested_tactics": subtask_data.get("suggested_tactics", []),
    }

    # Extract MCP client from verifier_service if available
    mcp_client = None
    if verifier_service and verifier_service._status.mcp_available:
        mcp_client = verifier_service._mcp_client
        logger.debug("Using connected MCP client from VerifierService for subtask")

    # Use iterative tactic loop (Phase 4.4)
    try:
        (
            tactics_used, success,
            final_goal, _subtask_records,
        ) = await iterative_tactic_loop(
            goal_state=initial_goal,
            file_path=sorry.file_path,
            line=sorry.line,
            model=state["model_used"],
            temperature=state["base_temperature"],
            max_steps=state.get("max_tactic_steps", 15),
            consecutive_failure_threshold=3,
            context=context,
            mcp_client=mcp_client,
            attempt_number=state.get("attempt_count", 1),
            sorry_index=1,
        )
    except Exception as e:
        logger.warning(f"Iterative tactic loop failed: {e}")
        error = str(e)

        # Fallback: try suggested tactics directly if iterative loop fails
        for tactic in subtask_data.get("suggested_tactics", [])[:3]:
            tactics_used.append(tactic)
            if verifier_service:
                try:
                    success, errors, _ = await verifier_service.verify_proof_on_copy(
                        sorry=sorry,
                        proof_code=tactic,
                        attempt=state["attempt_count"],
                        model_used=state["model_used"],
                        temperature=state["base_temperature"],
                    )
                    if success:
                        break
                    if errors:
                        error = errors[0]
                except Exception as fallback_e:
                    error = str(fallback_e)

    # Record subtask result
    subtask_record = SubTaskRecord(
        subtask_id=current_id,
        description=subtask_data.get("description", ""),
        goal_hint=subtask_data.get("goal_hint", ""),
        tactics_used=tactics_used,
        success=success,
        attempts=len(tactics_used),
        error=error,
    )

    # Create sub-proof if successful
    sub_proofs = list(state.get("roma_sub_proofs", []))
    if success:
        sub_proofs.append({
            "subtask_id": current_id,
            "tactic_sequence": tactics_used,
            "success": True,
        })

    # Update completed subtasks
    completed = list(state.get("roma_completed_subtasks", []))
    if success:
        completed.append(current_id)

    # Find next subtask
    next_subtask = None
    completed_set = set(completed)
    for st in plan_dict.get("subtasks", []):
        st_id = st["id"]
        if st_id in completed_set:
            continue
        deps = st.get("dependencies", [])
        if all(d in completed_set for d in deps):
            next_subtask = st_id
            break

    logger.info(
        f"ROMA subtask {current_id}: {'SUCCESS' if success else 'FAILED'}, "
        f"next={next_subtask}"
    )

    # Build goal state history
    goal_history = list(state.get("goal_state_history", []))
    if final_goal:
        goal_history.append(final_goal)

    return {
        "roma_subtask_records": [subtask_record],
        "roma_sub_proofs": sub_proofs,
        "roma_completed_subtasks": completed,
        "roma_current_subtask": next_subtask,
        "tried_tactics": tactics_used,
        # Phase 4.4: Iterative refinement state updates
        "tactic_sequence": tactics_used,  # Append via reducer
        "goal_state_history": goal_history,
        "goal_state": final_goal,  # Update current goal state
        "tactic_step": state.get("tactic_step", 0) + len(tactics_used),
        "consecutive_failures": 0 if success else state.get("consecutive_failures", 0) + 1,
    }


def subtask_router_node(state: ProofState) -> str:
    """
    Route after subtask execution.

    Returns one of:
    - "next_subtask": More subtasks to execute
    - "aggregate": All subtasks done, synthesize proof
    - "failed": Critical subtask failed
    """
    plan_dict = state.get("roma_plan")
    if not plan_dict:
        return "failed"

    completed = set(state.get("roma_completed_subtasks", []))
    all_subtasks = plan_dict.get("subtasks", [])

    # Check if all subtasks completed
    all_done = all(st["id"] in completed for st in all_subtasks)
    if all_done:
        return "aggregate"

    # Check if there's a next subtask to execute
    if state.get("roma_current_subtask"):
        return "next_subtask"

    # Check for critical failures
    records = state.get("roma_subtask_records", [])
    for st in all_subtasks:
        if st.get("is_critical", True):
            st_id = st["id"]
            # Find if this subtask failed
            for rec in records:
                if rec["subtask_id"] == st_id and not rec["success"]:
                    logger.info(f"Critical subtask {st_id} failed")
                    return "failed"

    # If no next subtask but not all done, try to aggregate partial
    return "aggregate"


async def aggregator_node(state: ProofState) -> dict[str, Any]:
    """
    Synthesize sub-proofs into final proof.

    Uses RomaAggregator to combine completed sub-proofs according
    to the plan's synthesis strategy.

    Returns:
        Partial state update with aggregated proof
    """
    aggregator = RomaAggregator()

    plan_dict = state.get("roma_plan")
    if not plan_dict:
        return {"status": ProofStatus.FAILED.value}

    # Reconstruct plan and sub-proofs
    from .roma.planner import DecompositionPlan, SubTask, DecompositionStrategy

    subtasks = [
        SubTask(
            id=st["id"],
            description=st.get("description", ""),
            goal_hint=st.get("goal_hint", ""),
            suggested_tactics=st.get("suggested_tactics", []),
            dependencies=st.get("dependencies", []),
            is_critical=st.get("is_critical", True),
        )
        for st in plan_dict.get("subtasks", [])
    ]

    plan = DecompositionPlan(
        strategy=DecompositionStrategy(plan_dict.get("strategy", "sequential")),
        subtasks=subtasks,
        synthesis_strategy=plan_dict.get("synthesis_strategy", "sequential"),
        entry_tactic=plan_dict.get("entry_tactic"),
        exit_tactic=plan_dict.get("exit_tactic"),
    )

    # Build sub-proofs
    sub_proofs = [
        SubProof(
            subtask_id=sp["subtask_id"],
            tactic_sequence=sp.get("tactic_sequence", []),
            success=sp.get("success", False),
        )
        for sp in state.get("roma_sub_proofs", [])
    ]

    # Synthesize
    result = await aggregator.synthesize(plan, sub_proofs)

    logger.info(
        f"ROMA aggregation: {result.result.value}, "
        f"proofs_used={len(result.sub_proofs_used)}, gaps={len(result.gaps)}"
    )

    from .roma.aggregator import SynthesisResult

    if result.result == SynthesisResult.SUCCESS:
        return {
            "roma_aggregated_proof": result.combined_proof,
            "current_proof": result.combined_proof,
            "status": ProofStatus.SUCCESS.value,
        }
    elif result.result == SynthesisResult.PARTIAL:
        # Partial success - might need more work
        return {
            "roma_aggregated_proof": result.combined_proof,
            "current_proof": result.combined_proof,
        }
    else:
        # Failed synthesis
        return {
            "status": ProofStatus.FAILED.value,
        }


# ==============================================================================
# Direct Iterative Node (Phase 4.4)
# ==============================================================================


async def direct_iterative_node(
    state: ProofState,
    verifier_service: Optional["VerifierService"] = None,
) -> dict[str, Any]:
    """
    Direct path node for atomic goals using iterative tactic loop.

    Replaces the ReAct chain for atomic goals in ROMA. Iteratively applies
    tactics using edit_file for safe edit-check-revert testing.

    Includes crash recovery: retries up to 3 times on MCPWorkerCrashError
    with exponential backoff, falls back to single-shot on MCPUnavailableError.

    Args:
        state: Current ProofState
        verifier_service: VerifierService for LSP verification (optional,
                         used as fallback if iterative loop fails)

    Returns:
        Partial state update with:
        - status: ProofStatus.SUCCESS or ProofStatus.FAILED
        - current_proof: The successful proof if found
        - tactic_sequence: Tactics that succeeded
        - goal_state: Final goal state after tactics
    """
    sorry = dict_to_sorry(state["sorry_location"])
    initial_goal = state.get("goal_state", "")
    model = state.get("model_used", "gemini-3-pro-preview")
    temperature = state.get("base_temperature", 0.2)
    max_steps = state.get("max_tactic_steps", 15)

    logger.info(
        f"Direct iterative node: {sorry.theorem_name} at line {sorry.line}"
    )

    # Build context from goal analysis and state
    context = {
        "theorem_name": sorry.theorem_name,
        "file_path": sorry.file_path,
    }

    # Add goal analysis if available (from goal_parser_node)
    goal_analysis = state.get("goal_analysis")
    if goal_analysis:
        context["goal_type"] = goal_analysis.get("goal_type", "")
        context["hypothesis_names"] = [
            h.get("name") for h in goal_analysis.get("hypotheses", [])
        ]
        context["rewrite_candidates"] = goal_analysis.get("rewrite_candidates", [])
        context["definitions_in_scope"] = goal_analysis.get("definitions_in_scope", [])

    # Add RAG results if available
    rag_results = state.get("rag_results", [])
    if rag_results:
        context["rag_tactics"] = [r.get("content", "") for r in rag_results[:3]]

    # Extract MCP client from verifier_service if available
    mcp_client = None
    if verifier_service and verifier_service._status.mcp_available:
        mcp_client = verifier_service._mcp_client
        logger.debug("Using connected MCP client from VerifierService")

    # Run iterative tactic loop with crash recovery
    import asyncio
    from verifier.mcp_client import (
        MCPUnavailableError,
        MCPWorkerCrashError,
    )

    tactics_applied: list[str] = []
    attempt_records: list[AttemptRecord] = []
    success = False
    final_goal = initial_goal

    # Extract attempt number and sorry index for VP files
    attempt_number = state.get("attempt_count", 1)
    sorry_data = state.get("sorry_location", {})
    sorry_index = (
        sorry_data.get("index", 1)
        if isinstance(sorry_data, dict) else 1
    )

    max_retries = 3
    for retry in range(max_retries):
        try:
            (
                tactics_applied, success,
                final_goal, attempt_records,
            ) = await iterative_tactic_loop(
                goal_state=initial_goal,
                file_path=sorry.file_path,
                line=sorry.line,
                model=model,
                temperature=temperature,
                max_steps=max_steps,
                consecutive_failure_threshold=3,
                context=context,
                mcp_client=mcp_client,
                attempt_number=attempt_number,
                sorry_index=sorry_index,
            )
            break  # Normal exit
        except MCPWorkerCrashError as e:
            logger.error(
                f"MCP crashed (retry {retry+1}/{max_retries}): "
                f"{e}"
            )
            mcp_client = None
            if retry < max_retries - 1:
                await asyncio.sleep(2 ** (retry + 1))
        except MCPUnavailableError as e:
            logger.error(f"MCP unavailable: {e}")
            mcp_client = None
            break
        except Exception as e:
            logger.warning(
                f"Iterative tactic loop failed: {e}"
            )
            break

    # Fallback: try simple verification
    if not tactics_applied and not success:
        if verifier_service and state.get("current_proof"):
            try:
                fallback_ok, errors, _ = (
                    await verifier_service.verify_proof_on_copy(
                        sorry=sorry,
                        proof_code=state["current_proof"],
                        attempt=state.get("attempt_count", 1),
                        model_used=model,
                        temperature=temperature,
                    )
                )
                if fallback_ok:
                    tactics_applied = [state["current_proof"]]
                    success = True
                    final_goal = "no goals"
            except Exception as fallback_e:
                logger.warning(
                    f"Fallback verification also failed: "
                    f"{fallback_e}"
                )

    # Build combined proof from tactics
    combined_proof = (
        "; ".join(tactics_applied)
        if tactics_applied
        else state.get("current_proof", "")
    )

    # Update goal state history
    goal_history = list(state.get("goal_state_history", []))
    if final_goal:
        goal_history.append(final_goal)

    # Determine VP file path
    from pathlib import Path as _Path
    vp_path_str: Optional[str] = None
    if sorry.file_path:
        original = _Path(sorry.file_path)
        vp_name = (
            f"{original.stem}"
            f"_VP{attempt_number}"
            f"{original.suffix}"
        )
        vp_candidate = original.parent / vp_name
        if vp_candidate.exists():
            vp_path_str = str(vp_candidate)

    logger.info(
        f"Direct iterative node result: "
        f"{'SUCCESS' if success else 'FAILED'}, "
        f"tactics={len(tactics_applied)}"
    )

    # Return state updates
    if success:
        return {
            "status": ProofStatus.SUCCESS.value,
            "current_proof": combined_proof,
            "tried_tactics": tactics_applied,
            "tactic_sequence": tactics_applied,
            "goal_state_history": goal_history,
            "goal_state": final_goal,
            "tactic_step": (
                state.get("tactic_step", 0)
                + len(tactics_applied)
            ),
            "consecutive_failures": 0,
            "attempt_records": attempt_records,
            "output_file": vp_path_str,
        }
    else:
        return {
            "status": ProofStatus.FAILED.value,
            "tried_tactics": tactics_applied,
            "tactic_sequence": tactics_applied,
            "goal_state_history": goal_history,
            "goal_state": final_goal,
            "tactic_step": (
                state.get("tactic_step", 0)
                + len(tactics_applied)
            ),
            "consecutive_failures": (
                state.get("consecutive_failures", 0) + 1
            ),
            "attempt_records": attempt_records,
            "output_file": vp_path_str,
        }
