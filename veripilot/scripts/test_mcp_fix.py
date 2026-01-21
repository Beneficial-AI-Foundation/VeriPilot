#!/usr/bin/env python3
"""
Test script for MCP client fix.

Tests the iterative tactic loop with proper MCP client passing.
Non-interactive - bypasses all CLI prompts.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parser import find_sorries
from verifier import VerifierService
from agent.react import ReActAgent, AgentMode
from rag.lean.llamaindex_lean import LeanRAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Test verification on M.lean with ROMA mode."""
    # M.lean has a simple sorry at line 23
    lean_file = Path(__file__).parent.parent.parent / \
                "lean-projects/dalek-verify-lean/Curve25519Dalek/Tests/M.lean"
    project_dir = Path(__file__).parent.parent.parent / \
                  "lean-projects/dalek-verify-lean"

    print(f"\n{'='*60}")
    print(f"Testing MCP fix on: {lean_file}")
    print(f"Project: {project_dir}")
    print(f"{'='*60}\n")

    # Find sorries
    sorries = find_sorries(str(lean_file))
    if not sorries:
        print("No sorries found!")
        return

    print(f"Found {len(sorries)} sorry(ies):")
    for s in sorries:
        print(f"  Line {s.line}: {s.theorem_name}")

    # Initialize VerifierService with LONGER timeout (150s to accommodate ~123s warmup)
    print("\n[1] Starting VerifierService with 150s MCP timeout...")
    verifier_service = VerifierService(
        str(project_dir),
        timeout=300,
        mcp_warmup_timeout=150,  # MCP warmup takes ~123s for this project
    )
    await verifier_service.start(wait_for_warmup=True)

    if verifier_service.status.mcp_available:
        print(f"✓ MCP is AVAILABLE (warmed up in {verifier_service.status.mcp_warm_up_time:.1f}s)")
        print("  → iterative_tactic_loop will use MCP multi_attempt")
    else:
        print("✗ MCP is NOT available - will use single-shot fallback")
        print("  → iterative_tactic_loop will use _single_shot_tactic_generation")

    # Initialize RAG (optional, but improves results)
    print("\n[2] Initializing RAG...")
    try:
        rag = LeanRAG()
        print("✓ RAG initialized")
    except Exception as e:
        print(f"✗ RAG failed: {e}")
        rag = None

    # Create agent
    print("\n[3] Creating ROMA agent...")
    agent = ReActAgent(
        model="gemini-openrouter",
        temperature=0.2,
        mode=AgentMode.ROMA,
        project_dir=str(project_dir),
    )

    # Run verification on first sorry
    sorry = sorries[0]
    print(f"\n[4] Running ROMA agent on {sorry.theorem_name} (line {sorry.line})...")
    print(f"{'='*60}")

    # Read file content
    file_content = lean_file.read_text()

    # Initial proof attempt (will be refined by agent)
    initial_proof = "simp [*]"

    try:
        result = await agent.verify(
            sorry=sorry,
            initial_proof=initial_proof,
            file_content=file_content,
            verifier_service=verifier_service,
            rag=rag,
            project_dir=str(project_dir),
        )

        print(f"\n{'='*60}")
        print("[5] RESULT:")
        if result.success:
            print(f"✓ SUCCESS! Proof found:")
            print(f"  {result.proof_code[:200]}...")
            print(f"  Attempts: {result.attempts}")
        else:
            print(f"✗ FAILED after {result.attempts} attempts")
            if result.errors:
                print(f"  Errors: {result.errors}")

    except Exception as e:
        print(f"\n✗ Agent error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("\n[6] Cleaning up...")
        await verifier_service.stop()
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
