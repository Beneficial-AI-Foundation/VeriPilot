#!/usr/bin/env python3
"""
Integration test for VeriPilot MVP.

Tests the full pipeline: Parser → Agent → Verifier
against actual Lean files and lake build (not mocked).

Usage:
    python scripts/test_integration.py --file ../lean-projects/dalek-verify-lean/ai-benchmark/tests/input.lean --line 56
    python scripts/test_integration.py --file ../lean-projects/dalek-verify-lean/ai-benchmark/tests/input.lean --all
"""

import asyncio
import sys
import argparse
import shutil
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load environment variables from .env file
def load_dotenv():
    """Load .env file if it exists."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

load_dotenv()

from parser import find_sorries, SorryLocation
from agent.llm_client import generate_proof
from verifier import verify_proof


def create_vp_copy(original_file: Path) -> Path:
    """
    Create a VP_ prefixed copy of the file for testing.

    Args:
        original_file: Path to original .lean file

    Returns:
        Path to VP_<name>.lean copy
    """
    vp_name = f"VP_{original_file.name}"
    vp_path = original_file.parent / vp_name

    # Copy the file
    shutil.copy2(original_file, vp_path)
    print(f"✓ Created test copy: {vp_path}")

    return vp_path


def cleanup_vp_copy(vp_file: Path):
    """Remove VP_ copy and backup files."""
    if vp_file.exists():
        vp_file.unlink()
        print(f"✓ Cleaned up: {vp_file}")

    # Also clean up any .bak files
    bak_file = Path(str(vp_file) + ".bak")
    if bak_file.exists():
        bak_file.unlink()
        print(f"✓ Cleaned up: {bak_file}")


async def test_single_sorry(
    original_file: Path,
    sorry_line: int,
    project_dir: Path,
    model: str = "gemini",
    rag = None,
):
    """
    Test filling a single sorry.

    Args:
        original_file: Original benchmark file
        sorry_line: Line number of sorry to fill
        project_dir: Lean project directory for lake build
        model: LLM model to use
        rag: RAG instance (optional)
    """
    print(f"\n{'='*60}")
    print(f"Testing sorry at {original_file.name}:{sorry_line}")
    print(f"{'='*60}\n")

    # Create VP_ copy
    vp_file = create_vp_copy(original_file)

    try:
        # Step 1: Find the sorry in the copy
        print(f"[1/3] Parsing {vp_file.name}...")
        sorries = find_sorries(str(vp_file))

        target_sorry = None
        for s in sorries:
            if s.line == sorry_line:
                target_sorry = s
                break

        if not target_sorry:
            print(f"✗ No sorry found at line {sorry_line}")
            print(f"  Available sorries at lines: {[s.line for s in sorries]}")
            return False

        print(f"✓ Found sorry: {target_sorry.theorem_name} at line {target_sorry.line}")
        print(f"  Signature: {target_sorry.theorem_signature[:80]}...")

        # Step 2: Generate proof with agent
        print(f"\n[2/3] Generating proof with {model}...")
        with open(vp_file, 'r') as f:
            file_content = f.read()

        proof_result = await generate_proof(
            sorry=target_sorry,
            file_content=file_content,
            rag=rag,
            model=model,
            max_attempts=1,  # Single attempt for now
        )

        if not proof_result.success:
            print(f"✗ Proof generation failed: {proof_result.error}")
            return False

        print(f"✓ Generated proof ({len(proof_result.proof_code)} chars):")
        print(f"  {proof_result.proof_code[:100]}...")
        if proof_result.rag_context:
            print(f"  Used {len(proof_result.rag_context)} RAG results")

        # Step 3: Verify proof with lake build
        print(f"\n[3/3] Verifying with lake build...")
        print(f"  (This may take 2-5 minutes...)")

        verification_result = await verify_proof(
            sorry=target_sorry,
            proof_result=proof_result,
            rag=rag,
            max_attempts=4,
            project_dir=str(project_dir),
            timeout=300,
        )

        print(f"\n{'='*60}")
        if verification_result.success:
            print(f"✅ SUCCESS! Proof verified in {verification_result.attempts} attempt(s)")
            print(f"  Time: {verification_result.elapsed_time:.1f}s")
            print(f"  Final proof:")
            for line in verification_result.proof_code.split('\n'):
                print(f"    {line}")
        else:
            print(f"❌ FAILED after {verification_result.attempts} attempt(s)")
            print(f"  Time: {verification_result.elapsed_time:.1f}s")
            print(f"  Errors:")
            for err in verification_result.errors[:3]:
                print(f"    - {err[:200]}")
        print(f"{'='*60}\n")

        return verification_result.success

    finally:
        # Always cleanup
        cleanup_vp_copy(vp_file)


async def test_all_sorries(
    original_file: Path,
    project_dir: Path,
    model: str = "gemini",
    rag = None,
):
    """Test all sorries in a file."""
    print(f"\n{'='*60}")
    print(f"Testing all sorries in {original_file.name}")
    print(f"{'='*60}\n")

    # Find all sorries in original
    sorries = find_sorries(str(original_file))
    print(f"Found {len(sorries)} sorries at lines: {[s.line for s in sorries]}\n")

    results = []
    for i, sorry in enumerate(sorries, 1):
        print(f"\n--- Sorry {i}/{len(sorries)} ---")
        success = await test_single_sorry(
            original_file=original_file,
            sorry_line=sorry.line,
            project_dir=project_dir,
            model=model,
            rag=rag,
        )
        results.append((sorry.line, success))

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {sum(r[1] for r in results)}/{len(results)} sorries filled")
    print(f"{'='*60}")
    for line, success in results:
        status = "✅" if success else "❌"
        print(f"  Line {line:3d}: {status}")
    print(f"{'='*60}\n")


async def main():
    parser = argparse.ArgumentParser(
        description="Integration test for VeriPilot MVP"
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to .lean file (relative to veripilot/ or absolute)",
    )
    parser.add_argument(
        "--line",
        type=int,
        help="Line number of sorry to test (omit to use --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all sorries in file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini",
        choices=["gemini", "claude", "claude-opus", "aristotle"],
        help="LLM model to use",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG (test agent only)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="../lean-projects/dalek-verify-lean",
        help="Lean project directory for lake build",
    )

    args = parser.parse_args()

    # Resolve paths
    file_path = Path(args.file)
    if not file_path.is_absolute():
        file_path = Path(__file__).parent.parent / file_path
    file_path = file_path.resolve()

    project_dir = Path(args.project)
    if not project_dir.is_absolute():
        project_dir = Path(__file__).parent.parent / project_dir
    project_dir = project_dir.resolve()

    # Validate inputs
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}")
        sys.exit(1)

    if not args.all and args.line is None:
        print("Error: Must specify either --line or --all")
        sys.exit(1)

    # RAG integration - for now, RAG is handled internally by agent
    # The agent's generate_proof() will use RAG if available
    rag = None
    if not args.no_rag:
        print("Note: RAG will be used internally by agent if available")
        print("  (No explicit initialization needed)\n")
    else:
        print("RAG disabled - testing agent without retrieval\n")

    # Run test
    if args.all:
        await test_all_sorries(
            original_file=file_path,
            project_dir=project_dir,
            model=args.model,
            rag=rag,
        )
    else:
        success = await test_single_sorry(
            original_file=file_path,
            sorry_line=args.line,
            project_dir=project_dir,
            model=args.model,
            rag=rag,
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
