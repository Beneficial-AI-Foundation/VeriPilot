#!/usr/bin/env python3
"""
Configure Lean 4 timeout and resource limits in lakefile.lean.

Parses a lakefile.lean and inserts recommended timeout configuration
blocks (leanOptions and moreServerOptions) into the package declaration.

Usage:
    python configure_lean_timeouts.py /path/to/lakefile.lean > lakefile.lean.new
    # Review changes, then:
    mv lakefile.lean.new lakefile.lean

Safety:
- Outputs to stdout (never overwrites original file)
- Warns if leanOptions/moreServerOptions already exist

Reference: docs/knowledge/lean-timeout-config.md
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Tuple


def parse_lakefile(content: str) -> Optional[Tuple[int, int]]:
    """
    Parse lakefile.lean to find the package declaration's 'where' clause.

    Returns:
        Tuple of (start_line, end_line) for the package where block,
        or None if not found. Line numbers are 0-indexed.
    """
    lines = content.split("\n")
    package_pattern = re.compile(r"^\s*package\s+\w+\s+where\s*$", re.IGNORECASE)

    for i, line in enumerate(lines):
        if package_pattern.match(line):
            end = len(lines)
            for j in range(i + 1, len(lines)):
                if re.match(r"^\s*(lean_lib|lean_exe|require|@\[)", lines[j]):
                    end = j
                    break
            return (i, end)

    return None


def check_existing_config(content: str) -> Tuple[bool, bool]:
    """Check if leanOptions or moreServerOptions already exist."""
    has_lean_options = bool(re.search(r"\bleanOptions\s*:=", content))
    has_more_server_options = bool(re.search(r"\bmoreServerOptions\s*:=", content))
    return (has_lean_options, has_more_server_options)


def generate_lean_options_block() -> str:
    """Generate the leanOptions configuration block."""
    return """  leanOptions := #[
    \u27e8`timeout, 20000\u27e9,         -- Logical timeout (ms or heartbeats)
    \u27e8`maxHeartbeats, 200000\u27e9   -- Deterministic timeout via heartbeat counting
  ]"""


def generate_server_options_block() -> str:
    """Generate the moreServerOptions configuration block."""
    return """  moreServerOptions := #[
    "--memory=4096",           -- Memory limit in MB
    "--timeout=20000"          -- Allocation-based timeout (CLI)
  ]"""


def update_lakefile(content: str) -> str:
    """
    Insert timeout configuration into lakefile.lean content.

    Warns if config already exists. Outputs modified content.
    """
    has_lean_opts, has_server_opts = check_existing_config(content)

    if has_lean_opts and has_server_opts:
        print(
            "Warning: Both leanOptions and moreServerOptions already exist.",
            file=sys.stderr,
        )
        print("Skipping insertion to avoid duplicates.", file=sys.stderr)
        return content

    if has_lean_opts or has_server_opts:
        existing = "leanOptions" if has_lean_opts else "moreServerOptions"
        missing = "moreServerOptions" if has_lean_opts else "leanOptions"
        print(f"Warning: {existing} already exists, {missing} is missing.", file=sys.stderr)
        print("Inserting missing block only.", file=sys.stderr)

    package_range = parse_lakefile(content)
    if package_range is None:
        print(
            "Error: Could not find 'package <name> where' declaration.",
            file=sys.stderr,
        )
        sys.exit(1)

    start_line, _ = package_range
    lines = content.split("\n")

    insert_pos = start_line + 1
    blocks_to_insert = []

    if not has_lean_opts:
        blocks_to_insert.append(generate_lean_options_block())
    if not has_server_opts:
        blocks_to_insert.append(generate_server_options_block())

    for block in blocks_to_insert:
        lines.insert(insert_pos, block)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Configure Lean 4 timeouts in lakefile.lean",
        epilog="Output goes to stdout. Redirect to a new file and review before replacing.",
    )
    parser.add_argument("lakefile", type=Path, help="Path to lakefile.lean")

    args = parser.parse_args()

    if not args.lakefile.exists():
        print(f"Error: File not found: {args.lakefile}", file=sys.stderr)
        sys.exit(1)

    content = args.lakefile.read_text()
    updated = update_lakefile(content)
    print(updated)

    print("\n# Done. Review output, then:", file=sys.stderr)
    print(f"#   python {sys.argv[0]} {args.lakefile} > lakefile.lean.new", file=sys.stderr)
    print("#   mv lakefile.lean.new lakefile.lean", file=sys.stderr)


if __name__ == "__main__":
    main()
