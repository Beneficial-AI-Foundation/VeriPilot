# Lean Timeout Configuration

## Overview

This document explains how to configure timeouts and resource limits for Lean 4 projects to prevent runaway tactics from hanging the Language Server Protocol (LSP) and blocking agent workflows.

## The Problem

When working with Lean 4 proof assistants (especially in automated agent workflows), certain tactics or proof strategies can consume excessive resources:

- **Runaway tactics**: Unbounded search tactics (e.g., `simp`, `omega`, custom decision procedures) may run indefinitely
- **Memory exhaustion**: Complex elaboration or unification can allocate large amounts of memory
- **LSP hangs**: Without timeouts, the language server becomes unresponsive, blocking both human and agent interactions

**Impact on agents**: An agent that proposes a resource-intensive tactic will hang waiting for LSP diagnostics/goals, causing the entire verification loop to stall.

## Recommended Configuration

### Timeout Values

Based on Lean community best practices and MCP integration experience:

| Setting | Value | Purpose |
|---------|-------|---------|
| `maxHeartbeats` | 200000 | Deterministic timeout via memory allocation counting |
| `timeout` | 20000 | Logical timeout (milliseconds or heartbeats, version-dependent) |
| `--memory` | 4096 | Maximum memory in MB (server CLI flag) |
| `--timeout` | 20000 | Allocation-based timeout (server CLI flag) |

These values balance:
- **Reasonable proof search**: Most legitimate tactics finish in <10s
- **Fail-fast on errors**: Runaway tactics fail within 20s instead of hanging indefinitely
- **Agent resilience**: Timeouts produce diagnostics that agents can react to (revert edit, try alternative)

### Configuration via `lakefile.lean`

Lean 4 projects use Lake as their build system. Lake exposes two mechanisms to configure the language server:

1. **`leanOptions`**: Logical options passed to the Lean compiler/elaborator
2. **`moreServerOptions`**: CLI flags passed to `lean --server`

Add these to your `lakefile.lean` package declaration:

```lean
package MyProject where
  leanOptions := #[
    ⟨`timeout, 20000⟩,         -- Logical timeout (semantics vary by Lean version)
    ⟨`maxHeartbeats, 200000⟩   -- Deterministic timeout via heartbeat counting
  ]
  moreServerOptions := #[
    "--memory=4096",           -- Memory limit (MB)
    "--timeout=20000"          -- Allocation-based timeout (CLI)
  ]
```

**Note**: The exact option names and semantics may vary slightly between Lean versions. Consult the [Lake README](https://github.com/leanprover/lean4/blob/master/src/lake/README.md) for your version.

## How Heartbeats Work

Lean implements **deterministic timeouts** via a mechanism called *heartbeats*:

- A "heartbeat" is roughly one small memory allocation or computation step
- The elaborator/tactic engine counts heartbeats during execution
- When `maxHeartbeats` is exceeded, elaboration fails with a timeout diagnostic

**Why deterministic?**
- Wall-clock timeouts are non-reproducible (depend on CPU speed, system load)
- Heartbeat-based timeouts ensure that the same proof always times out at the same complexity threshold
- This is critical for reproducible builds and testing

**Typical heartbeat consumption**:
- Simple tactics: 100-1000 heartbeats
- Complex `simp` calls: 10000-50000 heartbeats
- Runaway searches: 200000+ heartbeats (should be killed)

## Verification Steps

After adding timeout configuration to `lakefile.lean`:

1. **Restart the language server**:
   ```bash
   # If using lake serve directly
   killall lean
   lake serve

   # If using an editor (VS Code, Emacs, etc.)
   # Use the editor's "Restart Lean Server" command
   ```

2. **Test timeout behavior**:
   Create a test file with an intentionally expensive tactic:
   ```lean
   theorem timeout_test : True := by
     simp only [*]  -- Overly aggressive simp that might loop
   ```

   Expected outcome: Either completes quickly or produces a timeout diagnostic like:
   ```
   (deterministic) timeout at 'whnf', maximum number of heartbeats (200000) has been reached
   ```

3. **Monitor via LSP diagnostics**:
   - Tools like `lean_goal` (MCP) or `$/lean/plainGoal` (raw LSP) should return timeout diagnostics
   - Agents should parse these and treat them as "tactic failed, revert edit"

## Integration with MCP

When using `lean-lsp-mcp` (Model Context Protocol wrapper for Lean LSP):

- **Safe tools** (`lean_goal`, `lean_term_goal`, `lean_hover_info`): Query elaborated state without running new tactics — rarely timeout
- **Dangerous tools** (`lean_run_code`, `lean_multi_attempt`): Execute arbitrary code — always use with client-side timeouts (5-15s)

**Recommended MCP workflow**:
1. Configure Lean-side timeouts (this document) to ensure worst-case failure in 20s
2. Enforce client-side MCP tool timeouts (5-15s) to fail even faster
3. Use `lake build` before starting MCP server to avoid build timeouts

## References

- **Primary source**: Lean_MCP_pt2.md, Section 1.2
- **Lake options**: [Lake README](https://github.com/leanprover/lean4/blob/master/src/lake/README.md)

## Troubleshooting

| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| LSP still hangs despite config | Timeouts not applied (server not restarted) | Kill `lean` processes, restart server |
| All proofs timeout | Limits too aggressive | Increase `maxHeartbeats` to 500000 |
| Timeouts only on specific files | File imports expensive modules | Pre-build with `lake build`, or exclude heavy imports |
| MCP tools hang indefinitely | Client-side timeout missing | Add per-tool timeout (5-15s) in MCP client |
