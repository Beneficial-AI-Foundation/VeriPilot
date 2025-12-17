# Verifier Prompts

System prompts for the Lean 4 proof generation agent.

## Current Setup

**Active prompt:** `system_prompt_v2.md` (universal, used for all models)

The system uses `load_latest_prompt("system_prompt")` which automatically selects the highest version number (e.g., v2 over v1).

## File Naming Convention

```
{prompt_name}_v{version}.md
```

Examples:
- `system_prompt_v2.md` - Main system prompt (version 2)
- `retry_guidance_v1.md` - Retry-specific guidance

## Creating a New Prompt Version

1. Copy the current version: `cp system_prompt_v2.md system_prompt_v3.md`
2. Edit `system_prompt_v3.md` with your changes
3. The loader will automatically pick up v3 as the latest

No code changes needed - versioning is automatic.

## Adding Model-Specific Prompts (Future)

If you need model-specific prompts in the future:

1. **Create the prompt file:**
   ```
   system_prompt_gemini_v1.md
   system_prompt_claude_v1.md
   ```

2. **Update `src/agent/prompts.py`:**
   ```python
   def build_system_prompt(model: str = "default") -> str:
       if model == "aristotle":
           return ""

       # Try model-specific first
       if model in ("gemini", "claude"):
           loaded = _load_prompt_safe(f"system_prompt_{model}", use_latest=True)
           if loaded:
               return loaded

       # Fall back to universal prompt
       return _load_prompt_safe("system_prompt", use_latest=True) or SYSTEM_PROMPT_DEFAULT
   ```

## Prompt Loader API

```python
from agent.prompt_loader import load_prompt, load_latest_prompt, get_latest_version

# Load specific version
prompt = load_prompt("system_prompt", version="v2")

# Load latest version (recommended)
prompt = load_latest_prompt("system_prompt")

# Check what version is latest
version = get_latest_version("system_prompt")  # Returns "v2"
```

## Legacy Prompts

Previous model-specific prompts are preserved in `legacy/` for reference:
- `legacy/system_prompt_v1.md` - Original default prompt
- `legacy/system_prompt_gemini_v1.md` - Gemini-specific (deprecated)
- `legacy/system_prompt_claude_v1.md` - Claude-specific (deprecated)

These are not loaded by the system but kept for historical reference.
