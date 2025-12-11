"""
Dynamic prompt loading from markdown files.

Loads prompts from veripilot/prompts/ directory, enabling prompt iteration
without code changes. Uses LRU cache for performance.
"""

from pathlib import Path
from functools import lru_cache
import re
import logging

logger = logging.getLogger(__name__)

# Prompts directory relative to this file
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


@lru_cache(maxsize=32)
def load_prompt(name: str, version: str = "v1", category: str = "verifier") -> str:
    """
    Load a prompt from prompts/{category}/{name}_{version}.md.

    Args:
        name: Prompt name (e.g., "system_prompt", "retry_guidance")
        version: Version string (default "v1")
        category: Prompt category/subdirectory (default "verifier")

    Returns:
        Prompt content as string

    Raises:
        FileNotFoundError: If prompt file not found
    """
    # Try category-specific path first
    filename = f"{name}_{version}.md"
    category_path = PROMPTS_DIR / category / filename

    if category_path.exists():
        content = category_path.read_text().strip()
        logger.debug(f"Loaded prompt from {category_path}")
        return content

    # Try root prompts directory as fallback
    root_path = PROMPTS_DIR / filename
    if root_path.exists():
        content = root_path.read_text().strip()
        logger.debug(f"Loaded prompt from {root_path}")
        return content

    raise FileNotFoundError(
        f"Prompt not found: {name}_{version}.md "
        f"(searched: {category_path}, {root_path})"
    )


def get_latest_version(name: str, category: str = "verifier") -> str:
    """
    Find the latest version of a prompt.

    Args:
        name: Prompt name
        category: Prompt category

    Returns:
        Version string (e.g., "v1", "v2")
    """
    pattern = re.compile(rf"{re.escape(name)}_v(\d+)\.md")
    versions = []

    category_dir = PROMPTS_DIR / category
    if category_dir.exists():
        for path in category_dir.glob(f"{name}_v*.md"):
            match = pattern.match(path.name)
            if match:
                versions.append(int(match.group(1)))

    return f"v{max(versions)}" if versions else "v1"


def load_latest_prompt(name: str, category: str = "verifier") -> str:
    """
    Load the latest version of a prompt.

    Args:
        name: Prompt name
        category: Prompt category

    Returns:
        Prompt content
    """
    version = get_latest_version(name, category)
    return load_prompt(name, version, category)


def list_prompts(category: str = "verifier") -> list[str]:
    """
    List all available prompts in a category.

    Args:
        category: Prompt category

    Returns:
        List of prompt names (without version suffixes)
    """
    category_dir = PROMPTS_DIR / category
    if not category_dir.exists():
        return []

    pattern = re.compile(r"(.+)_v\d+\.md")
    names = set()

    for path in category_dir.glob("*_v*.md"):
        match = pattern.match(path.name)
        if match:
            names.add(match.group(1))

    return sorted(names)


def clear_cache() -> None:
    """Clear the prompt cache (useful for testing)."""
    load_prompt.cache_clear()
