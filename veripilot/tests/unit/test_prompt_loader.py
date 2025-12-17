"""
Tests for prompt_loader module.

Tests dynamic prompt loading from markdown files.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from agent.prompt_loader import (
    load_prompt,
    get_latest_version,
    load_latest_prompt,
    list_prompts,
    clear_cache,
    PROMPTS_DIR,
)


class TestLoadPrompt:
    """Tests for load_prompt function."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_load_system_prompt_latest(self):
        """Test loading the universal system prompt (latest version)."""
        prompt = load_latest_prompt("system_prompt")
        assert "Lean 4" in prompt
        assert "sorry" in prompt

    def test_load_retry_guidance(self):
        """Test loading retry guidance prompt."""
        prompt = load_prompt("retry_guidance")
        assert "Attempt" in prompt
        assert "simpler" in prompt.lower() or "Guidance" in prompt

    def test_load_nonexistent_prompt_raises(self):
        """Test that loading nonexistent prompt raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent_prompt_xyz")

    def test_load_prompt_caching(self):
        """Test that prompts are cached via load_latest_prompt."""
        # load_latest_prompt internally calls load_prompt which is cached
        prompt1 = load_latest_prompt("system_prompt")
        prompt2 = load_latest_prompt("system_prompt")

        # Should be same object due to caching
        assert prompt1 is prompt2


class TestGetLatestVersion:
    """Tests for get_latest_version function."""

    def test_get_latest_version_existing(self):
        """Test getting latest version of existing prompt."""
        version = get_latest_version("system_prompt")
        assert version.startswith("v")
        assert version[1:].isdigit()

    def test_get_latest_version_nonexistent(self):
        """Test getting version of nonexistent prompt returns v1."""
        version = get_latest_version("nonexistent_prompt")
        assert version == "v1"


class TestListPrompts:
    """Tests for list_prompts function."""

    def test_list_prompts_verifier(self):
        """Test listing prompts in verifier category."""
        prompts = list_prompts("verifier")
        assert "system_prompt" in prompts
        # Model-specific prompts moved to legacy/, only universal prompt remains
        assert "retry_guidance" in prompts

    def test_list_prompts_nonexistent_category(self):
        """Test listing prompts in nonexistent category returns empty."""
        prompts = list_prompts("nonexistent_category")
        assert prompts == []


class TestLoadLatestPrompt:
    """Tests for load_latest_prompt function."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_load_latest_prompt(self):
        """Test loading latest version of prompt."""
        prompt = load_latest_prompt("system_prompt")
        assert "Lean 4" in prompt


class TestPromptIntegration:
    """Integration tests for prompt loading with prompts.py."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_prompts_py_uses_loader(self):
        """Test that prompts.py can use the prompt loader."""
        from agent.prompts import build_system_prompt

        # Should load from file (or fall back gracefully)
        prompt = build_system_prompt("default")
        assert "Lean 4" in prompt

        prompt_gemini = build_system_prompt("gemini")
        assert "Lean 4" in prompt_gemini

        prompt_claude = build_system_prompt("claude")
        assert "Lean 4" in prompt_claude

    def test_aristotle_returns_empty(self):
        """Test that Aristotle model returns empty system prompt."""
        from agent.prompts import build_system_prompt

        prompt = build_system_prompt("aristotle")
        assert prompt == ""
