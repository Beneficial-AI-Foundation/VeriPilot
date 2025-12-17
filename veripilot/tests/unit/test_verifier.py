"""
Unit tests for the Verifier module.

Tests:
- File modification (backup/restore/replace)
- Error parsing
- Lake runner (mocked subprocess)
- Retry handler (mocked integration)
"""

import pytest
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from pathlib import Path


# Test fixtures
@dataclass
class MockSorryLocation:
    """Mock SorryLocation for testing."""

    file_path: str = "/test/file.lean"
    line: int = 10
    column: int = 5
    theorem_name: str = "test_theorem"
    theorem_signature: str = "theorem test_theorem (x : Nat) : x + 0 = x := by"
    proof_prefix: str = "  intro h"
    namespace: str = "Test"
    imports: list = field(default_factory=lambda: ["import Mathlib.Algebra.Basic"])


@dataclass
class MockProofResult:
    """Mock ProofResult for testing."""

    success: bool = True
    proof_code: str = "simp"
    model_used: str = "gemini"
    rag_context: list = field(default_factory=list)
    error: str = None
    attempts: int = 1
    temperature: float = 0.2


class TestFileModifier:
    """Tests for file_modifier.py functions."""

    def test_backup_file(self, tmp_path):
        """Test file backup creation."""
        from verifier.file_modifier import backup_file, restore_file

        # Create test file
        test_file = tmp_path / "test.lean"
        test_file.write_text("theorem foo : True := by sorry")

        # Backup
        backup_path = backup_file(str(test_file))

        assert os.path.exists(backup_path)
        assert backup_path == str(test_file) + ".bak"
        assert Path(backup_path).read_text() == "theorem foo : True := by sorry"

    def test_restore_file(self, tmp_path):
        """Test file restoration from backup."""
        from verifier.file_modifier import backup_file, restore_file

        # Create and backup test file
        test_file = tmp_path / "test.lean"
        original_content = "theorem foo : True := by sorry"
        test_file.write_text(original_content)

        backup_path = backup_file(str(test_file))

        # Modify the original
        test_file.write_text("modified content")
        assert test_file.read_text() == "modified content"

        # Restore
        restore_file(str(test_file), backup_path)
        assert test_file.read_text() == original_content

    def test_cleanup_backup(self, tmp_path):
        """Test backup cleanup."""
        from verifier.file_modifier import backup_file, cleanup_backup

        test_file = tmp_path / "test.lean"
        test_file.write_text("content")

        backup_path = backup_file(str(test_file))
        assert os.path.exists(backup_path)

        cleanup_backup(backup_path)
        assert not os.path.exists(backup_path)

    def test_format_proof_block_single_line(self):
        """Test formatting single-line proof."""
        from verifier.file_modifier import format_proof_block

        proof = "simp"
        result = format_proof_block(proof, 4)
        assert result == "simp"

    def test_format_proof_block_multi_line(self):
        """Test formatting multi-line proof."""
        from verifier.file_modifier import format_proof_block

        proof = "unfold f\nsimp\nrfl"
        result = format_proof_block(proof, 4)
        lines = result.split("\n")
        assert lines[0] == "unfold f"
        assert lines[1] == "    simp"
        assert lines[2] == "    rfl"

    def test_replace_sorry(self, tmp_path):
        """Test sorry replacement."""
        from verifier.file_modifier import replace_sorry

        # Create test file with sorry
        test_file = tmp_path / "test.lean"
        test_file.write_text(
            """theorem foo : True := by
  sorry
"""
        )

        sorry = MockSorryLocation(file_path=str(test_file), line=2)

        success = replace_sorry(str(test_file), sorry, "trivial")
        assert success

        content = test_file.read_text()
        assert "sorry" not in content
        assert "trivial" in content

    def test_replace_sorry_preserves_indentation(self, tmp_path):
        """Test that replacement preserves indentation."""
        from verifier.file_modifier import replace_sorry

        test_file = tmp_path / "test.lean"
        test_file.write_text(
            """theorem foo : True := by
    sorry
"""
        )

        sorry = MockSorryLocation(file_path=str(test_file), line=2)

        success = replace_sorry(str(test_file), sorry, "trivial")
        assert success

        content = test_file.read_text()
        assert "    trivial" in content

    def test_replace_sorry_multi_line_proof(self, tmp_path):
        """Test replacing sorry with multi-line proof."""
        from verifier.file_modifier import replace_sorry

        test_file = tmp_path / "test.lean"
        test_file.write_text(
            """theorem foo (n : Nat) : n + 0 = n := by
  sorry
"""
        )

        sorry = MockSorryLocation(file_path=str(test_file), line=2)

        success = replace_sorry(str(test_file), sorry, "induction n\nsimp\nrfl")
        assert success

        content = test_file.read_text()
        assert "sorry" not in content
        assert "induction n" in content

    def test_file_contains_sorry(self, tmp_path):
        """Test sorry detection."""
        from verifier.file_modifier import file_contains_sorry

        test_file = tmp_path / "test.lean"
        test_file.write_text("theorem foo : True := by sorry")

        assert file_contains_sorry(str(test_file))

        test_file.write_text("theorem foo : True := by trivial")
        assert not file_contains_sorry(str(test_file))


class TestErrorParser:
    """Tests for error_parser.py functions."""

    def test_parse_lean_errors_basic(self):
        """Test parsing standard Lean errors."""
        from verifier.error_parser import parse_lean_errors

        output = """/path/to/file.lean:42:5: error: type mismatch
  expected: Nat
  got: Int
"""
        errors = parse_lean_errors(output)

        assert len(errors) == 1
        assert errors[0].file_path == "/path/to/file.lean"
        assert errors[0].line == 42
        assert errors[0].column == 5
        assert errors[0].error_type == "type"
        assert "type mismatch" in errors[0].message

    def test_parse_lean_errors_tactic_failure(self):
        """Test parsing tactic failure errors."""
        from verifier.error_parser import parse_lean_errors

        output = """/test/file.lean:50:10: error: tactic 'simp' failed
  no simplification rules applied
"""
        errors = parse_lean_errors(output)

        assert len(errors) == 1
        assert errors[0].error_type == "tactic"
        assert "tactic" in errors[0].message.lower()

    def test_parse_lean_errors_unknown_identifier(self):
        """Test parsing unknown identifier errors."""
        from verifier.error_parser import parse_lean_errors

        output = """/test/file.lean:30:15: error: unknown identifier 'foo'
"""
        errors = parse_lean_errors(output)

        assert len(errors) == 1
        assert errors[0].error_type == "identifier"

    def test_parse_lean_errors_multiple(self):
        """Test parsing multiple errors."""
        from verifier.error_parser import parse_lean_errors

        output = """/test/a.lean:10:5: error: type mismatch
  expected: Nat
/test/b.lean:20:3: error: tactic 'omega' failed
"""
        errors = parse_lean_errors(output)

        assert len(errors) == 2
        assert errors[0].file_path.endswith("a.lean")
        assert errors[1].file_path.endswith("b.lean")

    def test_classify_error(self):
        """Test error classification."""
        from verifier.error_parser import classify_error

        assert classify_error("type mismatch") == "type"
        assert classify_error("tactic 'simp' failed") == "tactic"
        assert classify_error("unknown identifier 'x'") == "identifier"
        assert classify_error("declaration uses 'sorry'") == "sorry"
        assert classify_error("some other error") == "unknown"

    def test_filter_errors_for_file(self):
        """Test filtering errors by file."""
        from verifier.error_parser import parse_lean_errors, filter_errors_for_file

        output = """/test/a.lean:10:5: error: type mismatch
/test/b.lean:20:3: error: tactic failed
"""
        errors = parse_lean_errors(output)
        filtered = filter_errors_for_file(errors, "a.lean")

        assert len(filtered) == 1
        assert filtered[0].file_path.endswith("a.lean")

    def test_extract_error_summary(self):
        """Test error summary generation."""
        from verifier.error_parser import parse_lean_errors, extract_error_summary

        output = """/test/file.lean:42:5: error: type mismatch
  expected: Nat
  got: Int
"""
        errors = parse_lean_errors(output)
        summary = extract_error_summary(errors)

        assert "Type errors" in summary
        assert "42" in summary

    def test_extract_error_summary_empty(self):
        """Test error summary with no errors."""
        from verifier.error_parser import extract_error_summary

        summary = extract_error_summary([])
        assert "No specific errors" in summary


class TestLakeRunner:
    """Tests for lake_runner.py functions."""

    @pytest.mark.asyncio
    async def test_run_lake_build_missing_dir(self):
        """Test lake build with non-existent directory."""
        from verifier.lake_runner import run_lake_build

        result = await run_lake_build("/nonexistent/path")

        assert not result.success
        assert "does not exist" in result.stderr

    @pytest.mark.asyncio
    async def test_run_lake_build_no_lakefile(self, tmp_path):
        """Test lake build with no lakefile."""
        from verifier.lake_runner import run_lake_build

        result = await run_lake_build(str(tmp_path))

        assert not result.success
        assert "lakefile" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_run_lake_build_mock_success(self, tmp_path):
        """Test lake build with mocked subprocess (success)."""
        from verifier.lake_runner import run_lake_build

        # Create lakefile
        lakefile = tmp_path / "lakefile.lean"
        lakefile.write_text("-- lakefile")

        # Mock the subprocess
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"Build successful", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await run_lake_build(str(tmp_path))

        assert result.success
        assert result.return_code == 0
        assert "successful" in result.stdout.lower()

    @pytest.mark.asyncio
    async def test_run_lake_build_mock_failure(self, tmp_path):
        """Test lake build with mocked subprocess (failure)."""
        from verifier.lake_runner import run_lake_build

        lakefile = tmp_path / "lakefile.lean"
        lakefile.write_text("-- lakefile")

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(
            return_value=(b"", b"error: type mismatch")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await run_lake_build(str(tmp_path))

        assert not result.success
        assert result.return_code == 1

    @pytest.mark.asyncio
    async def test_run_lake_build_timeout(self, tmp_path):
        """Test lake build timeout handling."""
        from verifier.lake_runner import run_lake_build
        import asyncio

        lakefile = tmp_path / "lakefile.lean"
        lakefile.write_text("-- lakefile")

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_proc.kill = AsyncMock()
        mock_proc.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                result = await run_lake_build(str(tmp_path), timeout=1)

        assert not result.success
        assert "timed out" in result.stderr.lower()

    def test_get_module_from_file(self, tmp_path):
        """Test module name derivation from file path."""
        from verifier.lake_runner import get_module_from_file

        project_dir = str(tmp_path)
        file_path = str(tmp_path / "DalekLean" / "Specs" / "SubLoop.lean")

        module = get_module_from_file(file_path, project_dir)
        assert module == "DalekLean.Specs.SubLoop"


class TestRetryHandler:
    """Tests for retry_handler.py functions."""

    @pytest.mark.asyncio
    async def test_verify_proof_success(self, tmp_path):
        """Test successful proof verification."""
        from verifier.retry_handler import verify_proof
        from verifier import BuildResult

        # Create test file
        test_file = tmp_path / "test.lean"
        test_file.write_text(
            """theorem foo : True := by
  sorry
"""
        )

        # Create lakefile
        lakefile = tmp_path / "lakefile.lean"
        lakefile.write_text("-- lakefile")

        sorry = MockSorryLocation(file_path=str(test_file), line=2)
        proof_result = MockProofResult(proof_code="trivial")

        # Mock successful build
        mock_build_result = BuildResult(
            success=True,
            stdout="Build successful",
            stderr="",
            return_code=0,
            elapsed_time=1.0,
        )

        with patch(
            "verifier.retry_handler.run_lake_build",
            return_value=mock_build_result,
        ):
            result = await verify_proof(
                sorry, proof_result, project_dir=str(tmp_path)
            )

        assert result.success
        assert result.attempts == 1
        assert result.proof_code == "trivial"

    @pytest.mark.asyncio
    async def test_verify_proof_retry_on_error(self, tmp_path):
        """Test retry logic when build fails."""
        from verifier.retry_handler import verify_proof
        from verifier import BuildResult

        # Create test file
        test_file = tmp_path / "test.lean"
        original_content = """theorem foo : True := by
  sorry
"""
        test_file.write_text(original_content)

        lakefile = tmp_path / "lakefile.lean"
        lakefile.write_text("-- lakefile")

        sorry = MockSorryLocation(file_path=str(test_file), line=2)
        proof_result = MockProofResult(proof_code="bad_tactic")

        # Mock: first build fails, no retry regeneration
        mock_build_fail = BuildResult(
            success=False,
            stdout="",
            stderr="/test.lean:2:3: error: unknown tactic 'bad_tactic'",
            return_code=1,
            elapsed_time=1.0,
        )

        call_count = [0]

        async def mock_build(*args, **kwargs):
            call_count[0] += 1
            return mock_build_fail

        with patch("verifier.retry_handler.run_lake_build", side_effect=mock_build):
            with patch(
                "verifier.retry_handler._regenerate_with_feedback",
                return_value=None,  # No regeneration
            ):
                result = await verify_proof(
                    sorry,
                    proof_result,
                    max_attempts=2,
                    project_dir=str(tmp_path),
                )

        assert not result.success
        assert result.attempts == 2
        assert len(result.errors) > 0

        # File should be restored
        assert test_file.read_text() == original_content

    @pytest.mark.asyncio
    async def test_verify_proof_restores_on_failure(self, tmp_path):
        """Test that original file is restored on failure."""
        from verifier.retry_handler import verify_proof
        from verifier import BuildResult

        test_file = tmp_path / "test.lean"
        original_content = """theorem foo : True := by
  sorry
"""
        test_file.write_text(original_content)

        lakefile = tmp_path / "lakefile.lean"
        lakefile.write_text("-- lakefile")

        sorry = MockSorryLocation(file_path=str(test_file), line=2)
        proof_result = MockProofResult(proof_code="bad")

        mock_build_fail = BuildResult(
            success=False,
            stdout="",
            stderr="error",
            return_code=1,
            elapsed_time=1.0,
        )

        with patch(
            "verifier.retry_handler.run_lake_build",
            return_value=mock_build_fail,
        ):
            with patch(
                "verifier.retry_handler._regenerate_with_feedback",
                return_value=None,
            ):
                result = await verify_proof(
                    sorry,
                    proof_result,
                    max_attempts=1,
                    project_dir=str(tmp_path),
                )

        assert not result.success
        # Original file should be restored
        assert test_file.read_text() == original_content


class TestDataclasses:
    """Tests for verifier dataclasses."""

    def test_verification_result_defaults(self):
        """Test VerificationResult default values."""
        from verifier import VerificationResult

        result = VerificationResult(
            success=True,
            proof_code="simp",
            attempts=1,
            build_output="ok",
        )

        assert result.errors == []
        assert result.elapsed_time == 0.0

    def test_lean_error_creation(self):
        """Test LeanError creation."""
        from verifier import LeanError

        error = LeanError(
            file_path="/test/file.lean",
            line=42,
            column=5,
            error_type="type",
            message="type mismatch",
        )

        assert error.file_path == "/test/file.lean"
        assert error.line == 42
        assert error.context == ""  # default

    def test_build_result_creation(self):
        """Test BuildResult creation."""
        from verifier import BuildResult

        result = BuildResult(
            success=True,
            stdout="Built successfully",
            stderr="",
            return_code=0,
            elapsed_time=5.5,
        )

        assert result.success
        assert result.elapsed_time == 5.5
