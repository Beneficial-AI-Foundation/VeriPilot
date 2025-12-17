"""
Tests for attempt logging functionality.

Tests _VPN file copies and cumulative log file generation.
Uses the new naming convention: filename_VP1.lean, filename_VP2.lean, etc.
"""

import json
import pytest
import tempfile
from pathlib import Path

from verifier.file_modifier import (
    AttemptLog,
    create_attempt_copy,
    cleanup_intermediate_attempts,
    cleanup_all_attempt_files,
    write_attempt_log,
    read_attempt_log,
    cleanup_log_file,
)


class TestAttemptLog:
    """Tests for AttemptLog dataclass."""

    def test_create_attempt_log(self):
        """Test creating an AttemptLog entry."""
        log = AttemptLog.create(
            attempt=1,
            proof_code="simp",
            build_success=True,
            errors=[],
            elapsed_time=1.5,
            model_used="gemini",
            temperature=0.3,
        )

        assert log.attempt == 1
        assert log.proof_code == "simp"
        assert log.build_success is True
        assert log.errors == []
        assert log.elapsed_time == 1.5
        assert log.model_used == "gemini"
        assert log.temperature == 0.3
        assert log.timestamp  # Should be set automatically

    def test_create_failed_attempt_log(self):
        """Test creating a failed attempt log."""
        log = AttemptLog.create(
            attempt=2,
            proof_code="grind",
            build_success=False,
            errors=["type mismatch", "unknown identifier"],
            elapsed_time=2.3,
        )

        assert log.attempt == 2
        assert log.build_success is False
        assert len(log.errors) == 2


class TestCreateAttemptCopy:
    """Tests for create_attempt_copy function."""

    def test_create_attempt_copy(self, tmp_path):
        """Test creating a _VPN copy."""
        # Create test file
        test_file = tmp_path / "test.lean"
        test_file.write_text("theorem foo : True := by sorry")

        # Create attempt 1 copy
        copy_path = create_attempt_copy(str(test_file), 1)

        assert Path(copy_path).exists()
        assert Path(copy_path).name == "test_VP1.lean"
        assert Path(copy_path).read_text() == "theorem foo : True := by sorry"

    def test_create_multiple_attempt_copies(self, tmp_path):
        """Test creating multiple attempt copies."""
        test_file = tmp_path / "test.lean"
        test_file.write_text("original content")

        # Create multiple copies
        copy1 = create_attempt_copy(str(test_file), 1)
        copy2 = create_attempt_copy(str(test_file), 2)
        copy3 = create_attempt_copy(str(test_file), 3)

        assert Path(copy1).name == "test_VP1.lean"
        assert Path(copy2).name == "test_VP2.lean"
        assert Path(copy3).name == "test_VP3.lean"

        # All should exist
        assert Path(copy1).exists()
        assert Path(copy2).exists()
        assert Path(copy3).exists()


class TestCleanupIntermediateAttempts:
    """Tests for cleanup_intermediate_attempts function."""

    def test_cleanup_keeps_final_only(self, tmp_path):
        """Test that cleanup keeps only the final attempt."""
        test_file = tmp_path / "test.lean"
        test_file.write_text("content")

        # Create attempts 1-3
        create_attempt_copy(str(test_file), 1)
        create_attempt_copy(str(test_file), 2)
        create_attempt_copy(str(test_file), 3)

        # Cleanup, keeping only attempt 3
        deleted = cleanup_intermediate_attempts(str(test_file), 3)

        # Attempts 1 and 2 should be deleted
        assert len(deleted) == 2
        assert not (tmp_path / "test_VP1.lean").exists()
        assert not (tmp_path / "test_VP2.lean").exists()

        # Attempt 3 should remain
        assert (tmp_path / "test_VP3.lean").exists()

    def test_cleanup_no_files_to_delete(self, tmp_path):
        """Test cleanup when there are no intermediate files."""
        test_file = tmp_path / "test.lean"
        test_file.write_text("content")

        # Create only attempt 1
        create_attempt_copy(str(test_file), 1)

        # Cleanup for final_attempt=1 (nothing to delete)
        deleted = cleanup_intermediate_attempts(str(test_file), 1)

        assert len(deleted) == 0
        assert (tmp_path / "test_VP1.lean").exists()


class TestCleanupAllAttemptFiles:
    """Tests for cleanup_all_attempt_files function."""

    def test_cleanup_all(self, tmp_path):
        """Test cleaning up all attempt files."""
        test_file = tmp_path / "test.lean"
        test_file.write_text("content")

        # Create several attempts
        create_attempt_copy(str(test_file), 1)
        create_attempt_copy(str(test_file), 2)
        create_attempt_copy(str(test_file), 3)

        # Create basic _VP file too (legacy naming)
        (tmp_path / "test_VP.lean").write_text("vp content")

        # Cleanup all
        deleted = cleanup_all_attempt_files(str(test_file))

        # All should be deleted
        assert len(deleted) == 4
        assert not (tmp_path / "test_VP1.lean").exists()
        assert not (tmp_path / "test_VP2.lean").exists()
        assert not (tmp_path / "test_VP3.lean").exists()
        assert not (tmp_path / "test_VP.lean").exists()


class TestWriteAttemptLog:
    """Tests for write_attempt_log function."""

    def test_write_json_log(self, tmp_path):
        """Test writing log in JSON format."""
        test_file = tmp_path / "test.lean"
        test_file.write_text("content")

        logs = [
            AttemptLog.create(
                attempt=1,
                proof_code="simp",
                build_success=False,
                errors=["error 1"],
                elapsed_time=1.0,
            ),
            AttemptLog.create(
                attempt=2,
                proof_code="grind",
                build_success=True,
                errors=[],
                elapsed_time=2.0,
            ),
        ]

        log_path = write_attempt_log(str(test_file), logs, format="json")

        assert Path(log_path).exists()
        assert Path(log_path).name == "VP_log_test.json"

        # Verify content
        with open(log_path, "r") as f:
            data = json.load(f)

        assert len(data) == 2
        assert data[0]["attempt"] == 1
        assert data[0]["build_success"] is False
        assert data[1]["attempt"] == 2
        assert data[1]["build_success"] is True

    def test_write_markdown_log(self, tmp_path):
        """Test writing log in Markdown format."""
        test_file = tmp_path / "test.lean"
        test_file.write_text("content")

        logs = [
            AttemptLog.create(
                attempt=1,
                proof_code="simp",
                build_success=True,
                errors=[],
                elapsed_time=1.0,
                model_used="gemini",
            ),
        ]

        log_path = write_attempt_log(str(test_file), logs, format="md")

        assert Path(log_path).name == "VP_log_test.md"
        content = Path(log_path).read_text()

        assert "# Verification Log: test.lean" in content
        assert "Attempt 1" in content
        assert "SUCCESS" in content
        assert "gemini" in content

    def test_write_txt_log(self, tmp_path):
        """Test writing log in plain text format."""
        test_file = tmp_path / "test.lean"
        test_file.write_text("content")

        logs = [
            AttemptLog.create(
                attempt=1,
                proof_code="simp",
                build_success=False,
                errors=["test error"],
                elapsed_time=1.0,
            ),
        ]

        log_path = write_attempt_log(str(test_file), logs, format="txt")

        assert Path(log_path).name == "VP_log_test.txt"
        content = Path(log_path).read_text()

        assert "Verification Log" in content
        assert "FAILED" in content
        assert "test error" in content


class TestReadAttemptLog:
    """Tests for read_attempt_log function."""

    def test_read_json_log(self, tmp_path):
        """Test reading a JSON log file."""
        test_file = tmp_path / "test.lean"
        test_file.write_text("content")

        original_logs = [
            AttemptLog.create(
                attempt=1,
                proof_code="simp",
                build_success=True,
                errors=[],
                elapsed_time=1.5,
            ),
        ]

        log_path = write_attempt_log(str(test_file), original_logs, format="json")
        read_logs = read_attempt_log(log_path)

        assert len(read_logs) == 1
        assert read_logs[0].attempt == 1
        assert read_logs[0].proof_code == "simp"
        assert read_logs[0].build_success is True


class TestCleanupLogFile:
    """Tests for cleanup_log_file function."""

    def test_cleanup_log_file(self, tmp_path):
        """Test cleaning up a log file."""
        test_file = tmp_path / "test.lean"
        test_file.write_text("content")

        logs = [AttemptLog.create(1, "simp", True, [], 1.0)]
        log_path = write_attempt_log(str(test_file), logs)

        assert Path(log_path).exists()

        result = cleanup_log_file(str(test_file), "json")

        assert result is True
        assert not Path(log_path).exists()

    def test_cleanup_nonexistent_log_file(self, tmp_path):
        """Test cleanup when log file doesn't exist."""
        test_file = tmp_path / "test.lean"
        test_file.write_text("content")

        result = cleanup_log_file(str(test_file), "json")

        assert result is False
