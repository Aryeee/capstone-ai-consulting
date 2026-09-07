"""
Tests for security utilities.
"""

import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.security import (
    sanitize_input,
    validate_query,
    sanitize_filename,
    SecurityError,
)


class TestSecurity:
    def test_sanitize_input_normal(self):
        """Test sanitization of normal input."""
        text = "This is a normal business query about digital transformation"
        result = sanitize_input(text)
        assert result == text

    def test_sanitize_input_removes_null_bytes(self):
        """Test that null bytes are removed."""
        text = "Normal text\x00with null"
        result = sanitize_input(text)
        assert '\x00' not in result

    def test_sanitize_input_strips_whitespace(self):
        """Test excessive whitespace is normalized."""
        text = "Too    much     whitespace"
        result = sanitize_input(text)
        assert "    " not in result

    def test_sanitize_input_max_length(self):
        """Test max length enforcement."""
        long_text = "a" * 10001
        with pytest.raises(SecurityError):
            sanitize_input(long_text, max_length=10000)

    def test_validate_query_normal(self):
        """Test validation of normal query."""
        query = "This is a valid business query"
        assert validate_query(query) is True

    def test_validate_query_empty(self):
        """Test empty query rejection."""
        with pytest.raises(SecurityError):
            validate_query("")

    def test_validate_query_too_short(self):
        """Test short query rejection."""
        with pytest.raises(SecurityError):
            validate_query("short", min_length=10)

    def test_validate_query_too_long(self):
        """Test long query rejection."""
        long_query = "a" * 10001
        with pytest.raises(SecurityError):
            validate_query(long_query, max_length=10000)

    def test_validate_query_suspicious_script(self):
        """Test script tag detection."""
        malicious = "Normal text <script>alert('xss')</script>"
        with pytest.raises(SecurityError):
            validate_query(malicious)

    def test_validate_query_suspicious_javascript(self):
        """Test javascript: protocol detection."""
        malicious = "javascript:alert('xss')"
        with pytest.raises(SecurityError):
            validate_query(malicious)

    def test_sanitize_filename_normal(self):
        """Test normal filename."""
        filename = "report_2024.pdf"
        result = sanitize_filename(filename)
        assert result == filename

    def test_sanitize_filename_path_traversal(self):
        """Test path traversal prevention."""
        filename = "../../../etc/passwd"
        result = sanitize_filename(filename)
        assert '/' not in result
        assert '\\' not in result

    def test_sanitize_filename_special_chars(self):
        """Test special character removal."""
        filename = "file@#$%.txt"
        result = sanitize_filename(filename)
        assert '@' not in result
        assert '#' not in result
        assert '$' not in result
