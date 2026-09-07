"""
Security utilities for input validation and sanitization.
"""

import re
from typing import Optional


class SecurityError(Exception):
    """Raised when a security check fails."""
    pass


def sanitize_input(text: str, max_length: int = 10000) -> str:
    """
    Sanitize user input to prevent injection attacks.

    Parameters
    ----------
    text : str
        User input text
    max_length : int
        Maximum allowed length

    Returns
    -------
    str
        Sanitized text

    Raises
    ------
    SecurityError
        If input fails security checks
    """
    if not isinstance(text, str):
        raise SecurityError("Input must be a string")

    if len(text) > max_length:
        raise SecurityError(f"Input exceeds maximum length of {max_length} characters")

    # Remove null bytes
    text = text.replace('\x00', '')

    # Strip excessive whitespace
    text = ' '.join(text.split())

    return text


def validate_query(query: str, min_length: int = 10, max_length: int = 10000) -> bool:
    """
    Validate that a query meets basic requirements.

    Parameters
    ----------
    query : str
        Query string to validate
    min_length : int
        Minimum required length
    max_length : int
        Maximum allowed length

    Returns
    -------
    bool
        True if valid

    Raises
    ------
    SecurityError
        If validation fails
    """
    if not query or not query.strip():
        raise SecurityError("Query cannot be empty")

    if len(query) < min_length:
        raise SecurityError(f"Query must be at least {min_length} characters")

    if len(query) > max_length:
        raise SecurityError(f"Query must not exceed {max_length} characters")

    # Check for suspicious patterns
    suspicious_patterns = [
        r'<script',
        r'javascript:',
        r'onerror=',
        r'onclick=',
        r'eval\(',
        r'exec\(',
    ]

    for pattern in suspicious_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            raise SecurityError("Query contains potentially malicious content")

    return True


def rate_limit_check(
    user_id: str,
    max_requests: int = 100,
    window_seconds: int = 3600
) -> bool:
    """
    Simple rate limiting check (in-memory, for demo purposes).
    In production, use Redis or similar.

    Parameters
    ----------
    user_id : str
        Identifier for the user
    max_requests : int
        Maximum requests allowed in window
    window_seconds : int
        Time window in seconds

    Returns
    -------
    bool
        True if within rate limit
    """
    # This is a placeholder - implement with Redis in production
    return True


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent directory traversal.

    Parameters
    ----------
    filename : str
        Original filename

    Returns
    -------
    str
        Sanitized filename
    """
    # Remove path separators
    filename = filename.replace('/', '').replace('\\', '')

    # Remove special characters
    filename = re.sub(r'[^\w\s\-\.]', '', filename)

    # Limit length
    if len(filename) > 255:
        filename = filename[:255]

    return filename
