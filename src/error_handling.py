"""
Error handling utilities and custom exceptions.
"""

from typing import Optional, Dict, Any
import traceback


class StrategyAssistantError(Exception):
    """Base exception for all application errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.user_message = user_message or "An error occurred. Please try again."


class EmbeddingError(StrategyAssistantError):
    """Raised when embedding generation fails."""
    pass


class RetrievalError(StrategyAssistantError):
    """Raised when case retrieval fails."""
    pass


class PredictionError(StrategyAssistantError):
    """Raised when prediction fails."""
    pass


class DataLoadError(StrategyAssistantError):
    """Raised when data loading fails."""
    pass


class ValidationError(StrategyAssistantError):
    """Raised when input validation fails."""
    pass


def format_error_for_user(error: Exception) -> str:
    """
    Format an error message for display to the user.

    Parameters
    ----------
    error : Exception
        The error to format

    Returns
    -------
    str
        User-friendly error message
    """
    if isinstance(error, StrategyAssistantError):
        return error.user_message

    # Generic errors - don't expose internal details
    error_messages = {
        FileNotFoundError: "Required data files are missing. Please ensure the application is properly installed.",
        PermissionError: "Permission denied. Please check file access permissions.",
        MemoryError: "Insufficient memory to complete the operation. Try with a smaller query.",
        TimeoutError: "The operation took too long and was cancelled. Please try again.",
    }

    return error_messages.get(type(error), "An unexpected error occurred. Please try again later.")


def log_error_with_context(error: Exception, context: Dict[str, Any], logger) -> None:
    """
    Log an error with additional context information.

    Parameters
    ----------
    error : Exception
        The error to log
    context : dict
        Additional context information
    logger : logging.Logger
        Logger instance
    """
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context,
        "traceback": traceback.format_exc(),
    }

    logger.error(f"Error occurred: {error_info}")


class ErrorHandler:
    """Centralized error handling for the application."""

    def __init__(self, logger):
        self.logger = logger

    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        reraise: bool = False
    ) -> str:
        """
        Handle an error: log it and return a user-friendly message.

        Parameters
        ----------
        error : Exception
            The error to handle
        context : dict, optional
            Additional context
        reraise : bool
            Whether to re-raise the error after handling

        Returns
        -------
        str
            User-friendly error message
        """
        log_error_with_context(error, context or {}, self.logger)
        user_message = format_error_for_user(error)

        if reraise:
            raise

        return user_message
