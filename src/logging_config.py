"""
Logging configuration for the application.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Configure application logging.

    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : Path, optional
        Path to log file. If None, only logs to console.
    log_to_console : bool
        Whether to log to console

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger("strategy_assistant")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Parameters
    ----------
    name : str
        Logger name (typically __name__)

    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(f"strategy_assistant.{name}")


class PerformanceLogger:
    """Context manager for logging operation performance."""

    def __init__(self, operation: str, logger: logging.Logger):
        self.operation = operation
        self.logger = logger
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        if exc_type is None:
            self.logger.info(f"Completed: {self.operation} ({duration:.2f}s)")
        else:
            self.logger.error(
                f"Failed: {self.operation} ({duration:.2f}s) - {exc_type.__name__}: {exc_val}"
            )
        return False
