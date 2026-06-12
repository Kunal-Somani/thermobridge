"""Stdlib logging configuration for ThermoBridge.

Provides a consistently formatted logger across the entire project.
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Track whether root has been configured to avoid duplicate handlers
_root_configured = False


def _configure_root_logger(level: int = logging.INFO) -> None:
    """Configure the root logger once with a stream handler to stdout."""
    global _root_configured  # noqa: PLW0603
    if _root_configured:
        return

    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)

    _root_configured = True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a named logger with consistent formatting.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Logging level (default: INFO).

    Returns:
        Configured stdlib Logger instance.
    """
    _configure_root_logger(level)
    return logging.getLogger(name)
