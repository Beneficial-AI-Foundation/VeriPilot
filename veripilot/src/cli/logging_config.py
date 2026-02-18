"""
Centralized logging configuration for VeriPilot.

Sets up session and LLM-specific log files with rotation.
Called at CLI startup before any agent code runs.
"""

import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(verbose: bool = False) -> Path:
    """
    Configure centralized logging with rotating file handlers.

    Creates two log files in veripilot/logs/:
    - session_{timestamp}.log: All loggers at DEBUG level
    - llm_{timestamp}.log: Dedicated LLM I/O logger (veripilot.llm_output)

    Console output level is DEBUG if verbose, WARNING otherwise.
    Rich handles normal console output; logging is for diagnostics.

    Args:
        verbose: If True, console shows DEBUG-level output.

    Returns:
        Path to the log directory.
    """
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Session log (captures all loggers at DEBUG) ---
    session_handler = RotatingFileHandler(
        log_dir / f"session_{timestamp}.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
        encoding="utf-8",
    )
    session_handler.setLevel(logging.DEBUG)
    session_handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )

    # --- LLM log (dedicated veripilot.llm_output logger) ---
    llm_handler = RotatingFileHandler(
        log_dir / f"llm_{timestamp}.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    llm_handler.setLevel(logging.DEBUG)
    llm_handler.setFormatter(
        logging.Formatter("%(asctime)s %(message)s")
    )

    # --- Console handler ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
    console_handler.setFormatter(
        logging.Formatter("%(levelname)s %(message)s")
    )

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(session_handler)
    root.addHandler(console_handler)

    # Configure LLM output logger (propagate=True so it also goes to session log)
    llm_logger = logging.getLogger("veripilot.llm_output")
    llm_logger.addHandler(llm_handler)
    llm_logger.propagate = True

    return log_dir
