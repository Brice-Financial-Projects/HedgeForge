"""
Utility functions for the HedgeForge project.
---------------------------------------------

src/hedge_forge/utils/logger.py
"""

import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler


def setup_logging(
    log_dir: str = "logs",
    log_file: str = "hedgeforge.log",
    level: int = logging.DEBUG,
    max_bytes: int = 5_000_000,  # 5 MB per file
    backup_count: int = 5,  # keep 5 old log files
) -> logging.Logger:
    """
    Configure centralized logging for the HedgeForge project.
    Creates a rotating log file and a console stream.

    Args:
        log_dir (str): Directory to store log files.
        log_file (str): Log file name.
        level (int): Minimum log level for the logger.
        max_bytes (int): Maximum size of a single log file (in bytes).
        backup_count (int): Number of rotated backups to keep.

    Returns:
        logging.Logger: Configured logger instance.
    """

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger("hedgeforge")
    logger.setLevel(level)
    logger.propagate = False  # prevent double logging via root logger

    # Prevent duplicate handlers (important when reimported or in multiprocessing)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define log message format
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # Attach handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Initial log entry (UTC timestamp)
    logger.info(
        f"Logging initialized at {datetime.now(timezone.utc).isoformat()} UTC | Log file: {log_path}"
    )

    return logger
