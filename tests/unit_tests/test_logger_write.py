"""tests/unit_tests/test_logger_write.py"""

import pytest
from src.hedge_forge.utils.logger import setup_logging


@pytest.fixture
def log_setup(tmp_path):
    """
    Pytest fixture to create a temporary logs directory for testing.
    Returns the logger instance and path to the test log file.
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = "hedgeforge.log"
    log_path = log_dir / log_file

    logger = setup_logging(log_dir=str(log_dir), log_file=log_file)
    return logger, log_path


def test_log_file_write(log_setup):
    """
    ✅ Confirms that logs are written correctly to /logs/hedgeforge.log.
    """
    logger, log_path = log_setup
    test_message = "🧪 Test log entry for verification"

    # Write the log entry
    logger.info(test_message)

    # Flush to ensure the message is written to disk
    for handler in logger.handlers:
        handler.flush()

    # Verify the file exists
    assert log_path.exists(), f"Log file not found at {log_path}"

    # Verify contents include our message
    contents = log_path.read_text(encoding="utf-8")
    assert test_message in contents, "Test log entry not found in log file"


def test_multiple_log_entries(log_setup):
    """
    Ensures multiple log messages append correctly to the same file.
    """
    logger, log_path = log_setup
    messages = [f"Message {i}" for i in range(3)]

    for msg in messages:
        logger.info(msg)

    for handler in logger.handlers:
        handler.flush()

    contents = log_path.read_text(encoding="utf-8")
    for msg in messages:
        assert msg in contents, f"Missing log message: {msg}"
