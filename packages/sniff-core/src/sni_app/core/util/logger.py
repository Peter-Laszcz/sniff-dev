"""
SNIFF logging configuration.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOGGER_NAME = "SNIFF_Log"


def log_dir() -> Path:
    """
    Return the folder log files are written to.

    Returns
    -------
    Path
        absolute path to the log folder.
    """
    return Path.home() / ".sniff" / "logs"


def default_log_file() -> Path:
    """Return the file setup_logger logs to when given no path."""
    return log_dir() / "sniff.log"


def setup_logger(log_file=None, max_bytes=1024 * 1024, backup_count=10):
    """
    Return the shared SNIFF_Log logger, configured to write into the per-user
    log folder.

    Parameters
    ----------
    log_file : str or Path, optional
        Path or name for the log file. Defaults to default_log_file.
    max_bytes : int, optional
        Maximum size, in bytes, before the log file is rotated
    backup_count : int, optional
        Number of backup files to keep (default 10).

    Returns
    -------
    logging.Logger
        Shared configured logger
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    if log_file is None:
        log_file = default_log_file()
    log_file = Path(log_file).expanduser()
    if log_file.parent == Path("."):  # bare filename : keep it with the others
        log_file = log_dir() / log_file.name

    log_file.parent.mkdir(parents=True, exist_ok=True)
    target = str(log_file.resolve())

    for handler in logger.handlers:
        if getattr(handler, "_sniff_target", None) == target:
            return logger

    handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler._sniff_target = target
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)

    return logger
