from __future__ import annotations

import logging
from pathlib import Path


LOGGER_NAME = "paperpilot"
DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = "logs/app.log",
) -> logging.Logger:
    """Configure the PaperPilot logger without duplicating handlers."""

    normalized_level = _normalize_level(level)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(normalized_level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = logging.Formatter(DEFAULT_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(normalized_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(normalized_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the project logger or one of its children."""

    if name is None or not name.strip():
        return logging.getLogger(LOGGER_NAME)

    if name == LOGGER_NAME or name.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(name)

    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def _normalize_level(level: str) -> int:
    if not isinstance(level, str):
        raise TypeError("level must be a string")

    normalized = level.strip().upper()
    valid_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    if normalized not in valid_levels:
        raise ValueError(
            "level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL"
        )

    return valid_levels[normalized]
