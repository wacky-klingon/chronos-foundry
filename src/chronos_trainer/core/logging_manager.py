"""
Logging Manager for initialization before CLI logic starts
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional
import os


# Timestamp format includes milliseconds so log lines can be correlated precisely.
_DEFAULT_FORMAT = "%(asctime)s,%(msecs)03d - %(name)s - %(levelname)s - %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _FlushingRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """RotatingFileHandler that flushes the stream after every emitted record.

    On long-running EC2 jobs the OS write buffer can hold unflushed bytes
    when a process is killed or OOM-terminated.  Flushing after every record
    guarantees that ``tail -f chronos.log`` reflects live progress and that
    post-mortem log inspection is complete.
    """

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


class _FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that explicitly flushes after every record.

    Python's default StreamHandler only flushes when the buffer is full or
    the process exits.  Explicit flushing makes ``tail -f`` and CloudWatch
    Logs agents pick up output immediately.
    """

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


class LoggingManager:
    """Manages logging configuration initialization before CLI logic starts."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.loggers: Dict[str, logging.Logger] = {}
        self.initialized = False
        self._setup_log_directory()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default logging configuration."""
        return {
            "level": "INFO",
            "format": _DEFAULT_FORMAT,
            "datefmt": _DEFAULT_DATE_FORMAT,
            "file_logging": True,
            "log_dir": "logs",
            "max_log_size_mb": 100,
            "backup_count": 5,
            "force_flush": True,
        }

    def _setup_log_directory(self) -> None:
        """Create log directory if it doesn't exist."""
        log_dir = Path(self.config["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)

    def initialize_logging(self) -> None:
        """Initialize logging configuration before CLI logic starts."""
        if self.initialized:
            return

        log_level = getattr(logging, self.config["level"].upper())
        force_flush: bool = self.config.get("force_flush", True)

        formatter = logging.Formatter(
            fmt=self.config.get("format", _DEFAULT_FORMAT),
            datefmt=self.config.get("datefmt", _DEFAULT_DATE_FORMAT),
        )

        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.handlers.clear()

        console_cls = _FlushingStreamHandler if force_flush else logging.StreamHandler
        console_handler = console_cls()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        if self.config.get("file_logging", True):
            log_file = Path(self.config["log_dir"]) / "chronos.log"

            file_cls = (
                _FlushingRotatingFileHandler
                if force_flush
                else logging.handlers.RotatingFileHandler
            )
            file_handler = file_cls(
                log_file,
                maxBytes=self.config["max_log_size_mb"] * 1024 * 1024,
                backupCount=self.config["backup_count"],
                encoding="utf-8",
                delay=False,
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        self._create_component_loggers()
        self.initialized = True

        logger = logging.getLogger(__name__)
        logger.info(
            "Logging system initialised — level=%s force_flush=%s log_dir=%s",
            self.config["level"].upper(),
            force_flush,
            self.config["log_dir"],
        )

    def _create_component_loggers(self) -> None:
        """Create loggers for system components."""
        components = [
            "config_provider",
            "data_loader",
            "model_trainer",
            "covariate_trainer",
            "fx_trainer",
            "incremental_trainer",
            "prediction_api",
            "chronos_system",
        ]

        for component in components:
            self.loggers[component] = logging.getLogger(component)

    def get_logger(self, name: str) -> logging.Logger:
        """Get logger instance for specific component."""
        if not self.initialized:
            raise RuntimeError(
                "Logging not initialized. Call initialize_logging() first."
            )

        return self.loggers.get(name, logging.getLogger(name))

    def configure_logging(self, log_config: Dict[str, Any]) -> None:
        """Reconfigure logging with updated settings."""
        self.config.update(log_config)

        if self.initialized:
            self.initialized = False
            self.initialize_logging()
