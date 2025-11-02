"""
Logging Manager for initialization before CLI logic starts
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional
import os


class LoggingManager:
    """Manages logging configuration initialization before CLI logic starts."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.loggers = {}
        self.initialized = False
        self._setup_log_directory()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default logging configuration."""
        return {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file_logging": True,
            "log_dir": "logs",
            "max_log_size_mb": 100,
            "backup_count": 5,
        }

    def _setup_log_directory(self) -> None:
        """Create log directory if it doesn't exist."""
        log_dir = Path(self.config["log_dir"])
        log_dir.mkdir(exist_ok=True)

    def initialize_logging(self) -> None:
        """Initialize logging configuration before CLI logic starts."""
        if self.initialized:
            return

        # Configure root logger
        log_level = getattr(logging, self.config["level"].upper())

        # Create formatter
        formatter = logging.Formatter(self.config["format"])

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        # Clear existing handlers
        root_logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File handler if enabled
        if self.config.get("file_logging", True):
            log_file = Path(self.config["log_dir"]) / "chronos.log"

            # Rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config["max_log_size_mb"] * 1024 * 1024,
                backupCount=self.config["backup_count"],
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        # Create component loggers
        self._create_component_loggers()

        self.initialized = True

        # Log initialization success
        logger = logging.getLogger(__name__)
        logger.info("Logging system initialized successfully")

    def _create_component_loggers(self) -> None:
        """Create loggers for system components."""
        components = [
            "config_provider",
            "data_loader",
            "model_trainer",
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
        """Configure logging based on configuration settings."""
        self.config.update(log_config)

        if self.initialized:
            # Re-initialize with new config
            self.initialized = False
            self.initialize_logging()
