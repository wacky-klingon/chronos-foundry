"""
Central Configuration Provider with YAML validation and error handling
"""

import yaml
import os
from typing import Any, Dict, List, Optional
from pathlib import Path
import logging


class ConfigValidationError(Exception):
    """Raised when configuration validation fails"""

    pass


class CentralConfigProvider:
    """Central configuration provider with YAML validation and error handling."""

    def __init__(self, config_files: List[str]):
        self.config_files = config_files
        self.config_data = {}
        self.logger = logging.getLogger(__name__)

    def initialize(self) -> None:
        """Initialize configuration provider and validate all YAML files."""
        self.logger.info("Initializing central configuration provider")

        # Load and validate all configuration files
        for config_file in self.config_files:
            self.load_and_validate_config(config_file)

        self.logger.info(
            f"Successfully loaded {len(self.config_files)} configuration files"
        )

    def load_and_validate_config(self, config_file: str) -> None:
        """Load and validate a single YAML configuration file."""
        config_path = Path(config_file)

        if not config_path.exists():
            raise ConfigValidationError(f"Configuration file not found: {config_file}")

        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)

            if config_data is None:
                raise ConfigValidationError(
                    f"Configuration file is empty: {config_file}"
                )

            # Store config data with filename as key
            config_name = config_path.stem
            self.config_data[config_name] = config_data

            self.logger.info(f"Successfully loaded configuration: {config_file}")

        except yaml.YAMLError as e:
            raise ConfigValidationError(f"Invalid YAML in {config_file}: {e}")
        except Exception as e:
            raise ConfigValidationError(f"Error loading {config_file}: {e}")

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key. Throws error if key doesn't exist."""
        if default is None:
            if not self.check_key_exists(key):
                raise ConfigValidationError(f"Configuration key not found: {key}")

        # Parse key (e.g., "train.date_range.start")
        keys = key.split(".")
        current_data = self.config_data

        try:
            for k in keys:
                if isinstance(current_data, dict) and k in current_data:
                    current_data = current_data[k]
                else:
                    if default is not None:
                        return default
                    raise KeyError(f"Key '{k}' not found in path '{key}'")

            return current_data

        except (KeyError, TypeError) as e:
            if default is not None:
                return default
            raise ConfigValidationError(f"Configuration key not found: {key} - {e}")

    def get_config_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        if section not in self.config_data:
            raise ConfigValidationError(f"Configuration section not found: {section}")

        return self.config_data[section]

    def get_merged_config(
        self, include_covariates=True, include_incremental=False, model_path=None
    ):
        """Helper function to get merged configuration for different training types"""
        # Get base configurations
        chronos_config = self.get_config_section("train")["chronos_model"]

        # Always use parquet loader config - fail fast if not available
        parquet_config = self.get_config_section("parquet_loader_config")[
            "parquet_loader"
        ]
        data_config = {
            "timestamp_col": parquet_config["schema"]["datetime_column"],
            "target_col": "target",  # Use primary target column as per documentation
            "item_id_col": "item_id",  # Default item ID column
        }

        full_config = {
            **data_config,
            **chronos_config,
        }

        # Add parquet loader configuration if available
        try:
            parquet_config = self.get_config_section("parquet_loader_config")
            full_config["parquet_loader"] = parquet_config["parquet_loader"]
        except ConfigValidationError:
            pass  # Parquet config not available, continue without it

        # Add covariate configuration if requested
        if include_covariates:
            try:
                covariate_config = self.get_config_section("covariate_config")
                full_config.update(covariate_config)
            except ConfigValidationError:
                # Fallback to basic covariate config
                full_config.update(
                    {
                        "enabled": True,
                        "target_scaling": True,
                        "enable_regressor_analysis": True,
                    }
                )

        # Add incremental training configuration if requested
        if include_incremental:
            try:
                incremental_config = self.get_config_section(
                    "incremental_training_config"
                )
                full_config["incremental_training"] = incremental_config[
                    "incremental_training"
                ]
            except ConfigValidationError:
                # Fallback to basic incremental config
                incremental_config = {
                    "model_versioning": True,
                    "performance_threshold": 0.05,
                    "rollback_enabled": True,
                    "max_versions": 10,
                }
                full_config["incremental_training"] = incremental_config

        # Override model path if provided
        if model_path:
            full_config["model_path"] = model_path

        return full_config

    def check_key_exists(self, key: str) -> bool:
        """Check if configuration key exists."""
        try:
            keys = key.split(".")
            current_data = self.config_data

            for k in keys:
                if isinstance(current_data, dict) and k in current_data:
                    current_data = current_data[k]
                else:
                    return False

            return True

        except (KeyError, TypeError):
            return False

    def list_available_sections(self) -> List[str]:
        """List all available configuration sections."""
        return list(self.config_data.keys())

    def validate_required_keys(self, required_keys: List[str]) -> None:
        """Validate that all required keys exist."""
        missing_keys = []

        for key in required_keys:
            if not self.check_key_exists(key):
                missing_keys.append(key)

        if missing_keys:
            raise ConfigValidationError(
                f"Missing required configuration keys: {missing_keys}"
            )
