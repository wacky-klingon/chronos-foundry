"""
Configuration helper utilities for consistent config access across the system
"""

from typing import Dict, Any, Optional
import logging


class ConfigHelpers:
    """Helper class for consistent configuration access patterns"""

    @staticmethod
    def get_parquet_root_dir(config: Dict[str, Any]) -> str:
        """
        Get parquet data root directory from config with fail-fast validation

        Args:
            config: Full configuration dictionary

        Returns:
            Root directory path

        Raises:
            ValueError: If root_dir is not configured
        """
        parquet_config = config.get("parquet_loader", {})
        root_dir = parquet_config.get("data_paths", {}).get("root_dir")

        if not root_dir:
            raise ValueError(
                "Parquet data root_dir not configured in parquet_loader_config.yaml"
            )

        return root_dir

    @staticmethod
    def get_checkpoint_dir(config: Dict[str, Any]) -> str:
        """
        Get checkpoint directory from config with fail-fast validation

        Args:
            config: Full configuration dictionary

        Returns:
            Checkpoint directory path

        Raises:
            ValueError: If checkpoint_dir is not configured
        """
        incremental_config = config.get("incremental_training", {})
        checkpoint_dir = incremental_config.get("checkpoint_dir")

        if not checkpoint_dir:
            raise ValueError(
                "Checkpoint directory not configured in incremental_training_config.yaml"
            )

        return checkpoint_dir

    @staticmethod
    def get_column_mapping(config: Dict[str, Any]) -> Dict[str, str]:
        """
        Get column mapping from config

        Args:
            config: Full configuration dictionary

        Returns:
            Dictionary mapping config column names to standard names
        """
        return {
            config.get("item_id_col", "item_id"): "item_id",
            config.get("timestamp_col", "timestamp"): "timestamp",
            config.get("target_col", "target"): "target",
        }

    @staticmethod
    def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get model configuration parameters

        Args:
            config: Full configuration dictionary

        Returns:
            Dictionary with model configuration
        """
        return {
            "model_name": config.get("model_name", "amazon/chronos-t5-tiny"),
            "context_length": config.get("context_length", 512),
            "prediction_length": config.get("prediction_length", 64),
            "learning_rate": config.get("learning_rate", 0.0001),
            "batch_size": config.get("batch_size", 32),
            "max_epochs": config.get("max_epochs", 2),
            "warmup_steps": config.get("warmup_steps", 100),
            "weight_decay": config.get("weight_decay", 0.01),
            "max_grad_norm": config.get("max_grad_norm", 1.0),
            "d_model": config.get("d_model", 512),
            "num_heads": config.get("num_heads", 8),
            "num_layers": config.get("num_layers", 6),
            "dropout": config.get("dropout", 0.1),
        }
