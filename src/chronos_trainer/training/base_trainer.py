"""
Base model training components for the Chronos-Bolt system
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Import AutoGluon components
try:
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    from autogluon.timeseries.models import ChronosModel
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"AutoGluon import error: {e}")
    logger.error("Please install autogluon: pip install autogluon")


class TrainingError(Exception):
    """Raised when model training fails"""

    pass


class ChronosTrainer:
    """Basic Chronos model trainer for MVP"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("model_trainer")

        # Force CPU-only training
        self._configure_cpu_training()

        # Extract training configuration - NO DEFAULTS, FAIL FAST
        self.model_name = config["model_name"]
        self.context_length = config["context_length"]
        self.prediction_length = config["prediction_length"]
        self.learning_rate = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.max_epochs = config["max_epochs"]

        self.predictor = None
        self.model_path = None

    def _configure_cpu_training(self) -> None:
        """Configure environment for CPU-only training"""
        try:
            # Set environment variables to force CPU usage
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["AUTOGLUON_DEVICE"] = "cpu"

            # Additional environment variables for CPU-only training
            os.environ["TORCH_DEVICE"] = "cpu"
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

            self.logger.info("Configured environment for CPU-only training")
            self.logger.info("CUDA_VISIBLE_DEVICES set to empty string")
            self.logger.info("AUTOGLUON_DEVICE set to cpu")

        except Exception as e:
            self.logger.warning(f"Failed to configure CPU training environment: {e}")

    def prepare_timeseries_dataframe(self, df: pd.DataFrame) -> TimeSeriesDataFrame:
        """Convert pandas DataFrame to AutoGluon TimeSeriesDataFrame"""
        try:
            self.logger.info("Converting data to TimeSeriesDataFrame format")

            # Ensure we have the required columns
            required_cols = ["item_id", "timestamp", "target"]
            if not all(col in df.columns for col in required_cols):
                # Map columns if they have different names
                column_mapping = {
                    self.config.get("item_id_col", "item_id"): "item_id",
                    self.config.get("timestamp_col", "timestamp"): "timestamp",
                    self.config.get("target_col", "value"): "target",
                }

                df = df.rename(columns=column_mapping)

            # Create TimeSeriesDataFrame
            ts_df = TimeSeriesDataFrame.from_data_frame(
                df, id_column="item_id", timestamp_column="timestamp"
            )

            self.logger.info(f"Created TimeSeriesDataFrame with {len(ts_df)} records")
            return ts_df

        except Exception as e:
            raise TrainingError(f"Failed to create TimeSeriesDataFrame: {e}")

    def train_model(self, data: pd.DataFrame, model_save_path: str = None) -> str:
        """Train Chronos model on the provided data"""
        try:
            self.logger.info("Starting model training")

            # Convert to TimeSeriesDataFrame
            ts_data = self.prepare_timeseries_dataframe(data)

            # Create predictor
            self.predictor = TimeSeriesPredictor(
                prediction_length=self.prediction_length,
                target="target",
                eval_metric="MASE",
                path=model_save_path,
            )

            # Train the model
            self.logger.info(f"Training with {len(ts_data)} records")

            self.predictor.fit(
                ts_data,
                presets="high_quality",  # CPU-compatible preset for large datasets
                hyperparameters={
                    # CPU-compatible models for 20+ years of data
                    "ARIMA": {
                        "order": (2, 1, 2),  # More complex ARIMA
                        "seasonal_order": (1, 1, 1, 12),  # Seasonal patterns
                    },
                    "ETS": {
                        "trend": "add",
                        "seasonal": "add",
                        "seasonal_periods": 12,
                    },
                    "Theta": {
                        "seasonality_mode": "multiplicative",
                    },
                    "AutoETS": {
                        "seasonal": "auto",
                    },
                },
            )

            # Save the model
            self.predictor.save()
            self.model_path = self.predictor.path  # Get actual save location

            self.logger.info(
                f"Model training completed and saved to: {self.model_path}"
            )
            return self.model_path

        except Exception as e:
            raise TrainingError(f"Model training failed: {e}") from e

    def load_model(self, model_path: str) -> None:
        """Load a trained model"""
        try:
            self.logger.info(f"Loading model from: {model_path}")

            self.predictor = TimeSeriesPredictor.load(model_path)
            self.model_path = model_path

            self.logger.info("Model loaded successfully")

        except Exception as e:
            raise TrainingError(f"Failed to load model from {model_path}: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model"""
        if self.predictor is None:
            return {"status": "No model loaded"}

        return {
            "status": "Model loaded",
            "model_path": self.model_path,
            "prediction_length": self.prediction_length,
            "context_length": self.context_length,
            "model_name": self.model_name,
        }
