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

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class TrainingError(Exception):
    """Raised when model training fails"""

    pass


class ChronosTrainer:
    """Basic Chronos model trainer for MVP"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("model_trainer")

        # Resolve and configure compute device
        self.device: str = "cpu"
        self._configure_device()

        # Extract training configuration - NO DEFAULTS, FAIL FAST
        self.model_name = config["model_name"]
        self.context_length = config["context_length"]
        self.prediction_length = config["prediction_length"]
        self.learning_rate = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.max_epochs = config["max_epochs"]

        self.predictor = None
        self.model_path = None

    def _apply_cpu_env(self) -> None:
        """Set environment variables that force all frameworks to CPU."""
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["AUTOGLUON_DEVICE"] = "cpu"
        os.environ["TORCH_DEVICE"] = "cpu"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        self.logger.info("CPU-only environment variables applied")
        self.logger.debug(
            "CUDA_VISIBLE_DEVICES='', AUTOGLUON_DEVICE=cpu, TORCH_DEVICE=cpu"
        )

    def _configure_device(self) -> None:
        """Resolve the compute device from config and configure the environment.

        Supports three values for config key ``device``:
        - ``"cpu"``  (default) — forces all frameworks to CPU.
        - ``"cuda"`` — requires CUDA; raises TrainingError if unavailable.
        - ``"auto"`` — uses CUDA when available, falls back to CPU silently.
        """
        requested: str = self.config.get("device", "cpu")
        self.logger.info("Requested training device: %s", requested)

        if requested in ("cuda", "auto"):
            if not _TORCH_AVAILABLE:
                if requested == "cuda":
                    raise TrainingError(
                        "device='cuda' requested but PyTorch is not installed"
                    )
                self.logger.warning(
                    "PyTorch not available; cannot detect CUDA — falling back to CPU"
                )
                self.device = "cpu"
                self._apply_cpu_env()
                return

            cuda_available = torch.cuda.is_available()
            self.logger.debug("torch.cuda.is_available() = %s", cuda_available)

            if cuda_available:
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                vram_total_gb = props.total_memory / (1024 ** 3)
                compute_capability = f"{props.major}.{props.minor}"

                self.device = "cuda"
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                os.environ["AUTOGLUON_DEVICE"] = "gpu"
                os.environ["TORCH_DEVICE"] = "cuda"

                self.logger.info(
                    "CUDA available — %d device(s) detected", device_count
                )
                self.logger.info(
                    "GPU 0: %s | VRAM: %.1f GB | Compute capability: %s",
                    device_name,
                    vram_total_gb,
                    compute_capability,
                )
                self.logger.info(
                    "AUTOGLUON_DEVICE=gpu, TORCH_DEVICE=cuda"
                )
            else:
                if requested == "cuda":
                    raise TrainingError(
                        "device='cuda' requested but torch.cuda.is_available() is False. "
                        "Verify CUDA drivers and PyTorch CUDA build on this instance."
                    )
                self.logger.warning(
                    "device='auto' requested but CUDA is not available — falling back to CPU"
                )
                self.device = "cpu"
                self._apply_cpu_env()
        else:
            self.device = "cpu"
            self._apply_cpu_env()

        self.logger.info("Training device resolved to: %s", self.device)

    def preprocess_raw_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Hook for subclass-level feature engineering on the raw parquet DataFrame.

        Called by the incremental training loop after each parquet file is loaded
        and before the DataFrame is converted to a TimeSeriesDataFrame. The base
        implementation is a no-op and returns the input unchanged.

        Subclasses must override this method to inject domain-specific feature
        engineering. The contract is:

        - Accept the raw DataFrame as loaded from parquet.
        - Return a DataFrame with the same index and at least the same columns.
        - Never raise from within the override; log a WARNING and return the
          input unchanged if prerequisite columns are absent.
        - Do not perform I/O or modify any shared state.

        Args:
            df: Raw pandas DataFrame as loaded by ResumableDataLoader.

        Returns:
            The input DataFrame, optionally enriched with additional columns.
        """
        return df

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

            self.logger.info(
                "Fitting predictor on %d records using device=%s", len(ts_data), self.device
            )
            self.predictor.fit(
                ts_data,
                presets="high_quality",
                hyperparameters={
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
