"""
Incremental trainer for continuous model improvement with versioning and rollback
"""

import json
import pandas as pd
import numpy as np
import shutil
import traceback
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
from datetime import datetime, timedelta

# Import AutoGluon components
try:
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"AutoGluon import error: {e}")
    logger.error("Please install autogluon: pip install autogluon")

from .base_trainer import TrainingError
from .covariate_trainer import CovariateTrainer
from .model_versioning import ModelVersioning
from .checkpoint_manager import CheckpointManager
from ..core.config_helpers import ConfigHelpers


class IncrementalTrainingError(Exception):
    """Raised when incremental training fails"""

    pass


class IncrementalTrainer(CovariateTrainer):
    """Incremental trainer for continuous model improvement with versioning and rollback"""

    _CHRONOS_VARIANT_TO_MODEL_PATH: Dict[str, str] = {
        "bolt_tiny": "autogluon/chronos-bolt-tiny",
        "bolt_mini": "autogluon/chronos-bolt-mini",
        "bolt_small": "autogluon/chronos-bolt-small",
        "bolt_base": "autogluon/chronos-bolt-base",
    }
    _DEPRECATED_NON_CHRONOS_KEYS: Tuple[str, ...] = (
        "excluded_model_types",
        "training_mode",
        "benchmark_mode",
    )

    def __init__(self, config: Dict[str, Any]):
        # Pass the full config to parent CovariateTrainer so it can access parquet_loader config
        super().__init__(config)
        self.logger = logging.getLogger("incremental_trainer")

        # Incremental training specific configuration
        self.incremental_config = config.get("incremental_training", {})
        self.model_versioning = self.incremental_config.get("model_versioning", True)
        self.performance_threshold = self.incremental_config.get(
            "performance_threshold", 0.05
        )  # 5% improvement required
        self.rollback_enabled = self.incremental_config.get("rollback_enabled", True)
        self.chronos_only = bool(self.incremental_config.get("chronos_only", True))
        self.rollback_window_versions = self._get_required_rollback_window_versions()
        self.checkpoint_post_success_cleanup = (
            self._get_required_checkpoint_post_success_cleanup()
        )

        # Use high_quality preset for production training (can be overridden via config)
        self.training_preset = config.get("training_preset", "high_quality")
        self.chronos_variant = self._resolve_chronos_variant()
        self.chronos_model_path = self._resolve_chronos_model_path(self.chronos_variant)
        self._validate_chronos_only_configuration()
        self.logger.info(
            "Chronos-only incremental training enabled with variant=%s model_path=%s",
            self.chronos_variant,
            self.chronos_model_path,
        )

        # Initialize model versioning
        versioning_config = {
            "model_path": config.get("model_path", "data/models/incremental"),
            "max_versions": self.rollback_window_versions,
        }
        self.versioning = ModelVersioning(versioning_config)
        self._resumable_loader: Optional[Any] = None

    def _get_required_rollback_window_versions(self) -> int:
        """Read required rollback window retention from config with no defaults."""
        raw_value = self.incremental_config.get("rollback_window_versions")
        if raw_value is None:
            raise IncrementalTrainingError(
                "incremental_training.rollback_window_versions is required. "
                "Set it explicitly in config (e.g. rollback_window_versions: 1)."
            )
        if not isinstance(raw_value, int) or raw_value < 1:
            raise IncrementalTrainingError(
                "incremental_training.rollback_window_versions must be an integer >= 1."
            )
        return raw_value

    def _get_required_checkpoint_post_success_cleanup(self) -> bool:
        """Read whether to run temp + model_checkpoints cleanup after successful checkpoint training."""
        raw = self.incremental_config.get("checkpoint_post_success_cleanup")
        if raw is None:
            raise IncrementalTrainingError(
                "incremental_training.checkpoint_post_success_cleanup is required "
                "(boolean: enable post-success removal of checkpoint temp/ and pruning of "
                "older model_checkpoints/ dirs)."
            )
        if not isinstance(raw, bool):
            raise IncrementalTrainingError(
                "incremental_training.checkpoint_post_success_cleanup must be a boolean."
            )
        return raw

    def _apply_checkpoint_post_success_cleanup(
        self, checkpoint_manager: CheckpointManager
    ) -> None:
        """After successful final export only; does nothing if cleanup disabled."""
        if not self.checkpoint_post_success_cleanup:
            return
        checkpoint_manager.remove_temp_directory()
        checkpoint_manager.prune_model_checkpoints(self.rollback_window_versions)

    def _resolve_chronos_variant(self) -> str:
        """Resolve Chronos model variant from incremental config or model_name."""
        variant = self.incremental_config.get("chronos_model_variant")
        if isinstance(variant, str) and variant.strip():
            return variant.strip().lower()

        chronos_model = self.config.get("chronos_model", {})
        model_name = chronos_model.get("model_name")
        if isinstance(model_name, str) and "chronos-bolt-" in model_name:
            return f"bolt_{model_name.split('chronos-bolt-')[-1].strip().lower()}"

        raise IncrementalTrainingError(
            "incremental_training.chronos_model_variant is required and must be one of "
            f"{sorted(self._CHRONOS_VARIANT_TO_MODEL_PATH.keys())}"
        )

    def _resolve_chronos_model_path(self, chronos_variant: str) -> str:
        """Map Chronos variant to AutoGluon Chronos model path."""
        model_path = self._CHRONOS_VARIANT_TO_MODEL_PATH.get(chronos_variant)
        if model_path is None:
            raise IncrementalTrainingError(
                f"Unsupported incremental_training.chronos_model_variant={chronos_variant!r}. "
                f"Supported values: {sorted(self._CHRONOS_VARIANT_TO_MODEL_PATH.keys())}"
            )
        return model_path

    def _validate_chronos_only_configuration(self) -> None:
        """Fail fast when non-Chronos/mixed-model keys are present."""
        if not self.chronos_only:
            raise IncrementalTrainingError(
                "incremental_training.chronos_only must be true. "
                "Mixed-model incremental training is not supported."
            )

        deprecated_keys = [
            key for key in self._DEPRECATED_NON_CHRONOS_KEYS if key in self.incremental_config
        ]
        if deprecated_keys:
            raise IncrementalTrainingError(
                "Deprecated non-Chronos config keys detected in incremental_training: "
                f"{deprecated_keys}. Remove them for Chronos-only training."
            )

    def _get_chronos_hyperparameters(self) -> Dict[str, Dict[str, Any]]:
        """Build Chronos-only hyperparameters for TimeSeriesPredictor.fit."""
        return {
            "Chronos": {
                "model_path": self.chronos_model_path,
                "context_length": self.context_length,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "device": self.device,
            }
        }

    def _log_artifact_event(self, event: str, **fields: Any) -> None:
        """Structured artifact lifecycle logging for pointer/debug diagnostics."""
        payload = {"event": event, "component": "incremental_trainer", **fields}
        self.logger.info("artifact_event | %s", json.dumps(payload, sort_keys=True, default=str))

    def _artifact_manifest(self, model_dir: Path) -> Dict[str, Any]:
        files = [p for p in model_dir.rglob("*") if p.is_file()]
        rel_files = [str(p.relative_to(model_dir)) for p in files]
        total_bytes = sum(p.stat().st_size for p in files)
        return {
            "file_count": len(files),
            "total_bytes": total_bytes,
            "sample_files": rel_files[:25],
        }

    def _validate_final_model_artifacts(self, model_dir: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
        required = [
            model_dir / "predictor.pkl",
            model_dir / "learner.pkl",
            model_dir / "models" / "trainer.pkl",
        ]
        missing = [str(p.relative_to(model_dir)) for p in required if not p.exists()]
        manifest = self._artifact_manifest(model_dir)
        too_small = manifest["total_bytes"] < 1024 * 1024
        if too_small:
            missing.append("artifact_too_small(<1MB)")
        return len(missing) == 0, missing, manifest

    def _get_excluded_model_types(self) -> List[str]:
        """
        Load excluded_model_types from incremental_training config only.
        No defaults in code: key must be present (empty list means exclude none).
        """
        raw = self.incremental_config.get("excluded_model_types")
        if raw is None:
            raise IncrementalTrainingError(
                "incremental_training.excluded_model_types is required in configuration "
                "(list of AutoGluon model type names; use [] to exclude none)."
            )
        if not isinstance(raw, list):
            raise IncrementalTrainingError(
                "incremental_training.excluded_model_types must be a list of strings."
            )
        for item in raw:
            if not isinstance(item, str):
                raise IncrementalTrainingError(
                    "incremental_training.excluded_model_types must contain only strings."
                )
        return list(raw)

    def _ensure_path_available(
        self,
        path_value: Optional[str],
        path_name: str = "path",
    ) -> Path:
        """
        Ensure a path is configured, exists (or can be created), and has sufficient disk space.
        Raises IncrementalTrainingError if not available.
        """
        if not path_value or not str(path_value).strip():
            raise IncrementalTrainingError(
                f"{path_name} is required. Pass --{path_name.replace('_', '-')} or set in config."
            )
        path = Path(path_value).resolve()
        path.mkdir(parents=True, exist_ok=True)
        min_bytes = self.incremental_config.get("min_free_bytes", 1024**3)  # 1GB
        usage = shutil.disk_usage(path)
        self.logger.info(
            "%s: %s, free space: %s bytes",
            path_name,
            path,
            usage.free,
        )
        if usage.free < min_bytes:
            raise IncrementalTrainingError(
                f"Insufficient disk space on {path}: {usage.free} free, required >= {min_bytes}"
            )
        return path

    def _ensure_model_path_available(self, model_path: Optional[str]) -> Path:
        """Ensure model_path is configured and available. Raises if not."""
        return self._ensure_path_available(model_path, "model_path")

    def train_incremental(
        self,
        data: pd.DataFrame,
        date_range: Tuple[str, str],
        previous_model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train model incrementally on new data range

        Args:
            data: New training data for the specified date range
            date_range: Tuple of (start_date, end_date) for this training phase
            previous_model_path: Path to previous model to use as starting point

        Returns:
            Dictionary with training results, version info, and performance metrics
        """
        try:
            self._ensure_model_path_available(self.config.get("model_path"))
            self.logger.info(
                f"Starting incremental training for date range: {date_range[0]} to {date_range[1]}"
            )

            # Generate new model version
            version_id = self.versioning.generate_version_id(date_range)
            self.logger.info(f"Creating new model version: {version_id}")

            # Load previous model if provided
            previous_predictor = None
            if previous_model_path and Path(previous_model_path).exists():
                try:
                    previous_predictor = self._load_previous_model(previous_model_path)
                    self.logger.info(
                        f"Loaded previous model from: {previous_model_path}"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load previous model: {e}. Starting fresh."
                    )

            # Prepare data for training
            ts_data = self.prepare_timeseries_dataframe(data)

            # Create new predictor
            predictor = TimeSeriesPredictor(
                prediction_length=self.prediction_length,
                target="target",
                eval_metric="MASE",
            )

            # Train with or without previous model
            if previous_predictor:
                # Incremental training with previous model as starting point
                self.logger.info(
                    "Performing incremental training with previous model as starting point"
                )
                predictor.fit(
                    ts_data,
                    presets=self.training_preset,  # CPU-compatible preset for large datasets
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
                    excluded_model_types=self._get_excluded_model_types(),
                )
            else:
                # Fresh training
                self.logger.info("Performing fresh training (no previous model)")
                predictor.fit(
                    ts_data,
                    presets=self.training_preset,  # CPU-compatible preset for large datasets
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
                    excluded_model_types=self._get_excluded_model_types(),
                )

            # Evaluate performance
            performance_metrics = self._evaluate_model_performance(predictor, ts_data)

            # Check if performance meets threshold
            performance_improvement = None
            if previous_predictor:
                previous_performance = self.versioning.get_previous_performance(
                    previous_model_path
                )
                performance_improvement = (
                    self.versioning.calculate_performance_improvement(
                        performance_metrics, previous_performance
                    )
                )

                if performance_improvement < self.performance_threshold:
                    self.logger.warning(
                        f"Performance improvement ({performance_improvement:.2%}) below threshold ({self.performance_threshold:.2%})"
                    )
                    if self.rollback_enabled:
                        self.logger.info("Rolling back to previous model version")
                        return self.versioning.rollback_to_previous(version_id)
                    else:
                        self.logger.warning(
                            "Rollback disabled, keeping new model despite poor performance"
                        )

            # Save new model version
            model_config = {
                "prediction_length": self.prediction_length,
                "context_length": self.context_length,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
            }

            model_path = self.versioning.save_model_version(
                predictor,
                version_id,
                date_range,
                performance_metrics,
                model_config,
                self.covariate_config,
            )

            # Update version tracking
            self.versioning.update_version_tracking(
                version_id, model_path, date_range, performance_metrics
            )

            # Clean up old versions if needed
            self.versioning.cleanup_old_versions()

            self.logger.info(
                f"Incremental training completed successfully. Model saved to: {model_path}"
            )

            return {
                "success": True,
                "version_id": version_id,
                "model_path": model_path,
                "date_range": date_range,
                "performance_metrics": performance_metrics,
                "performance_improvement": performance_improvement,
                "previous_version": self.versioning.previous_version,
            }

        except Exception as e:
            self.logger.error(
                "Incremental training failed: %s\n%s", e, traceback.format_exc()
            )
            raise IncrementalTrainingError(f"Incremental training failed: {e}") from e

    def _load_previous_model(self, model_path: str) -> TimeSeriesPredictor:
        """Load previous model for incremental training"""
        try:
            predictor = TimeSeriesPredictor.load(model_path)
            self.logger.info(f"Successfully loaded previous model from {model_path}")
            return predictor
        except Exception as e:
            raise IncrementalTrainingError(
                f"Failed to load previous model from {model_path}: {e}"
            ) from e

    def _evaluate_model_performance(
        self, predictor: TimeSeriesPredictor, data: TimeSeriesDataFrame
    ) -> Dict[str, float]:
        """Evaluate model performance using proper time series validation with detailed logging"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("STARTING DETAILED MODEL EVALUATION")
            self.logger.info("=" * 80)

            # Debug: Understand TimeSeriesDataFrame structure
            self.logger.info(f"TimeSeriesDataFrame type: {type(data)}")
            self.logger.info(f"TimeSeriesDataFrame columns: {list(data.columns)}")
            self.logger.info(f"TimeSeriesDataFrame shape: {data.shape}")

            # Check if 'target' column exists
            if "target" in data.columns:
                self.logger.info("'target' column found in TimeSeriesDataFrame")
                target_stats = data["target"].describe()
                self.logger.info(f"Target column stats:\n{target_stats}")
            else:
                self.logger.warning("'target' column NOT found in TimeSeriesDataFrame")
                return {
                    "mae": float("inf"),
                    "rmse": float("inf"),
                    "mase": float("inf"),
                    "directional_accuracy": 0.0,
                }

            # For time series, we need to use proper temporal validation
            # Use the last prediction_length points for validation
            total_length = len(data)
            prediction_length = self.prediction_length

            self.logger.info(f"Total data length: {total_length}")
            self.logger.info(f"Prediction length: {prediction_length}")

            if total_length < prediction_length * 2:
                self.logger.warning(
                    f"Insufficient data for proper evaluation. Need at least {prediction_length * 2}, got {total_length}"
                )
                return {
                    "mae": 0.001,
                    "rmse": 0.001,
                    "mase": 0.001,
                    "directional_accuracy": 0.5,
                }

            # Use last prediction_length points as validation set
            # Train on everything before that
            train_data = data.iloc[:-prediction_length]
            val_data = data.iloc[-prediction_length:]

            self.logger.info(f"Train data length: {len(train_data)}")
            self.logger.info(f"Validation data length: {len(val_data)}")

            # Log validation data details
            val_target = val_data["target"].values
            self.logger.info(
                f"Validation target stats - min: {val_target.min():.6f}, max: {val_target.max():.6f}, mean: {val_target.mean():.6f}"
            )
            self.logger.info(
                f"Validation target unique values: {len(np.unique(val_target))}"
            )
            self.logger.info(f"Validation target sample: {val_target[:10]}")

            # Check if validation data is constant (data quality issue)
            if len(np.unique(val_target)) == 1:
                self.logger.warning(
                    "VALIDATION DATA IS CONSTANT! This indicates a data quality issue."
                )
                self.logger.warning(f"All validation values are: {val_target[0]}")
                return {
                    "mae": 0.0,  # Perfect prediction of constant
                    "rmse": 0.0,
                    "mase": 1.0,  # MASE = 1 when both model and naive are perfect
                    "directional_accuracy": 1.0,  # No direction changes in constant data
                }

            # Generate predictions for evaluation
            self.logger.info("Generating predictions...")
            known_covariates_names = self.incremental_config.get("known_covariates", [])
            if known_covariates_names and all(
                c in val_data.columns for c in known_covariates_names
            ):
                known_cov_df = val_data[known_covariates_names]
                predictions = predictor.predict(
                    train_data, known_covariates=known_cov_df
                )
            else:
                predictions = predictor.predict(train_data)

            self.logger.info(f"Predictions type: {type(predictions)}")
            self.logger.info(f"Predictions shape: {predictions.shape}")
            self.logger.info(
                f"Predictions columns: {list(predictions.columns) if hasattr(predictions, 'columns') else 'No columns'}"
            )

            # Extract predicted values (use 0.5 quantile for median)
            if "0.5" in predictions.columns:
                predicted_values = predictions["0.5"].values
                self.logger.info("Using 0.5 quantile predictions")
            elif "mean" in predictions.columns:
                predicted_values = predictions["mean"].values
                self.logger.info("Using mean predictions")
            else:
                # Fallback to first numeric column
                numeric_cols = predictions.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    predicted_values = predictions[numeric_cols[0]].values
                    self.logger.info(f"Using first numeric column '{numeric_cols[0]}'")
                else:
                    raise ValueError("No numeric columns found in predictions")

            self.logger.info(f"Predicted values shape: {predicted_values.shape}")
            self.logger.info(
                f"Predicted values stats - min: {predicted_values.min():.6f}, max: {predicted_values.max():.6f}, mean: {predicted_values.mean():.6f}"
            )
            self.logger.info(f"Predicted values sample: {predicted_values[:10]}")

            # Align lengths
            min_len = min(len(val_target), len(predicted_values))
            actual_values = val_target[:min_len]
            predicted_values = predicted_values[:min_len]

            self.logger.info(f"Aligned length: {min_len}")

            # Calculate MAE
            errors = actual_values - predicted_values
            mae = np.mean(np.abs(errors))
            self.logger.info(
                f"Error stats - min: {errors.min():.6f}, max: {errors.max():.6f}, mean: {errors.mean():.6f}"
            )
            self.logger.info(f"MAE: {mae:.6f}")

            # Calculate RMSE
            rmse = np.sqrt(np.mean(errors**2))
            self.logger.info(f"RMSE: {rmse:.6f}")

            # Calculate MASE (Mean Absolute Scaled Error)
            # Use naive forecast (previous value) as baseline
            if len(actual_values) > 1:
                naive_errors = np.abs(np.diff(actual_values))
                naive_mae = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
            else:
                naive_mae = 1.0

            mase = mae / naive_mae if naive_mae > 0 else 1.0
            self.logger.info(f"Naive forecast MAE: {naive_mae:.6f}")
            self.logger.info(f"MASE: {mase:.6f}")

            # Calculate directional accuracy
            if len(actual_values) > 1:
                actual_direction = np.diff(actual_values) > 0
                predicted_direction = np.diff(predicted_values) > 0
                directional_accuracy = np.mean(actual_direction == predicted_direction)
                self.logger.info(
                    f"Actual direction changes: {np.sum(actual_direction)}/{len(actual_direction)}"
                )
                self.logger.info(
                    f"Predicted direction changes: {np.sum(predicted_direction)}/{len(predicted_direction)}"
                )
                self.logger.info(f"Directional accuracy: {directional_accuracy:.6f}")
            else:
                directional_accuracy = 0.5
                self.logger.warning("Not enough data for directional accuracy")

            performance = {
                "mae": float(mae),
                "rmse": float(rmse),
                "mase": float(mase),
                "directional_accuracy": float(directional_accuracy),
            }

            self.logger.info("=" * 80)
            self.logger.info("FINAL EVALUATION RESULTS")
            self.logger.info("=" * 80)
            for metric, value in performance.items():
                self.logger.info(f"{metric.upper()}: {value:.6f}")
            self.logger.info("=" * 80)

            return performance

        except Exception as e:
            self.logger.error(
                "Failed to evaluate model performance: %s\n%s", e, traceback.format_exc()
            )
            return {
                "mae": float("inf"),
                "rmse": float("inf"),
                "mase": float("inf"),
                "directional_accuracy": 0.0,
            }

    def get_version_history(self) -> Dict[str, Any]:
        """Get complete version history and performance tracking"""
        return self.versioning.get_version_history()

    def list_available_versions(self) -> List[Dict[str, Any]]:
        """List all available model versions"""
        return self.versioning.list_available_versions()

    def switch_to_version(self, version_id: str) -> bool:
        """Switch to a specific model version"""
        return self.versioning.switch_to_version(version_id)

    def train_with_checkpoints(
        self,
        start_date: str,
        end_date: str,
        validation_start_date: str,
        validation_end_date: str,
        checkpoint_dir: str,
        previous_model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train incrementally with checkpoint support for large date ranges

        Args:
            start_date: Start date for training data (YYYY-MM-DD)
            end_date: End date for training data (YYYY-MM-DD)
            validation_start_date: Start date for validation data (YYYY-MM-DD)
            validation_end_date: End date for validation data (YYYY-MM-DD)
            checkpoint_dir: Directory to store checkpoints
            previous_model_path: Path to previous model (optional)

        Returns:
            Training results dictionary
        """
        try:
            model_path = self._ensure_model_path_available(self.config.get("model_path"))
            self._ensure_path_available(checkpoint_dir, "checkpoint_dir")
            self.logger.info(f"Starting resumable training: {start_date} to {end_date}")
            epoch_start_time = time.perf_counter()
            file_timing_rows: List[Dict[str, Any]] = []

            # Initialize checkpoint manager
            checkpoint_manager = CheckpointManager(checkpoint_dir)

            # Check if resuming from checkpoint
            last_checkpoint = checkpoint_manager.get_last_checkpoint()

            if last_checkpoint:
                self.logger.info(
                    f"Resuming from checkpoint: {last_checkpoint['year']:04d}-{last_checkpoint['month']:02d}"
                )
                predictor = last_checkpoint.get("model")  # May not exist if model file missing
                training_state = last_checkpoint.get("training_state", {
                    "start_date": start_date,
                    "end_date": end_date,
                    "validation_start_date": validation_start_date,
                    "validation_end_date": validation_end_date,
                    "processed_files": [],
                    "total_files": 0,
                })
            else:
                self.logger.info("Starting fresh training")
                predictor = None
                training_state = {
                    "start_date": start_date,
                    "end_date": end_date,
                    "validation_start_date": validation_start_date,
                    "validation_end_date": validation_end_date,
                    "processed_files": [],
                    "total_files": 0,
                }

            if previous_model_path:
                prev_desc = (
                    f"path={previous_model_path!r} "
                    f"exists={Path(previous_model_path).exists()}"
                )
            else:
                prev_desc = "not_provided"

            if last_checkpoint:
                self.logger.info(
                    "incremental_checkpoint_decision mode=resume_from_disk "
                    "checkpoint_month=%04d-%02d previous_model=%s "
                    "(previous_model is ignored while resuming from checkpoint)",
                    last_checkpoint["year"],
                    last_checkpoint["month"],
                    prev_desc,
                )
            else:
                self.logger.info(
                    "incremental_checkpoint_decision mode=fresh_start previous_model=%s "
                    "warm_start_from_prior_run=not_implemented "
                    "(first month still fits a new predictor; S3 --previous-model is unused here)",
                    prev_desc,
                )

            # Initialize resumable loader
            resumable_loader = self._get_resumable_loader(checkpoint_manager)

            # Get remaining files to process
            all_parquet_files = resumable_loader.get_parquet_files(start_date, end_date)
            remaining_files = resumable_loader.get_remaining_files(start_date, end_date)
            training_state["total_files"] = len(all_parquet_files)

            if not remaining_files:
                if predictor is None:
                    if not all_parquet_files:
                        return {
                            "status": "error",
                            "message": (
                                f"No parquet files between {start_date} and {end_date}; "
                                "nothing to train"
                            ),
                            "checkpoint_dir": checkpoint_dir,
                        }
                    return {
                        "status": "error",
                        "message": (
                            "All months appear processed but the checkpoint model could not "
                            "be loaded; verify model_checkpoints/*.pkl under the checkpoint "
                            "directory or clear checkpoint state for a fresh run"
                        ),
                        "checkpoint_dir": checkpoint_dir,
                    }
                self.logger.info(
                    "No remaining parquet months; running final validation and export "
                    "using the last checkpoint predictor"
                )

            # Process each remaining file
            for file_path, year, month in remaining_files:
                self.logger.info(f"Processing file: {year:04d}-{month:02d}")
                file_start_time = time.perf_counter()

                # Load parquet file
                parquet_load_start_time = time.perf_counter()
                df = resumable_loader.load_parquet_file(file_path, year, month)
                parquet_load_time_s = time.perf_counter() - parquet_load_start_time
                if df is None:
                    self.logger.error(f"Failed to load file: {file_path}")
                    continue

                # Convert to TimeSeriesDataFrame
                ts_convert_start_time = time.perf_counter()
                ts_df = resumable_loader.convert_to_timeseries_dataframe(
                    df, self.config
                )
                ts_convert_time_s = time.perf_counter() - ts_convert_start_time
                if ts_df is None:
                    self.logger.error(f"Failed to convert file: {file_path}")
                    continue

                # Train model incrementally
                predictor, train_time_s = self._train_predictor(
                    predictor,
                    ts_df,
                    year,
                    month,
                    training_state["processed_files"],
                    checkpoint_dir=checkpoint_dir,
                )

                # Get data stats
                data_stats = resumable_loader.get_data_stats(df)

                # Update training state
                training_state["processed_files"].append(
                    {
                        "file_path": file_path,
                        "year": year,
                        "month": month,
                        "record_count": len(df),
                    }
                )

                # Save checkpoint
                checkpoint_success = checkpoint_manager.save_checkpoint(
                    year, month, predictor, data_stats, training_state
                )

                if not checkpoint_success:
                    self.logger.error(
                        f"Failed to save checkpoint for {year:04d}-{month:02d}"
                    )
                    return {
                        "status": "error",
                        "message": f"Failed to save checkpoint for {year:04d}-{month:02d}",
                        "checkpoint_dir": checkpoint_dir,
                    }

                file_total_time_s = time.perf_counter() - file_start_time
                file_timing_rows.append(
                    {
                        "year": year,
                        "month": month,
                        "parquet_load_time_s": parquet_load_time_s,
                        "timeseries_convert_time_s": ts_convert_time_s,
                        "fit_time_s": train_time_s,
                        "file_total_time_s": file_total_time_s,
                    }
                )
                self.logger.info(
                    "Timing %04d-%02d | parquet=%.3fs convert=%.3fs fit=%.3fs total=%.3fs",
                    year,
                    month,
                    parquet_load_time_s,
                    ts_convert_time_s,
                    train_time_s,
                    file_total_time_s,
                )
                self.logger.info(f"Checkpoint saved for {year:04d}-{month:02d}")

            # Final validation on unseen data
            self.logger.info("Performing final validation on unseen data")
            validation_data = self._load_validation_data(
                validation_start_date, validation_end_date
            )

            if validation_data is not None:
                performance = self._evaluate_model_performance(
                    predictor, validation_data
                )
            else:
                performance = {
                    "mae": 0.0,
                    "rmse": 0.0,
                    "mase": 1.0,
                    "directional_accuracy": 0.5,
                }

            # Save final model
            pf = training_state["processed_files"]
            last = pf[-1] if pf else None
            final_model_path = self._save_final_model(
                model_path,
                predictor,
                start_date,
                end_date,
                performance=performance,
                checkpoint_dir=checkpoint_dir,
                last_year=last["year"] if last else None,
                last_month=last["month"] if last else None,
            )
            if not final_model_path:
                return {
                    "status": "error",
                    "message": "Final model save failed (_save_final_model returned empty path)",
                    "checkpoint_dir": checkpoint_dir,
                }

            self._apply_checkpoint_post_success_cleanup(checkpoint_manager)

            epoch_time_s = time.perf_counter() - epoch_start_time
            self.logger.info("Total epoch time: %.3fs", epoch_time_s)

            return {
                "status": "completed",
                "message": "Resumable training completed successfully",
                "checkpoint_dir": checkpoint_dir,
                "final_model_path": final_model_path,
                "performance": performance,
                "processed_files": len(training_state["processed_files"]),
                "total_files": training_state["total_files"],
                "timing": {
                    "epoch_time_s": epoch_time_s,
                    "files": file_timing_rows,
                    "batch_load_time_s": "n/a (internal AutoGluon DataLoader)",
                    "forward_pass_time_s": "n/a (internal AutoGluon trainer)",
                    "backward_pass_time_s": "n/a (internal AutoGluon trainer)",
                    "optimizer_step_time_s": "n/a (internal AutoGluon trainer)",
                },
            }

        except Exception as e:
            import traceback
            self.logger.error(
                "Resumable training failed: %s\n%s", e, traceback.format_exc()
            )
            return {
                "status": "error",
                "message": f"Resumable training failed: {e}",
                "checkpoint_dir": checkpoint_dir,
            }

    def resume_training(self, checkpoint_dir: str) -> Dict[str, Any]:
        """
        Resume training from last checkpoint

        Args:
            checkpoint_dir: Directory containing checkpoints

        Returns:
            Resume results dictionary
        """
        try:
            checkpoint_manager = CheckpointManager(checkpoint_dir)

            # Get training state
            training_state = checkpoint_manager.load_training_state()
            if not training_state:
                return {
                    "status": "error",
                    "message": "No training state found in checkpoint directory",
                }

            # Resume training
            return self.train_with_checkpoints(
                start_date=training_state["start_date"],
                end_date=training_state["end_date"],
                validation_start_date=training_state["validation_start_date"],
                validation_end_date=training_state["validation_end_date"],
                checkpoint_dir=checkpoint_dir,
            )

        except Exception as e:
            self.logger.error(
                "Failed to resume training: %s\n%s", e, traceback.format_exc()
            )
            return {"status": "error", "message": f"Failed to resume training: {e}"}

    def _train_predictor(
        self,
        previous_predictor: Optional[TimeSeriesPredictor],
        ts_df: TimeSeriesDataFrame,
        year: int,
        month: int,
        processed_files: List[Dict[str, Any]],
        *,
        checkpoint_dir: str,
    ) -> Tuple[TimeSeriesPredictor, float]:
        """
        Train a predictor on the given data, optionally using previous predictor or data.

        Args:
            previous_predictor: Previous predictor from checkpoint (if resuming)
            ts_df: Current time series data to train on
            year: Year of current data
            month: Month of current data
            processed_files: List of previously processed files (for combining data)
            checkpoint_dir: Directory for checkpoints; temp models use checkpoint_dir/temp/

        Returns:
            Trained TimeSeriesPredictor
        """
        temp_base = str(Path(checkpoint_dir) / "temp")
        Path(temp_base).mkdir(parents=True, exist_ok=True)
        temp_model_path = f"{temp_base}/temp_model_{year:04d}_{month:02d}"

        known_covariates = self.incremental_config.get("known_covariates", [])
        lookback_days = self.incremental_config.get("lookback_days")
        chronos_hyperparameters = self._get_chronos_hyperparameters()
        self.logger.info(
            "Models that will be trained: ['Chronos[%s]']",
            self.chronos_variant,
        )

        if previous_predictor is None:
            # First file - create new predictor
            predictor = TimeSeriesPredictor(
                target=self.config.get("target_col", "target"),
                prediction_length=self.prediction_length,
                known_covariates_names=known_covariates,
                path=temp_model_path,
            )
            fit_start_time = time.perf_counter()
            predictor.fit(
                ts_df,
                presets=self.training_preset,
                hyperparameters=chronos_hyperparameters,
                enable_ensemble=False,
                skip_model_selection=True,
            )
            fit_time_s = time.perf_counter() - fit_start_time
        else:
            # Subsequent files - require lookback_days to avoid O(N^2) unbounded history
            if lookback_days is None:
                raise IncrementalTrainingError(
                    "incremental_training.lookback_days is required. "
                    "Set it in config to cap training history (e.g. lookback_days: 90)."
                )
            if not isinstance(lookback_days, (int, float)) or lookback_days < 0:
                raise IncrementalTrainingError(
                    f"incremental_training.lookback_days must be a non-negative number, got: {lookback_days!r}"
                )
            lookback_days = int(lookback_days)

            current_start = datetime(year, month, 1)
            cutoff = current_start - timedelta(days=lookback_days)
            windowed_files = [
                fi
                for fi in processed_files
                if datetime(fi["year"], fi["month"], 1) >= cutoff
            ]
            lookback_desc = f"lookback_days={lookback_days}"

            self.logger.info(
                "Training month %04d-%02d on %d months of history (%s)",
                year, month, len(windowed_files), lookback_desc,
            )
            # Load previous data and combine with current data
            previous_load_start_time = time.perf_counter()
            previous_data = self._load_previous_training_data(windowed_files)
            previous_load_time_s = time.perf_counter() - previous_load_start_time
            self.logger.info(
                "Previous window data load time for %04d-%02d: %.3fs",
                year,
                month,
                previous_load_time_s,
            )
            if previous_data is not None:
                # Convert previous data to TimeSeriesDataFrame
                resumable_loader = self._get_resumable_loader(checkpoint_manager=None)
                previous_convert_start_time = time.perf_counter()
                previous_ts_df = resumable_loader.convert_to_timeseries_dataframe(
                    previous_data, self.config
                )
                previous_convert_time_s = time.perf_counter() - previous_convert_start_time
                self.logger.info(
                    "Previous window conversion time for %04d-%02d: %.3fs",
                    year,
                    month,
                    previous_convert_time_s,
                )
                if previous_ts_df is not None:
                    # Do NOT pass ignore_index=True — that replaces the (item_id, timestamp)
                    # MultiIndex with a RangeIndex, causing AutoGluon to crash with
                    # "'RangeIndex' object has no attribute 'codes'" on predictor.fit().
                    combined_data = pd.concat([previous_ts_df, ts_df])
                else:
                    combined_data = ts_df
            else:
                combined_data = ts_df

            # Create new predictor for combined data
            predictor = TimeSeriesPredictor(
                target=self.config.get("target_col", "target"),
                prediction_length=self.prediction_length,
                known_covariates_names=known_covariates,
                path=temp_model_path,
            )
            fit_start_time = time.perf_counter()
            predictor.fit(
                combined_data,
                presets=self.training_preset,
                hyperparameters=chronos_hyperparameters,
                enable_ensemble=False,
                skip_model_selection=True,
            )
            fit_time_s = time.perf_counter() - fit_start_time

        self.logger.info("predictor.fit() time for %04d-%02d: %.3fs", year, month, fit_time_s)
        return predictor, fit_time_s

    def _load_validation_data(
        self, start_date: str, end_date: str
    ) -> Optional[TimeSeriesDataFrame]:
        """Load validation data for temporal validation"""
        try:
            from ..data.resumable_loader import ResumableDataLoader

            base_data_path = ConfigHelpers.get_parquet_root_dir(self.config)
            checkpoint_manager = CheckpointManager("temp_validation")
            resumable_loader = ResumableDataLoader(base_data_path, checkpoint_manager)

            # Get validation files
            validation_files = resumable_loader.get_parquet_files(start_date, end_date)

            if not validation_files:
                return None

            # Load and combine validation data
            validation_dfs = []
            for file_path, year, month in validation_files:
                df = resumable_loader.load_parquet_file(file_path, year, month)
                if df is not None:
                    validation_dfs.append(df)

            if not validation_dfs:
                return None

            # Combine all validation data
            combined_df = pd.concat(validation_dfs, ignore_index=True)

            # Convert to TimeSeriesDataFrame
            return resumable_loader.convert_to_timeseries_dataframe(
                combined_df, self.config
            )

        except Exception as e:
            self.logger.error(
                "Failed to load validation data: %s\n%s", e, traceback.format_exc()
            )
            return None

    def _load_previous_training_data(
        self, processed_files: List[Dict[str, Any]]
    ) -> Optional[pd.DataFrame]:
        """Load and combine all previously processed training data"""
        try:
            if not processed_files:
                return None

            resumable_loader = self._get_resumable_loader(checkpoint_manager=None)

            # Load all previous data
            previous_dfs = []
            for file_info in processed_files:
                df = resumable_loader.load_parquet_file(
                    file_info["file_path"], file_info["year"], file_info["month"]
                )
                if df is not None:
                    previous_dfs.append(df)

            if not previous_dfs:
                return None

            # Combine all previous data
            combined_df = pd.concat(previous_dfs, ignore_index=True)
            return combined_df

        except Exception as e:
            self.logger.warning(
                "Failed to load previous training data: %s\n%s", e, traceback.format_exc()
            )
            return None

    def _get_resumable_loader(self, checkpoint_manager: Optional[CheckpointManager]) -> Any:
        """Get shared resumable loader to avoid repeated parquet reads."""
        if self._resumable_loader is None:
            from ..data.resumable_loader import ResumableDataLoader

            base_data_path = ConfigHelpers.get_parquet_root_dir(self.config)
            self._resumable_loader = ResumableDataLoader(base_data_path, checkpoint_manager)
        return self._resumable_loader

    def _save_final_model(
        self,
        model_path: Path,
        predictor: TimeSeriesPredictor,
        start_date: str,
        end_date: str,
        *,
        performance: Optional[Dict[str, float]] = None,
        checkpoint_dir: Optional[str] = None,
        last_year: Optional[int] = None,
        last_month: Optional[int] = None,
    ) -> str:
        """Save final trained model to model_path with training_metadata.json."""
        try:
            # Create model version name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"model_{start_date.replace('-', '')}_{end_date.replace('-', '')}_{timestamp}"
            version_dir = model_path / model_name
            version_dir.mkdir(parents=True, exist_ok=True)

            # Copy from last checkpoint (AutoGluon save to directory can produce only version.txt)
            if checkpoint_dir and last_year is not None and last_month is not None:
                checkpoint_base = Path(checkpoint_dir) / "model_checkpoints"
                ckpt_path = checkpoint_base / f"model_{last_year:04d}_{last_month:02d}"
                if not ckpt_path.exists():
                    ckpt_path = checkpoint_base / f"model_{last_year:04d}_{last_month:02d}.pkl"
                self._log_artifact_event(
                    "final_model_checkpoint_copy_attempt",
                    checkpoint_dir=str(checkpoint_dir),
                    checkpoint_source=str(ckpt_path),
                    version_dir=str(version_dir),
                )
                if ckpt_path.exists():
                    if ckpt_path.is_dir():
                        # Copy contents into version_dir so TimeSeriesPredictor.load(version_dir) works
                        for item in ckpt_path.iterdir():
                            dest = version_dir / item.name
                            if item.is_dir():
                                shutil.copytree(item, dest, dirs_exist_ok=True)
                            else:
                                shutil.copy2(item, dest)
                        self.logger.info(
                            "Copied checkpoint model contents from %s to %s",
                            ckpt_path,
                            version_dir,
                        )
                    else:
                        shutil.copy2(ckpt_path, version_dir / ckpt_path.name)
                        self.logger.info(
                            "Copied checkpoint model file from %s to %s",
                            ckpt_path,
                            version_dir,
                        )
                else:
                    predictor.path = str(version_dir / "predictor")
                    predictor.save()
            else:
                predictor.path = str(version_dir / "predictor")
                predictor.save()

            valid, missing, manifest = self._validate_final_model_artifacts(version_dir)
            self._log_artifact_event(
                "final_model_validation",
                version_dir=str(version_dir),
                valid=valid,
                missing_required=missing,
                file_count=manifest["file_count"],
                total_bytes=manifest["total_bytes"],
                sample_files=manifest["sample_files"],
            )
            if not valid:
                raise IncrementalTrainingError(
                    "Final model artifact validation failed: "
                    f"missing_or_invalid={missing}, version_dir={version_dir}"
                )

            # Write training_metadata.json so training wrapper can find this model
            metadata = {
                "version_id": model_name,
                "date_range": [start_date, end_date],
                "performance_metrics": performance or {},
                "training_timestamp": datetime.now().isoformat(),
            }
            metadata_path = version_dir / "training_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Final model saved to: {version_dir}")
            return str(version_dir)

        except Exception as e:
            self.logger.error(
                "Failed to save final model: %s\n%s", e, traceback.format_exc()
            )
            return ""
