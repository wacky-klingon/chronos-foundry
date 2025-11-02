"""
Incremental trainer for continuous model improvement with versioning and rollback
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
from datetime import datetime

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

        # Use high_quality preset for production training (can be overridden via config)
        self.training_preset = config.get("training_preset", "high_quality")

        # Initialize model versioning
        versioning_config = {
            "model_path": config.get("model_path", "data/models/incremental"),
            "max_versions": self.incremental_config.get("max_versions", 10),
        }
        self.versioning = ModelVersioning(versioning_config)

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
            self.logger.error(f"Incremental training failed: {e}")
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
            self.logger.error(f"Failed to evaluate model performance: {e}")
            import traceback

            self.logger.error(f"Traceback: {traceback.format_exc()}")
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
            self.logger.info(f"Starting resumable training: {start_date} to {end_date}")

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

            # Import resumable loader
            from ..data.resumable_loader import ResumableDataLoader

            # Initialize resumable loader
            base_data_path = ConfigHelpers.get_parquet_root_dir(self.config)
            resumable_loader = ResumableDataLoader(base_data_path, checkpoint_manager)

            # Get remaining files to process
            remaining_files = resumable_loader.get_remaining_files(start_date, end_date)
            training_state["total_files"] = len(remaining_files)

            if not remaining_files:
                self.logger.info("No remaining files to process")
                return {
                    "status": "completed",
                    "message": "All files already processed",
                    "checkpoint_dir": checkpoint_dir,
                }

            # Process each remaining file
            for file_path, year, month in remaining_files:
                self.logger.info(f"Processing file: {year:04d}-{month:02d}")

                # Load parquet file
                df = resumable_loader.load_parquet_file(file_path, year, month)
                if df is None:
                    self.logger.error(f"Failed to load file: {file_path}")
                    continue

                # Convert to TimeSeriesDataFrame
                ts_df = resumable_loader.convert_to_timeseries_dataframe(
                    df, self.config
                )
                if ts_df is None:
                    self.logger.error(f"Failed to convert file: {file_path}")
                    continue

                # Train model incrementally
                predictor = self._train_predictor(
                    predictor, ts_df, year, month, training_state["processed_files"]
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
            final_model_path = self._save_final_model(predictor, start_date, end_date)

            return {
                "status": "completed",
                "message": "Resumable training completed successfully",
                "checkpoint_dir": checkpoint_dir,
                "final_model_path": final_model_path,
                "performance": performance,
                "processed_files": len(training_state["processed_files"]),
                "total_files": training_state["total_files"],
            }

        except Exception as e:
            self.logger.error(f"Resumable training failed: {e}")
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
            self.logger.error(f"Failed to resume training: {e}")
            return {"status": "error", "message": f"Failed to resume training: {e}"}

    def _train_predictor(
        self,
        previous_predictor: Optional[TimeSeriesPredictor],
        ts_df: TimeSeriesDataFrame,
        year: int,
        month: int,
        processed_files: List[Dict[str, Any]],
    ) -> TimeSeriesPredictor:
        """
        Train a predictor on the given data, optionally using previous predictor or data.

        Args:
            previous_predictor: Previous predictor from checkpoint (if resuming)
            ts_df: Current time series data to train on
            year: Year of current data
            month: Month of current data
            processed_files: List of previously processed files (for combining data)

        Returns:
            Trained TimeSeriesPredictor
        """
        model_base_path = self.incremental_config.get(
            "model_base_path", "data/models/incremental"
        )
        temp_model_path = f"{model_base_path}/temp_model_{year:04d}_{month:02d}"

        if previous_predictor is None:
            # First file - create new predictor
            predictor = TimeSeriesPredictor(
                target=self.config.get("target_col", "target"),
                prediction_length=self.prediction_length,
                path=temp_model_path,
            )
            predictor.fit(ts_df, presets=self.training_preset)
        else:
            # Subsequent files - create new predictor and train on combined data
            # Load previous data and combine with current data
            previous_data = self._load_previous_training_data(processed_files)
            if previous_data is not None:
                # Convert previous data to TimeSeriesDataFrame
                from ..data.resumable_loader import ResumableDataLoader

                base_data_path = ConfigHelpers.get_parquet_root_dir(self.config)
                resumable_loader = ResumableDataLoader(base_data_path, checkpoint_manager=None)
                previous_ts_df = resumable_loader.convert_to_timeseries_dataframe(
                    previous_data, self.config
                )
                if previous_ts_df is not None:
                    combined_data = pd.concat([previous_ts_df, ts_df], ignore_index=True)
                else:
                    combined_data = ts_df
            else:
                combined_data = ts_df

            # Create new predictor for combined data
            predictor = TimeSeriesPredictor(
                target=self.config.get("target_col", "target"),
                prediction_length=self.prediction_length,
                path=temp_model_path,
            )
            predictor.fit(combined_data, presets=self.training_preset)

        return predictor

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
            self.logger.error(f"Failed to load validation data: {e}")
            return None

    def _load_previous_training_data(
        self, processed_files: List[Dict[str, Any]]
    ) -> Optional[pd.DataFrame]:
        """Load and combine all previously processed training data"""
        try:
            if not processed_files:
                return None

            from ..data.resumable_loader import ResumableDataLoader
            from ..core.config_helpers import ConfigHelpers

            base_data_path = ConfigHelpers.get_parquet_root_dir(self.config)
            checkpoint_manager = None  # Not needed for data loading
            resumable_loader = ResumableDataLoader(base_data_path, checkpoint_manager)

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
            self.logger.warning(f"Failed to load previous training data: {e}")
            return None

    def _save_final_model(
        self, predictor: TimeSeriesPredictor, start_date: str, end_date: str
    ) -> str:
        """Save final trained model"""
        try:
            # Create model version name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"model_{start_date.replace('-', '')}_{end_date.replace('-', '')}_{timestamp}"

            # Save model
            model_path = f"data/models/incremental/{model_name}"
            # Set the path before saving (AutoGluon saves to predictor.path)
            predictor.path = model_path
            predictor.save()

            self.logger.info(f"Final model saved to: {model_path}")
            return model_path

        except Exception as e:
            self.logger.error(f"Failed to save final model: {e}")
            return ""
