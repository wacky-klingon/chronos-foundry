"""
Covariate trainer for enhanced Chronos training with external variables
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import AutoGluon components
try:
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error("AutoGluon import error: %s", e)
    logger.error("Please install autogluon: pip install autogluon")

# Import covariate regressors
try:
    import catboost as cb
    import xgboost as xgb
    import lightgbm as lgb

    COVARIATE_REGRESSORS_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning("Covariate regressor import warning: %s", e)
    logger.warning(
        "Some regressors may not be available. Install with: pip install catboost xgboost lightgbm"
    )
    COVARIATE_REGRESSORS_AVAILABLE = False

from .base_trainer import ChronosTrainer, TrainingError


class CovariateTrainer(ChronosTrainer):
    """Enhanced Chronos trainer with covariate integration"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger("covariate_trainer")

        # Force CPU-only training
        self._configure_cpu_training()

        # Covariate-specific configuration
        self.covariate_config = config.get("covariates", {})
        self.target_scaling = self.covariate_config.get("target_scaling", True)
        self.scaler = None
        self.covariate_regressors = {}
        self.covariate_importance = {}

        # Use high_quality preset for production training
        self.training_preset = "high_quality"

        # Log the preset being used
        self.logger.info(f"Using training preset: {self.training_preset}")

        # Get available covariate columns from parquet config
        parquet_config = config.get("parquet_loader", {})
        schema_config = parquet_config.get("schema", {})
        feature_columns = schema_config.get("feature_columns", {})

        self.available_covariates = []
        for columns in feature_columns.values():
            self.available_covariates.extend(columns)

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
        """Convert pandas DataFrame to AutoGluon TimeSeriesDataFrame with covariate features"""
        try:
            self.logger.info(
                "Converting data to TimeSeriesDataFrame format with covariates"
            )

            # Ensure we have the required columns
            required_cols = ["item_id", "timestamp", "target"]
            if not all(col in df.columns for col in required_cols):
                # Map columns if they have different names
                column_mapping = {
                    self.config.get("item_id_col", "item_id"): "item_id",
                    self.config.get(
                        "timestamp_col", "ds"
                    ): "timestamp",  # Map ds to timestamp
                    self.config.get("target_col", "target"): "target",
                }
                df = df.rename(columns=column_mapping)

            # Apply target scaling if enabled
            if self.target_scaling:
                df = self._apply_target_scaling(df)

            # Prepare covariate features for AutoGluon
            df = self._prepare_covariate_features_for_autogluon(df)

            # Create TimeSeriesDataFrame with proper covariate structure
            ts_df = TimeSeriesDataFrame.from_data_frame(
                df, id_column="item_id", timestamp_column="timestamp"
            )

            self.logger.info(
                "Created TimeSeriesDataFrame with %d records and covariate features",
                len(ts_df),
            )
            return ts_df

        except Exception as e:
            raise TrainingError(
                f"Failed to create TimeSeriesDataFrame with covariates: {e}"
            ) from e

    def train_model(self, data: pd.DataFrame, model_save_path: str = None) -> str:
        """Train Chronos model on the provided data with covariate integration"""
        try:
            self.logger.info("Starting covariate model training")

            # Convert to TimeSeriesDataFrame with covariates
            ts_data = self.prepare_timeseries_dataframe(data)

            # Create predictor with covariate support
            self.predictor = TimeSeriesPredictor(
                prediction_length=self.prediction_length,
                target="target",
                eval_metric="MASE",
            )

            # Train the model with covariate integration
            self.logger.info(
                "Training with %d records and covariate features", len(ts_data)
            )

            # Get available covariate columns for training
            covariate_columns = [
                col for col in self.available_covariates if col in ts_data.columns
            ]

            if covariate_columns:
                self.logger.info("Using covariate features: %s", covariate_columns)
                # AutoGluon will automatically use these as known covariates
                self.predictor.fit(
                    ts_data,
                    presets=self.training_preset,
                    hyperparameters={
                        "Chronos": {
                            "model_path": self.model_name,
                            "context_length": self.context_length,
                            "learning_rate": self.learning_rate,
                            "batch_size": self.batch_size,
                            "max_epochs": self.max_epochs,
                            "device": "cpu",  # Force CPU device
                            "torch_device": "cpu",  # Additional CPU enforcement
                        }
                    },
                )
            else:
                self.logger.warning(
                    "No covariate features available, training without covariates"
                )
                self.predictor.fit(
                    ts_data,
                    presets=self.training_preset,
                    hyperparameters={
                        "Chronos": {
                            "model_path": self.model_name,
                            "context_length": self.context_length,
                            "learning_rate": self.learning_rate,
                            "batch_size": self.batch_size,
                            "max_epochs": self.max_epochs,
                            "device": "cpu",  # Force CPU device
                            "torch_device": "cpu",  # Additional CPU enforcement
                        }
                    },
                )

            # Save the model
            self.predictor.save()
            self.model_path = self.predictor.path  # Get actual save location

            self.logger.info(
                "Model training completed and saved to: %s", self.model_path
            )
            return self.model_path

        except Exception as e:
            raise TrainingError(f"Model training failed: {e}") from e

    def _apply_target_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply target scaling for improved model performance"""
        try:
            if self.scaler is None:
                self.scaler = StandardScaler()
                df["target"] = self.scaler.fit_transform(df[["target"]])
                self.logger.info("Applied target scaling with StandardScaler")
            else:
                df["target"] = self.scaler.transform(df[["target"]])
                self.logger.info("Applied target scaling using fitted scaler")

            return df
        except Exception as e:
            self.logger.warning(
                f"Target scaling failed: {e}. Continuing without scaling."
            )
            return df

    def _prepare_covariate_features_for_autogluon(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepare covariate features for AutoGluon TimeSeriesDataFrame integration"""
        try:
            # Check which covariate columns are available in the data
            available_in_data = [
                col for col in self.available_covariates if col in df.columns
            ]

            if not available_in_data:
                self.logger.warning("No covariate columns found in data")
                return df

            # Process each covariate column for AutoGluon
            for col in available_in_data:
                if col not in df.columns:
                    self.logger.warning("Covariate column %s not found in data", col)
                    continue

                # Ensure covariate data is numeric
                if not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    except Exception as e:
                        self.logger.warning(
                            "Could not convert %s to numeric: %s", col, e
                        )
                        continue

                # For AutoGluon, we keep the original column names
                # AutoGluon will automatically detect these as features
                self.logger.debug("Prepared covariate feature: %s", col)

            self.logger.info(
                "Prepared %d covariate features for AutoGluon integration",
                len(available_in_data),
            )
            return df

        except Exception as e:
            self.logger.warning("Failed to prepare covariate features: %s", e)
            return df

    def train_covariate_regressors(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train covariate regressors for feature importance analysis"""
        if not COVARIATE_REGRESSORS_AVAILABLE:
            self.logger.warning(
                "Covariate regressors not available. Skipping regressor training."
            )
            return {}

        try:
            self.logger.info("Training covariate regressors")

            # Prepare data for regressor training
            X, y = self._prepare_regressor_data(data)

            if X is None or y is None:
                self.logger.warning("No suitable data for regressor training")
                return {}

            # Train different regressors
            regressors = self._get_regressor_models()
            results = {}

            for name, model in regressors.items():
                try:
                    self.logger.info(f"Training {name} regressor")
                    model.fit(X, y)

                    # Get feature importance
                    importance = self._get_feature_importance(model, X.columns)

                    # Evaluate performance
                    y_pred = model.predict(X)
                    mae = mean_absolute_error(y, y_pred)
                    rmse = np.sqrt(mean_squared_error(y, y_pred))

                    results[name] = {
                        "model": model,
                        "importance": importance,
                        "mae": mae,
                        "rmse": rmse,
                        "r2": model.score(X, y) if hasattr(model, "score") else None,
                    }

                    self.logger.info(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

                except Exception as e:
                    self.logger.warning(f"Failed to train {name} regressor: {e}")
                    continue

            self.covariate_regressors = results
            return results

        except Exception as e:
            self.logger.error(f"Failed to train covariate regressors: {e}")
            return {}

    def _prepare_regressor_data(
        self, data: pd.DataFrame
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Prepare data for regressor training"""
        try:
            # Get covariate columns
            covariate_cols = [
                col for col in data.columns if col.startswith("covariate_")
            ]

            if not covariate_cols:
                return None, None

            # Prepare features and target
            X = data[covariate_cols].copy()
            y = data["target"].copy()

            # Remove rows with missing values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]

            if len(X) == 0:
                return None, None

            return X, y

        except Exception as e:
            self.logger.error(f"Failed to prepare regressor data: {e}")
            return None, None

    def _get_regressor_models(self) -> Dict[str, Any]:
        """Get available regressor models"""
        models = {}

        if COVARIATE_REGRESSORS_AVAILABLE:
            try:
                models["catboost"] = cb.CatBoostRegressor(
                    iterations=100, learning_rate=0.1, depth=6, verbose=False
                )
            except Exception:
                self.logger.warning("CatBoost not available, skipping")

            try:
                models["xgboost"] = xgb.XGBRegressor(
                    n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
                )
            except Exception:
                self.logger.warning("XGBoost not available, skipping")

            try:
                models["lightgbm"] = lgb.LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    verbose=-1,
                )
            except Exception:
                self.logger.warning("LightGBM not available, skipping")

        # Always include RandomForest as fallback
        models["random_forest"] = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        )

        return models

    def _get_feature_importance(
        self, model: Any, feature_names: List[str]
    ) -> Dict[str, float]:
        """Get feature importance from trained model"""
        try:
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
            elif hasattr(model, "coef_"):
                importance = np.abs(model.coef_)
            else:
                return {}

            # Create importance dictionary
            importance_dict = {}
            for i, name in enumerate(feature_names):
                if i < len(importance):
                    importance_dict[name] = float(importance[i])

            return importance_dict

        except Exception as e:
            self.logger.warning(f"Failed to get feature importance: {e}")
            return {}

    def get_covariate_analysis(self) -> Dict[str, Any]:
        """Get comprehensive covariate analysis results"""
        analysis = {
            "available_covariates": self.available_covariates,
            "regressor_results": self.covariate_regressors,
            "target_scaling_applied": self.target_scaling,
            "scaler_info": str(type(self.scaler)) if self.scaler else None,
        }

        return analysis

    def export_training_report_csv(self, output_path: str) -> str:
        """Export training results and metrics to CSV format"""
        try:
            import csv
            from datetime import datetime

            self.logger.info(f"Exporting training report to CSV: {output_path}")

            # Prepare training summary data
            training_summary = {
                "timestamp": datetime.now().isoformat(),
                "model_path": self.model_path,
                "training_preset": self.training_preset,
                "target_scaling": self.target_scaling,
                "available_covariates_count": len(self.available_covariates),
                "regressors_trained": len(self.covariate_regressors),
            }

            # Create CSV with training summary
            with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)

                # Write header
                writer.writerow(["Metric", "Value"])

                # Write training summary
                for key, value in training_summary.items():
                    writer.writerow([key, value])

                # Add separator
                writer.writerow([])
                writer.writerow(["Available Covariates"])
                writer.writerow(["Covariate Name"])

                # Write available covariates
                for covariate in self.available_covariates:
                    writer.writerow([covariate])

                # Add separator
                writer.writerow([])
                writer.writerow(["Regressor Performance"])
                writer.writerow(["Regressor", "MAE", "RMSE", "R2"])

                # Write regressor results
                for name, result in self.covariate_regressors.items():
                    writer.writerow(
                        [
                            name,
                            f"{result.get('mae', 0):.6f}",
                            f"{result.get('rmse', 0):.6f}",
                            f"{result.get('r2', 0):.6f}"
                            if result.get("r2") is not None
                            else "N/A",
                        ]
                    )

                # Add separator
                writer.writerow([])
                writer.writerow(["Feature Importance"])
                writer.writerow(["Regressor", "Feature", "Importance"])

                # Write feature importance for each regressor
                for name, result in self.covariate_regressors.items():
                    importance = result.get("importance", {})
                    if importance:
                        for feature, imp_value in importance.items():
                            writer.writerow([name, feature, f"{imp_value:.6f}"])
                    else:
                        writer.writerow([name, "N/A", "N/A"])

            self.logger.info(f"Training report exported successfully to: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Failed to export training report CSV: {e}")
            raise TrainingError(f"CSV export failed: {e}") from e
