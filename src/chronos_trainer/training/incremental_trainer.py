"""
Incremental trainer for continuous model improvement with versioning and rollback
"""

import json
import hashlib
import os
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
from ..data.resumable_loader import log_autogluon_timeseries_dataframe_probe
from ..metrics.recorder import NullMetricsRecorder


class IncrementalTrainingError(Exception):
    """Raised when incremental training fails"""

    pass


class IncrementalTrainer(CovariateTrainer):
    """Incremental trainer for continuous model improvement with versioning and rollback"""

    def __init__(self, config: Dict[str, Any]) -> None:
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
        self.max_model_checkpoints = self._get_required_max_model_checkpoints()

        # Use high_quality preset for production training (can be overridden via config)
        self.training_preset = config.get("training_preset", "high_quality")
        # chronos_variant is retained for logging and training_metadata.json only.
        self.chronos_variant = (
            str(self.incremental_config.get("chronos_model_variant", "")).strip().lower()
            or "unknown"
        )
        self.chronos_model_path = self._resolve_chronos_local_model_path()
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

    def _get_fine_tune_verification_settings(self) -> Dict[str, Any]:
        raw_settings = self.incremental_config.get("fine_tune_verification", {})
        settings = dict(raw_settings) if isinstance(raw_settings, dict) else {}
        settings.setdefault("enabled", True)
        settings.setdefault("min_fit_runtime_seconds", 30.0)
        settings.setdefault("allow_constant_validation_target", False)
        settings.setdefault("allow_covariate_evaluation_skip", False)
        return settings

    def _new_verification_state(self) -> Dict[str, Any]:
        return {
            "row_count": 0,
            "item_ids": [],
            "observed_start_timestamp": None,
            "observed_end_timestamp": None,
            "known_covariates": [],
            "fit_runtime_seconds": 0.0,
            "processed_files": [],
        }

    def _update_verification_state(
        self, verification_state: Dict[str, Any], df: pd.DataFrame, fit_time_s: float
    ) -> None:
        timestamp_col = self.config.get("timestamp_col", "timestamp")
        item_id_col = self.config.get("item_id_col", "item_id")
        known_covariates = self.incremental_config.get("known_covariates", [])

        verification_state["row_count"] = int(
            verification_state.get("row_count", 0) + len(df)
        )

        seen_item_ids = set(verification_state.get("item_ids", []))
        if item_id_col in df.columns:
            seen_item_ids.update(str(value) for value in df[item_id_col].dropna().unique())
        verification_state["item_ids"] = sorted(seen_item_ids)

        if timestamp_col in df.columns and not df.empty:
            current_start = pd.to_datetime(df[timestamp_col]).min().isoformat()
            current_end = pd.to_datetime(df[timestamp_col]).max().isoformat()
            existing_start = verification_state.get("observed_start_timestamp")
            existing_end = verification_state.get("observed_end_timestamp")
            if existing_start is None or current_start < existing_start:
                verification_state["observed_start_timestamp"] = current_start
            if existing_end is None or current_end > existing_end:
                verification_state["observed_end_timestamp"] = current_end

        seen_covariates = set(verification_state.get("known_covariates", []))
        seen_covariates.update(col for col in known_covariates if col in df.columns)
        verification_state["known_covariates"] = sorted(seen_covariates)
        verification_state["fit_runtime_seconds"] = round(
            float(verification_state.get("fit_runtime_seconds", 0.0)) + float(fit_time_s),
            6,
        )

    def _build_dataset_fingerprint(
        self, verification_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        fingerprint_payload = {
            "row_count": int(verification_state.get("row_count", 0)),
            "item_count": len(verification_state.get("item_ids", [])),
            "observed_start_timestamp": verification_state.get("observed_start_timestamp"),
            "observed_end_timestamp": verification_state.get("observed_end_timestamp"),
            "target_col": self.config.get("target_col", "target"),
            "known_covariates": sorted(verification_state.get("known_covariates", [])),
        }
        fingerprint_payload["fingerprint_hash"] = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return fingerprint_payload

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

    def _get_required_max_model_checkpoints(self) -> int:
        """Read required per-save checkpoint retention cap from config with no defaults."""
        raw_value = self.incremental_config.get("max_model_checkpoints")
        if raw_value is None:
            raise IncrementalTrainingError(
                "incremental_training.max_model_checkpoints is required. "
                "Set it explicitly in config (e.g. max_model_checkpoints: 2)."
            )
        if not isinstance(raw_value, int) or raw_value < 1:
            raise IncrementalTrainingError(
                "incremental_training.max_model_checkpoints must be an integer >= 1."
            )
        return raw_value

    def _apply_checkpoint_post_success_cleanup(
        self, checkpoint_manager: CheckpointManager
    ) -> None:
        """After successful final export only; does nothing if cleanup disabled."""
        if not self.checkpoint_post_success_cleanup:
            return
        checkpoint_manager.remove_temp_directory()
        checkpoint_manager.prune_model_checkpoints(self.rollback_window_versions)

    def _resolve_chronos_local_model_path(self) -> str:
        """
        Resolve and validate the local Chronos base model directory from config.

        Reads ``incremental_training.chronos_local_model_dir``, verifies the path
        exists on disk, and returns its absolute string form.  No fallback to a
        HuggingFace Hub ID is permitted under any condition.

        Raises:
            IncrementalTrainingError: if the config key is absent, the value is
                empty, or the resolved path does not exist as a directory on disk.
        """
        local_dir = self.incremental_config.get("chronos_local_model_dir")
        if not local_dir or not str(local_dir).strip():
            raise IncrementalTrainingError(
                "incremental_training.chronos_local_model_dir is required. "
                "Set it to the local filesystem path of the downloaded Chronos base model. "
                "Run scripts/bootstrap_base_model.py to populate the directory."
            )
        path = Path(str(local_dir)).resolve()
        if not path.exists():
            raise IncrementalTrainingError(
                f"incremental_training.chronos_local_model_dir does not exist on disk: {path}"
            )
        if not path.is_dir():
            raise IncrementalTrainingError(
                f"incremental_training.chronos_local_model_dir is not a directory: {path}"
            )
        return str(path)

    def _validate_chronos_only_configuration(self) -> None:
        """Raise if incremental_training.chronos_only is not true."""
        if not self.chronos_only:
            raise IncrementalTrainingError(
                "incremental_training.chronos_only must be true. "
                "Mixed-model incremental training is not supported."
            )

    def _get_fine_tune_config(self) -> Dict[str, Any]:
        """Return the fine_tune sub-block from incremental_training config with defaults."""
        raw = self.incremental_config.get("fine_tune", {})
        cfg = dict(raw) if isinstance(raw, dict) else {}
        cfg.setdefault("enabled", True)
        cfg.setdefault("learning_rate", 1e-5)
        cfg.setdefault("steps", 200)
        cfg.setdefault("batch_size", 8)
        return cfg

    def _get_chronos_hyperparameters(self) -> Dict[str, Dict[str, Any]]:
        """Build Chronos-only hyperparameters for TimeSeriesPredictor.fit.

        The fine_tune keys are required for AutoGluon's Chronos backend to perform
        real fine-tuning and write a fine-tuned-ckpt directory.  Without them,
        AutoGluon runs in zero-shot mode and produces only pickle artifacts, which
        causes safetensors export to fail with no checkpoint found.
        """
        ft = self._get_fine_tune_config()
        params: Dict[str, Any] = {
            "model_path": self.chronos_model_path,
            "context_length": self.context_length,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "device": self.device,
            "fine_tune": ft["enabled"],
            "fine_tune_lr": ft["learning_rate"],
            "fine_tune_steps": ft["steps"],
            "fine_tune_batch_size": ft["batch_size"],
        }
        return {"Chronos": params}

    def _assert_fine_tuned_checkpoint_exists(self, predictor_path: Path) -> None:
        """Raise IncrementalTrainingError if fine-tuning was requested but AutoGluon
        did not produce a fine-tuned-ckpt directory under predictor_path/models/.

        A missing checkpoint means AutoGluon ran in zero-shot mode despite receiving
        fine_tune: true, which will cause safetensors export to fail downstream.
        Failing here with diagnostics is cheaper than running the rest of the pipeline
        against an incomplete artifact.
        """
        ft = self._get_fine_tune_config()
        if not ft.get("enabled", True):
            return

        models_dir = predictor_path / "models"
        if not models_dir.exists():
            discovered: List[str] = []
        else:
            discovered = sorted(
                str(p.relative_to(predictor_path))
                for p in models_dir.rglob("*")
                if p.is_dir()
            )

        ckpt_dirs = [d for d in discovered if "fine-tuned-ckpt" in d]
        if ckpt_dirs:
            self.logger.info(
                "fine-tuned-ckpt confirmed at: %s",
                ckpt_dirs,
            )
            return

        raise IncrementalTrainingError(
            "fine_tune is enabled but no fine-tuned-ckpt directory was found under "
            f"{predictor_path / 'models'}. AutoGluon may have silently fallen back to "
            "zero-shot mode. Discovered model subdirectories: "
            + (", ".join(discovered) if discovered else "(none)")
            + ". Check that fine_tune_lr, fine_tune_steps, and fine_tune_batch_size "
            "are accepted by the installed AutoGluon version."
        )

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

    @staticmethod
    def _candidate_has_safetensors(path: Path) -> bool:
        """Return True if path contains a valid top-level safetensors/ export."""
        return (
            (path / "safetensors" / "config.json").exists()
            and (path / "safetensors" / "model.safetensors").exists()
        )

    def _resolve_warm_start_predictor(
        self, previous_model_path: Optional[str]
    ) -> Tuple[Optional[TimeSeriesPredictor], str, str]:
        """
        Resolve best-effort warm start predictor for checkpoint training.

        Warm-start eligibility now requires a safetensors-verified artifact.
        If the candidate exists but is missing top-level safetensors artifacts,
        the predictor is still loaded (migration compatibility) but the mode is
        logged as ``warm_start_fallback_legacy_predictor`` rather than
        ``warm_start_from_safetensors_verified``.

        Returns:
            (predictor, mode_token, detail)
        """
        if not previous_model_path:
            return None, "fresh_start_no_previous_model", "previous_model=not_provided"

        path = Path(previous_model_path)
        if not path.exists():
            reason = "previous_model_missing_on_disk"
            detail = (
                f"previous_model=path={previous_model_path!r} exists=False "
                f"fallback_reason={reason}"
            )
            self.logger.warning(
                "incremental_checkpoint_decision mode=fresh_start_fallback_from_previous_model %s",
                detail,
            )
            return None, "fresh_start_fallback_from_previous_model", detail

        has_safetensors = self._candidate_has_safetensors(path)
        if has_safetensors:
            warm_start_mode = "warm_start_from_safetensors_verified"
        else:
            warm_start_mode = "warm_start_fallback_legacy_predictor"
            self.logger.warning(
                "incremental_checkpoint_decision mode=warm_start_fallback_legacy_predictor "
                "previous_model=path=%r safetensors_missing=True "
                "(proceeding with legacy predictor load; re-run with updated wrapper to satisfy contract)",
                previous_model_path,
            )

        try:
            predictor = self._load_previous_model(str(path))
            detail = (
                f"previous_model=path={previous_model_path!r} exists=True "
                f"safetensors_verified={has_safetensors}"
            )
            self.logger.info(
                "incremental_checkpoint_decision mode=%s %s",
                warm_start_mode,
                detail,
            )
            return predictor, warm_start_mode, detail
        except Exception as exc:
            reason = str(exc).replace("\n", " ").replace("\r", " ")
            detail = (
                f"previous_model=path={previous_model_path!r} exists=True "
                f"safetensors_verified={has_safetensors} "
                f"fallback_reason={reason}"
            )
            self.logger.warning(
                "incremental_checkpoint_decision mode=fresh_start_fallback_from_previous_model %s",
                detail,
            )
            return None, "fresh_start_fallback_from_previous_model", detail

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

    def _invalid_validation_metrics(
        self,
        reason: str,
        summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a consistent invalid-validation payload."""
        return {
            "mae": None,
            "rmse": None,
            "mase": None,
            "directional_accuracy": None,
            "validation_valid": False,
            "validation_reason": reason,
            "validation_summary": summary or {},
        }

    def _evaluate_model_performance(
        self, predictor: TimeSeriesPredictor, data: TimeSeriesDataFrame
    ) -> Dict[str, Any]:
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
                return self._invalid_validation_metrics("missing_target_column")

            # Build per-series temporal splits so each item_id contributes a holdout horizon.
            total_length = len(data)
            prediction_length = self.prediction_length

            self.logger.info(f"Total data length: {total_length}")
            self.logger.info(f"Prediction length: {prediction_length}")

            eval_df = data.reset_index()
            if "item_id" not in eval_df.columns or "timestamp" not in eval_df.columns:
                self.logger.warning(
                    "TimeSeriesDataFrame reset_index missing required columns: item_id/timestamp"
                )
                return self._invalid_validation_metrics("missing_index_columns")

            series_count = eval_df["item_id"].nunique()
            self.logger.info(f"Detected item_id series count: {series_count}")

            min_series_len = prediction_length * 2
            series_groups = eval_df.groupby("item_id", sort=False)
            train_frames: List[pd.DataFrame] = []
            val_frames: List[pd.DataFrame] = []
            excluded_series = 0

            for _, group_df in series_groups:
                ordered = group_df.sort_values("timestamp")
                if len(ordered) < min_series_len:
                    excluded_series += 1
                    continue
                train_frames.append(ordered.iloc[:-prediction_length])
                val_frames.append(ordered.iloc[-prediction_length:])

            included_series = len(train_frames)
            self.logger.info(
                "Per-series split: included_series=%d excluded_series=%d min_required_length=%d",
                included_series,
                excluded_series,
                min_series_len,
            )

            if included_series == 0:
                self.logger.warning(
                    "No series had enough rows for validation: required >= %d per item_id",
                    min_series_len,
                )
                return self._invalid_validation_metrics(
                    "insufficient_series_length",
                    summary={
                        "series_total": int(series_count),
                        "series_included": 0,
                        "series_excluded": int(excluded_series),
                        "min_required_length": int(min_series_len),
                    },
                )

            train_eval_df = pd.concat(train_frames, ignore_index=True)
            val_eval_df = pd.concat(val_frames, ignore_index=True)
            train_data = TimeSeriesDataFrame.from_data_frame(
                train_eval_df, id_column="item_id", timestamp_column="timestamp"
            )
            val_data = TimeSeriesDataFrame.from_data_frame(
                val_eval_df, id_column="item_id", timestamp_column="timestamp"
            )

            self.logger.info(f"Train data length: {len(train_data)}")
            self.logger.info(f"Validation data length: {len(val_data)}")
            self.logger.info(
                "Validation timestamp bounds: %s -> %s",
                val_eval_df["timestamp"].min(),
                val_eval_df["timestamp"].max(),
            )

            # Log validation data details
            val_target = val_data["target"].values
            val_unique_count = len(np.unique(val_target))
            self.logger.info(
                f"Validation target stats - min: {val_target.min():.6f}, max: {val_target.max():.6f}, mean: {val_target.mean():.6f}"
            )
            self.logger.info(f"Validation target unique values: {val_unique_count}")
            self.logger.info(f"Validation target sample: {val_target[:10]}")

            # Check if validation data is constant (data quality issue)
            if val_unique_count == 1:
                self.logger.warning(
                    "VALIDATION DATA IS CONSTANT! This indicates a data quality issue."
                )
                self.logger.warning(f"All validation values are: {val_target[0]}")
                return self._invalid_validation_metrics(
                    "constant_validation_target",
                    summary={
                        "series_total": int(series_count),
                        "series_included": int(included_series),
                        "series_excluded": int(excluded_series),
                        "validation_rows": int(len(val_data)),
                        "constant_value": float(val_target[0]),
                    },
                )

            # Generate predictions for evaluation
            self.logger.info("Generating predictions...")
            known_covariates_names = self.incremental_config.get("known_covariates", [])
            missing_covariates = [
                c for c in known_covariates_names if c not in val_data.columns
            ]
            if known_covariates_names and missing_covariates:
                # The predictor was trained with known_covariates but the validation
                # parquet data is raw (engineered features not recomputed at eval time).
                # Calling predict without covariates would raise ValueError from AutoGluon.
                # Return a named reason so the caller can allow this via config rather
                # than silently catching a generic evaluation_exception.
                self.logger.warning(
                    "Evaluation skipped: predictor requires %d known covariate(s) "
                    "that are absent from the validation data. "
                    "Missing: %s",
                    len(missing_covariates),
                    missing_covariates,
                )
                return self._invalid_validation_metrics(
                    "evaluation_skipped_covariates_not_in_validation_data",
                    summary={
                        "required_covariates": len(known_covariates_names),
                        "missing_covariates": missing_covariates,
                        "validation_rows": int(len(val_data)),
                    },
                )
            if known_covariates_names:
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
            if min_len == 0:
                self.logger.warning("No overlapping rows between predictions and validation")
                return self._invalid_validation_metrics(
                    "empty_prediction_alignment",
                    summary={
                        "validation_rows": int(len(val_target)),
                        "predicted_rows": int(len(predicted_values)),
                    },
                )

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
                "validation_valid": True,
                "validation_reason": "ok",
                "validation_summary": {
                    "series_total": int(series_count),
                    "series_included": int(included_series),
                    "series_excluded": int(excluded_series),
                    "validation_rows": int(len(val_data)),
                },
            }

            self.logger.info("=" * 80)
            self.logger.info("FINAL EVALUATION RESULTS")
            self.logger.info("=" * 80)
            for metric in ("mae", "rmse", "mase", "directional_accuracy"):
                value = performance.get(metric)
                self.logger.info(f"{metric.upper()}: {value:.6f}")
            self.logger.info(
                "VALIDATION_STATUS: valid=%s reason=%s",
                performance["validation_valid"],
                performance["validation_reason"],
            )
            self.logger.info("=" * 80)

            return performance

        except Exception as e:
            self.logger.error(
                "Failed to evaluate model performance: %s\n%s", e, traceback.format_exc()
            )
            return self._invalid_validation_metrics("evaluation_exception")

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
        metrics_recorder: Optional[NullMetricsRecorder] = None,
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
            _rec: NullMetricsRecorder = (
                metrics_recorder if metrics_recorder is not None else NullMetricsRecorder()
            )
            _current_phase: Optional[str] = None
            model_path = self._ensure_model_path_available(self.config.get("model_path"))
            self._ensure_path_available(checkpoint_dir, "checkpoint_dir")
            self.logger.info(f"Starting resumable training: {start_date} to {end_date}")
            epoch_start_time = time.perf_counter()
            file_timing_rows: List[Dict[str, Any]] = []

            # Initialize checkpoint manager
            checkpoint_manager = CheckpointManager(
                checkpoint_dir,
                max_model_checkpoints=self.max_model_checkpoints,
            )

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
                predictor, warm_start_mode, warm_start_detail = (
                    self._resolve_warm_start_predictor(previous_model_path)
                )
                if warm_start_mode == "warm_start_from_previous_model":
                    self.logger.info(
                        "Starting training with warm-start predictor from previous model"
                    )
                else:
                    self.logger.info("Starting fresh training")
                    if warm_start_mode == "fresh_start_no_previous_model":
                        self.logger.info(
                            "incremental_checkpoint_decision mode=fresh_start_no_previous_model %s",
                            warm_start_detail,
                        )
                training_state = {
                    "start_date": start_date,
                    "end_date": end_date,
                    "validation_start_date": validation_start_date,
                    "validation_end_date": validation_end_date,
                    "processed_files": [],
                    "total_files": 0,
                }

            verification_state = training_state.get("verification_state")
            if not isinstance(verification_state, dict):
                verification_state = self._new_verification_state()
                training_state["verification_state"] = verification_state

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

            # Initialize resumable loader
            resumable_loader = self._get_resumable_loader(checkpoint_manager)

            # Get remaining files to process
            _current_phase = "data_download"
            _rec.start_phase("data_download")
            all_parquet_files = resumable_loader.get_parquet_files(start_date, end_date)
            remaining_files = resumable_loader.get_remaining_files(start_date, end_date)
            training_state["total_files"] = len(all_parquet_files)
            _rec.set_file_counts(len(training_state["processed_files"]), len(all_parquet_files))
            _rec.end_phase("data_download")
            _current_phase = None

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
            _current_phase = "train"
            _rec.start_phase("train")
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

                df = self.preprocess_raw_dataframe(df)

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
                self._update_verification_state(verification_state, df, train_time_s)

                # Update training state
                training_state["processed_files"].append(
                    {
                        "file_path": file_path,
                        "year": year,
                        "month": month,
                        "record_count": len(df),
                    }
                )
                verification_state["processed_files"] = list(
                    training_state["processed_files"]
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
                try:
                    checkpoint_manager.remove_temp_model_directory(year, month)
                except OSError as e:
                    self.logger.warning(
                        "Failed to cleanup temp model dir for %04d-%02d: %s",
                        year,
                        month,
                        e,
                    )

                try:
                    _rec.add_bytes_in(Path(file_path).stat().st_size)
                except Exception:
                    pass
                _rec.set_file_counts(
                    len(training_state["processed_files"]),
                    training_state["total_files"],
                )

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

            _rec.end_phase("train")
            _current_phase = None

            # Final validation on unseen data
            _current_phase = "validate"
            _rec.start_phase("validate")
            self.logger.info("Performing final validation on unseen data")
            validation_data = self._load_validation_data(
                validation_start_date, validation_end_date
            )

            if validation_data is not None:
                performance = self._evaluate_model_performance(
                    predictor, validation_data
                )
            else:
                self.logger.warning(
                    "Validation data unavailable for %s to %s; final metrics marked invalid",
                    validation_start_date,
                    validation_end_date,
                )
                performance = self._invalid_validation_metrics("validation_data_unavailable")

            _rec.end_phase("validate")
            _current_phase = None

            if not performance.get("validation_valid", False):
                self.logger.warning(
                    "Training completed with invalid validation state: reason=%s summary=%s",
                    performance.get("validation_reason"),
                    performance.get("validation_summary"),
                )

            # Save final model
            _current_phase = "cleanup"
            _rec.start_phase("cleanup")
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
                verification_state=verification_state,
            )
            if not final_model_path:
                return {
                    "status": "error",
                    "message": "Final model save failed (_save_final_model returned empty path)",
                    "checkpoint_dir": checkpoint_dir,
                }

            self._apply_checkpoint_post_success_cleanup(checkpoint_manager)
            _rec.end_phase("cleanup")
            _current_phase = None

            epoch_time_s = time.perf_counter() - epoch_start_time
            self.logger.info("Total epoch time: %.3fs", epoch_time_s)

            return {
                "status": "completed",
                "message": "Resumable training completed successfully",
                "checkpoint_dir": checkpoint_dir,
                "model_path": final_model_path,
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
            try:
                if _current_phase is not None:
                    _rec.fail_phase(_current_phase, e)
            except Exception:
                pass
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

        log_autogluon_timeseries_dataframe_probe(
            ts_df,
            self.logger,
            phase=f"_train_predictor_pre_fit y={year:04d} m={month:02d} branch=initial",
        )

        known_covariates = self.incremental_config.get("known_covariates", [])
        lookback_days = self.incremental_config.get("lookback_days")
        chronos_hyperparameters = self._get_chronos_hyperparameters()

        for env_var in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[env_var] = "1"
            self.logger.info("offline_mode env_var=%s value=1", env_var)

        self.logger.info(
            "Models that will be trained: ['Chronos[%s]']",
            self.chronos_variant,
        )

        if previous_predictor is None:
            # First file - create new predictor
            # ResumableDataLoader maps config target_col (e.g. target_close) to column "target".
            predictor = TimeSeriesPredictor(
                target="target",
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
                previous_data = self.preprocess_raw_dataframe(previous_data)
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

            log_autogluon_timeseries_dataframe_probe(
                combined_data,
                self.logger,
                phase=f"_train_predictor_pre_fit y={year:04d} m={month:02d} branch=combined_window",
            )

            # ResumableDataLoader maps config target_col (e.g. target_close) to column "target".
            predictor = TimeSeriesPredictor(
                target="target",
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
        self._assert_fine_tuned_checkpoint_exists(Path(temp_model_path))
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
        performance: Optional[Dict[str, Any]] = None,
        checkpoint_dir: Optional[str] = None,
        last_year: Optional[int] = None,
        last_month: Optional[int] = None,
        verification_state: Optional[Dict[str, Any]] = None,
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
            if performance is not None:
                metadata["validation"] = {
                    "valid": bool(performance.get("validation_valid", False)),
                    "reason": performance.get("validation_reason", "unknown"),
                    "summary": performance.get("validation_summary", {}),
                }
            verification_state = verification_state or self._new_verification_state()
            metadata["fine_tune_verification"] = {
                "settings": self._get_fine_tune_verification_settings(),
                "dataset_fingerprint": self._build_dataset_fingerprint(
                    verification_state
                ),
                "training_run": {
                    "selected_model": (
                        "Chronos[autogluon__chronos-"
                        f"{self.chronos_variant.replace('_', '-')}]"
                    ),
                    "chronos_model_path": self.chronos_model_path,
                    "fit_runtime_seconds": round(
                        float(verification_state.get("fit_runtime_seconds", 0.0)),
                        6,
                    ),
                    "processed_file_count": len(
                        verification_state.get("processed_files", [])
                    ),
                    "requested_hyperparameters": {
                        "learning_rate": self.learning_rate,
                        "max_epochs": self.max_epochs,
                        "batch_size": self.batch_size,
                        "context_length": self.context_length,
                        "device": self.device,
                    },
                },
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
