"""
Checkpoint Manager for Resumable Training

This module provides checkpoint management functionality for resumable training
on large date ranges. It saves progress after every parquet file is processed
and allows resuming from the last successful checkpoint.
"""

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging

from autogluon.timeseries import TimeSeriesPredictor


class CheckpointManager:
    """Manages checkpoints for resumable training"""

    def __init__(self, checkpoint_dir: str, max_model_checkpoints: Optional[int] = None):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_dir: Directory to store checkpoints
            max_model_checkpoints: If set, prune model_checkpoints/ to this many
                dirs after every successful save. Must be >= 1 when provided.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.checkpoints_dir = self.checkpoint_dir / "checkpoints"
        self.model_checkpoints_dir = self.checkpoint_dir / "model_checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.model_checkpoints_dir.mkdir(exist_ok=True)

        self.max_model_checkpoints = max_model_checkpoints

        self.logger = logging.getLogger("checkpoint_manager")
        self.logger.info(
            "checkpoint_init | checkpoint_dir=%s checkpoints_dir=%s model_checkpoints_dir=%s max_model_checkpoints=%s",
            self.checkpoint_dir,
            self.checkpoints_dir,
            self.model_checkpoints_dir,
            self.max_model_checkpoints,
        )

    def _log_event(self, event: str, **fields: Any) -> None:
        """Emit structured checkpoint logs for AWS debugging."""
        payload = {"event": event, "component": "checkpoint_manager", **fields}
        self.logger.info("checkpoint_event | %s", json.dumps(payload, sort_keys=True, default=str))

    def _required_model_artifacts_present(self, model_dir: Path) -> Tuple[bool, List[str], List[str], int]:
        """
        Validate that a model directory contains a loadable AutoGluon predictor structure.
        """
        required_paths = [
            model_dir / "predictor.pkl",
            model_dir / "learner.pkl",
            model_dir / "models" / "trainer.pkl",
        ]
        missing = [str(p.relative_to(model_dir)) for p in required_paths if not p.exists()]
        files = [p for p in model_dir.rglob("*") if p.is_file()]
        total_bytes = sum(p.stat().st_size for p in files)
        sample = [str(p.relative_to(model_dir)) for p in files[:20]]
        return len(missing) == 0, missing, sample, total_bytes

    def _checkpoint_model_dir(self, year: int, month: int) -> Path:
        """Canonical checkpoint model directory (directory, not .pkl extension)."""
        return self.model_checkpoints_dir / f"model_{year:04d}_{month:02d}"

    def _legacy_checkpoint_model_dir(self, year: int, month: int) -> Path:
        """Legacy checkpoint model path retained for backward compatibility."""
        return self.model_checkpoints_dir / f"model_{year:04d}_{month:02d}.pkl"

    def _resolve_checkpoint_model_path(self, model_path_value: str) -> Optional[Path]:
        """Resolve checkpoint model path across canonical and legacy layouts."""
        primary = Path(model_path_value)
        if primary.exists():
            return primary
        if model_path_value.endswith(".pkl"):
            fallback = Path(model_path_value[:-4])
            if fallback.exists():
                return fallback
        else:
            legacy = Path(f"{model_path_value}.pkl")
            if legacy.exists():
                return legacy
        return None

    def _assert_sufficient_disk_space_for_copy(
        self, source_model_path: Path, destination_root: Path
    ) -> None:
        """
        Fail fast when destination free space is lower than source artifact size.
        """
        required_bytes = sum(
            p.stat().st_size for p in source_model_path.rglob("*") if p.is_file()
        )
        free_bytes = shutil.disk_usage(destination_root).free
        self._log_event(
            "checkpoint_disk_space_check",
            source_model_path=str(source_model_path),
            destination_root=str(destination_root),
            required_bytes=required_bytes,
            free_bytes=free_bytes,
        )
        if free_bytes < required_bytes:
            raise RuntimeError(
                "Insufficient disk space for checkpoint copy: "
                f"required_bytes={required_bytes}, free_bytes={free_bytes}, "
                f"source_model_path={source_model_path}, destination_root={destination_root}"
            )

    def save_checkpoint(
        self,
        year: int,
        month: int,
        model: TimeSeriesPredictor,
        data_stats: Dict[str, Any],
        training_state: Dict[str, Any],
    ) -> bool:
        """
        Save checkpoint after processing a parquet file

        Args:
            year: Year of processed data
            month: Month of processed data
            model: Trained model to save
            data_stats: Statistics about processed data
            training_state: Current training state

        Returns:
            True if save successful, False otherwise
        """
        try:
            # Clean up previous checkpoint
            self._cleanup_previous_checkpoint()

            # Create checkpoint filename
            checkpoint_name = f"checkpoint_{year:04d}_{month:02d}.json"
            model_path = self._checkpoint_model_dir(year, month)
            legacy_model_path = self._legacy_checkpoint_model_dir(year, month)

            # Remove legacy path if present to avoid stale pointer confusion.
            if legacy_model_path.exists() and legacy_model_path != model_path:
                if legacy_model_path.is_dir():
                    for item in legacy_model_path.rglob("*"):
                        if item.is_file():
                            item.unlink()
                    for item in sorted(legacy_model_path.glob("**/*"), reverse=True):
                        if item.is_dir():
                            item.rmdir()
                    legacy_model_path.rmdir()
                else:
                    legacy_model_path.unlink()

            # Save model by copying already-trained predictor artifacts.
            # Changing predictor.path and calling save() here can emit only version metadata.
            source_model_path = Path(getattr(model, "path", "")).resolve()
            if not source_model_path.exists():
                raise RuntimeError(
                    f"Checkpoint source predictor path does not exist: {source_model_path}"
                )
            if model_path.exists():
                shutil.rmtree(model_path)
            self._assert_sufficient_disk_space_for_copy(
                source_model_path, self.model_checkpoints_dir
            )
            self._log_event(
                "checkpoint_model_save_start",
                year=year,
                month=month,
                model_path=str(model_path),
                source_model_path=str(source_model_path),
            )
            shutil.copytree(source_model_path, model_path, dirs_exist_ok=True)
            valid, missing, sample, total_bytes = self._required_model_artifacts_present(model_path)
            self._log_event(
                "checkpoint_model_save_done",
                year=year,
                month=month,
                model_path=str(model_path),
                model_total_bytes=total_bytes,
                missing_required_paths=missing,
                file_sample=sample,
            )
            if not valid:
                raise RuntimeError(
                    f"Checkpoint model artifacts incomplete at {model_path}; missing required paths: {missing}"
                )

            # Create checkpoint data
            checkpoint_data = {
                "year": year,
                "month": month,
                "timestamp": datetime.now().isoformat(),
                "model_path": str(model_path),
                "data_stats": data_stats,
                "training_state": training_state,
                "checkpoint_name": checkpoint_name,
            }

            # Save checkpoint
            checkpoint_path = self.checkpoints_dir / checkpoint_name
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f, indent=2)

            # Save overall training state
            self._save_training_state(training_state)

            # //fixme-max-checkpoint: per-save prune keeps model_checkpoints/ bounded
            # during long date ranges. Replace with streaming upload + pointer-only
            # resume once the checkpoint restore path no longer requires local dirs.
            if self.max_model_checkpoints is not None:
                self.prune_model_checkpoints(self.max_model_checkpoints)

            self.logger.info(f"Checkpoint saved: {checkpoint_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            self._log_event(
                "checkpoint_save_failed",
                year=year,
                month=month,
                error=str(e),
            )
            return False

    def load_checkpoint(self, year: int, month: int) -> Optional[Dict[str, Any]]:
        """
        Load specific checkpoint

        Args:
            year: Year of checkpoint
            month: Month of checkpoint

        Returns:
            Checkpoint data or None if not found
        """
        try:
            checkpoint_name = f"checkpoint_{year:04d}_{month:02d}.json"
            checkpoint_path = self.checkpoints_dir / checkpoint_name

            if not checkpoint_path.exists():
                return None

            with open(checkpoint_path, "r") as f:
                checkpoint_data = json.load(f)

            # Load model
            model_path = checkpoint_data["model_path"]
            resolved_model_path = self._resolve_checkpoint_model_path(model_path)
            self._log_event(
                "checkpoint_load_attempt",
                checkpoint_name=checkpoint_name,
                model_path=model_path,
                resolved_model_path=str(resolved_model_path) if resolved_model_path else None,
            )
            if resolved_model_path and resolved_model_path.exists():
                model = TimeSeriesPredictor.load(str(resolved_model_path))
                checkpoint_data["model"] = model
            else:
                self._log_event(
                    "checkpoint_load_missing_model",
                    checkpoint_name=checkpoint_name,
                    model_path=model_path,
                )

            self.logger.info(f"Checkpoint loaded: {checkpoint_name}")
            return checkpoint_data

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None

    def get_last_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent checkpoint

        Returns:
            Latest checkpoint data or None if no checkpoints exist
        """
        try:
            # Find all checkpoint files
            checkpoint_files = list(self.checkpoints_dir.glob("checkpoint_*.json"))

            if not checkpoint_files:
                return None

            # Sort by modification time (newest first)
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Load the most recent checkpoint
            latest_checkpoint = checkpoint_files[0]
            with open(latest_checkpoint, "r") as f:
                checkpoint_data = json.load(f)

            # Load model
            model_path = checkpoint_data["model_path"]
            resolved_model_path = self._resolve_checkpoint_model_path(model_path)
            self._log_event(
                "last_checkpoint_load_attempt",
                checkpoint_name=latest_checkpoint.name,
                model_path=model_path,
                resolved_model_path=str(resolved_model_path) if resolved_model_path else None,
            )
            if resolved_model_path and resolved_model_path.exists():
                model = TimeSeriesPredictor.load(str(resolved_model_path))
                checkpoint_data["model"] = model
            else:
                self._log_event(
                    "last_checkpoint_model_missing",
                    checkpoint_name=latest_checkpoint.name,
                    model_path=model_path,
                )

            self.logger.info(f"Last checkpoint loaded: {latest_checkpoint.name}")
            return checkpoint_data

        except Exception as e:
            self.logger.error(f"Failed to get last checkpoint: {e}")
            return None

    def cleanup_old_checkpoints(self) -> None:
        """Remove all checkpoints (keep only latest)"""
        try:
            # Find all checkpoint files
            checkpoint_files = list(self.checkpoints_dir.glob("checkpoint_*.json"))

            if len(checkpoint_files) <= 1:
                return  # Keep at least one checkpoint

            # Sort by modification time (newest first)
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Remove all but the latest
            for checkpoint_file in checkpoint_files[1:]:
                checkpoint_file.unlink()
                self.logger.info(f"Removed old checkpoint: {checkpoint_file.name}")

        except Exception as e:
            self.logger.error(f"Failed to cleanup old checkpoints: {e}")

    def get_training_progress(self) -> Dict[str, Any]:
        """
        Get current training progress

        Returns:
            Training progress information
        """
        try:
            last_checkpoint = self.get_last_checkpoint()

            if not last_checkpoint:
                return {
                    "status": "not_started",
                    "last_processed": None,
                    "total_checkpoints": 0,
                }

            # Count total checkpoints
            checkpoint_files = list(self.checkpoints_dir.glob("checkpoint_*.json"))

            return {
                "status": "in_progress",
                "last_processed": f"{last_checkpoint['year']:04d}-{last_checkpoint['month']:02d}",
                "total_checkpoints": len(checkpoint_files),
                "last_checkpoint_time": last_checkpoint.get("timestamp"),
            }

        except Exception as e:
            self.logger.error(f"Failed to get training progress: {e}")
            return {"status": "error", "error": str(e)}

    def _cleanup_previous_checkpoint(self) -> None:
        """Remove previous checkpoint before saving new one"""
        try:
            checkpoint_files = list(self.checkpoints_dir.glob("checkpoint_*.json"))
            for checkpoint_file in checkpoint_files:
                checkpoint_file.unlink()

        except Exception as e:
            self.logger.warning(f"Failed to cleanup previous checkpoint: {e}")

    def _save_training_state(self, training_state: Dict[str, Any]) -> None:
        """Save overall training state"""
        try:
            state_path = self.checkpoint_dir / "training_state.json"
            with open(state_path, "w") as f:
                json.dump(training_state, f, indent=2)

        except Exception as e:
            self.logger.warning(f"Failed to save training state: {e}")

    def load_training_state(self) -> Optional[Dict[str, Any]]:
        """Load overall training state"""
        try:
            state_path = self.checkpoint_dir / "training_state.json"
            if state_path.exists():
                with open(state_path, "r") as f:
                    return json.load(f)
            return None

        except Exception as e:
            self.logger.error(f"Failed to load training state: {e}")
            return None

    _MODEL_DIR_RE = re.compile(r"^model_(\d{4})_(\d{2})$")

    def remove_temp_directory(self) -> None:
        """Remove checkpoint_dir/temp (AutoGluon scratch trees). Idempotent."""
        temp_path = self.checkpoint_dir / "temp"
        if not temp_path.exists():
            self._log_event(
                "checkpoint_cleanup_temp",
                reason="skipped_missing",
                path=str(temp_path),
            )
            return
        try:
            shutil.rmtree(temp_path)
            self._log_event(
                "checkpoint_cleanup_temp",
                reason="deleted",
                path=str(temp_path),
            )
        except OSError as e:
            self._log_event(
                "checkpoint_cleanup_temp",
                reason="failed",
                path=str(temp_path),
                error=str(e),
            )
            raise

    def remove_temp_model_directory(self, year: int, month: int) -> None:
        """Remove checkpoint_dir/temp/temp_model_YYYY_MM. Idempotent."""
        temp_model_path = (
            self.checkpoint_dir / "temp" / f"temp_model_{year:04d}_{month:02d}"
        )
        if not temp_model_path.exists():
            self._log_event(
                "checkpoint_cleanup_temp_model",
                reason="skipped_missing",
                path=str(temp_model_path),
                year=year,
                month=month,
            )
            return
        try:
            shutil.rmtree(temp_model_path)
            self._log_event(
                "checkpoint_cleanup_temp_model",
                reason="deleted",
                path=str(temp_model_path),
                year=year,
                month=month,
            )
        except OSError as e:
            self._log_event(
                "checkpoint_cleanup_temp_model",
                reason="failed",
                path=str(temp_model_path),
                year=year,
                month=month,
                error=str(e),
            )
            raise

    def _list_sorted_model_checkpoint_dirs(self) -> List[Tuple[Tuple[int, int], Path]]:
        """Canonical monthly dirs model_YYYY_MM (directories only)."""
        found: List[Tuple[Tuple[int, int], Path]] = []
        if not self.model_checkpoints_dir.exists():
            return found
        for entry in self.model_checkpoints_dir.iterdir():
            if not entry.is_dir():
                continue
            match = self._MODEL_DIR_RE.match(entry.name)
            if not match:
                continue
            year, month = int(match.group(1)), int(match.group(2))
            found.append(((year, month), entry))
        found.sort(key=lambda x: x[0])
        return found

    def prune_model_checkpoints(self, keep_n: int) -> None:
        """
        Keep the N most recent model_YYYY_MM directories; remove older ones.

        Legacy single-file paths (model_YYYY_MM.pkl) are not removed here.
        """
        if keep_n < 1:
            self._log_event(
                "checkpoint_prune_model_checkpoints",
                reason="skipped_invalid_keep_n",
                keep_n=keep_n,
            )
            return

        sorted_dirs = self._list_sorted_model_checkpoint_dirs()
        if len(sorted_dirs) <= keep_n:
            self._log_event(
                "checkpoint_prune_model_checkpoints",
                reason="skipped_no_prune_needed",
                keep_n=keep_n,
                dir_count=len(sorted_dirs),
            )
            return

        to_remove = sorted_dirs[: len(sorted_dirs) - keep_n]
        for (_ym, path) in to_remove:
            try:
                shutil.rmtree(path)
                self._log_event(
                    "checkpoint_prune_model_checkpoints",
                    reason="deleted",
                    path=str(path),
                    keep_n=keep_n,
                )
            except OSError as e:
                self._log_event(
                    "checkpoint_prune_model_checkpoints",
                    reason="failed",
                    path=str(path),
                    error=str(e),
                )
                raise
