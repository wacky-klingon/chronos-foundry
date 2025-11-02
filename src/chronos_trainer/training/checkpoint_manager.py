"""
Checkpoint Manager for Resumable Training

This module provides checkpoint management functionality for resumable training
on large date ranges. It saves progress after every parquet file is processed
and allows resuming from the last successful checkpoint.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

from autogluon.timeseries import TimeSeriesPredictor


class CheckpointManager:
    """Manages checkpoints for resumable training"""

    def __init__(self, checkpoint_dir: str):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.checkpoints_dir = self.checkpoint_dir / "checkpoints"
        self.model_checkpoints_dir = self.checkpoint_dir / "model_checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.model_checkpoints_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger("checkpoint_manager")

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
            model_name = f"model_{year:04d}_{month:02d}.pkl"

            # Save model
            model_path = self.model_checkpoints_dir / model_name
            # Set the path before saving (AutoGluon saves to predictor.path)
            model.path = str(model_path)
            model.save()

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

            self.logger.info(f"Checkpoint saved: {checkpoint_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
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
            if os.path.exists(model_path):
                model = TimeSeriesPredictor.load(model_path)
                checkpoint_data["model"] = model

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
            if os.path.exists(model_path):
                model = TimeSeriesPredictor.load(model_path)
                checkpoint_data["model"] = model

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
