"""
Model versioning utilities for incremental training
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import shutil
import logging


class ModelVersioning:
    """Handles model versioning, metadata tracking, and cleanup"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("model_versioning")

        # Versioning configuration
        self.model_base_path = Path(config.get("model_path", "data/models/incremental"))
        self.max_versions = config.get("max_versions", 10)

        # Version tracking
        self.model_versions = {}
        self.current_version = None
        self.previous_version = None
        self.performance_history = {}

        # Ensure base path exists
        self.model_base_path.mkdir(parents=True, exist_ok=True)

    def generate_version_id(self, date_range: Tuple[str, str]) -> str:
        """Generate timestamped version ID for model versioning"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_date = date_range[0].replace("-", "")[:8]  # YYYYMMDD
        end_date = date_range[1].replace("-", "")[:8]  # YYYYMMDD
        return f"model_{start_date}_{end_date}_{timestamp}"

    def save_model_version(
        self,
        predictor,
        version_id: str,
        date_range: Tuple[str, str],
        performance_metrics: Dict[str, float],
        model_config: Dict[str, Any],
        covariate_config: Dict[str, Any],
    ) -> str:
        """Save model version with metadata"""
        try:
            # Create version directory
            version_dir = self.model_base_path / version_id
            version_dir.mkdir(parents=True, exist_ok=True)

            # Save model - AutoGluon saves to its own path, we'll move it
            predictor.save()
            # Get the actual save path from AutoGluon
            autogluon_path = predictor.path
            model_path = str(version_dir)

            # Move the saved model to our version directory
            import shutil

            if Path(autogluon_path).exists():
                shutil.move(autogluon_path, model_path)

            # Save metadata
            metadata = {
                "version_id": version_id,
                "date_range": date_range,
                "performance_metrics": performance_metrics,
                "training_timestamp": datetime.now().isoformat(),
                "model_config": model_config,
                "covariate_config": covariate_config,
            }

            metadata_path = version_dir / "training_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Model version saved to: {model_path}")
            return model_path

        except Exception as e:
            self.logger.error(f"Failed to save model version: {e}")
            raise

    def update_version_tracking(
        self,
        version_id: str,
        model_path: str,
        date_range: Tuple[str, str],
        performance_metrics: Dict[str, float],
    ):
        """Update version tracking information"""
        self.previous_version = self.current_version
        self.current_version = version_id

        self.model_versions[version_id] = {
            "model_path": model_path,
            "date_range": date_range,
            "performance_metrics": performance_metrics,
            "created_at": datetime.now().isoformat(),
        }

        self.performance_history[version_id] = performance_metrics

    def get_previous_performance(self, previous_model_path: str) -> Dict[str, float]:
        """Get performance metrics from previous model"""
        try:
            # Load previous model metadata
            metadata_path = Path(previous_model_path) / "training_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    return metadata.get("performance_metrics", {})

            # Fallback to default values
            return {"mae": 1.0, "rmse": 1.0, "mase": 1.0, "directional_accuracy": 0.5}

        except Exception as e:
            self.logger.warning(f"Failed to get previous performance: {e}")
            return {"mae": 1.0, "rmse": 1.0, "mase": 1.0, "directional_accuracy": 0.5}

    def calculate_performance_improvement(
        self, current_metrics: Dict[str, float], previous_metrics: Dict[str, float]
    ) -> float:
        """Calculate performance improvement over previous model"""
        try:
            # Use MAE as primary metric for improvement calculation
            current_mae = current_metrics.get("mae", float("inf"))
            previous_mae = previous_metrics.get("mae", 1.0)

            if previous_mae == 0:
                return 0.0

            improvement = (previous_mae - current_mae) / previous_mae
            return max(0.0, improvement)  # Ensure non-negative improvement

        except Exception as e:
            self.logger.warning(f"Failed to calculate performance improvement: {e}")
            return 0.0

    def cleanup_old_versions(self):
        """Clean up old model versions to prevent storage bloat"""
        try:
            if len(self.model_versions) <= self.max_versions:
                return

            # Sort versions by creation time and remove oldest
            sorted_versions = sorted(
                self.model_versions.items(), key=lambda x: x[1]["created_at"]
            )

            versions_to_remove = len(self.model_versions) - self.max_versions
            for i in range(versions_to_remove):
                version_id, version_info = sorted_versions[i]

                # Remove directory
                model_path = Path(version_info["model_path"])
                if model_path.parent.exists():
                    shutil.rmtree(model_path.parent)

                # Remove from tracking
                del self.model_versions[version_id]
                if version_id in self.performance_history:
                    del self.performance_history[version_id]

                self.logger.info(f"Cleaned up old version: {version_id}")

        except Exception as e:
            self.logger.warning(f"Failed to cleanup old versions: {e}")

    def get_version_history(self) -> Dict[str, Any]:
        """Get complete version history and performance tracking"""
        return {
            "current_version": self.current_version,
            "previous_version": self.previous_version,
            "model_versions": self.model_versions,
            "performance_history": self.performance_history,
            "total_versions": len(self.model_versions),
        }

    def list_available_versions(self) -> List[Dict[str, Any]]:
        """List all available model versions"""
        versions = []
        for version_id, info in self.model_versions.items():
            versions.append(
                {
                    "version_id": version_id,
                    "date_range": info["date_range"],
                    "performance_metrics": info["performance_metrics"],
                    "created_at": info["created_at"],
                    "is_current": version_id == self.current_version,
                }
            )

        return sorted(versions, key=lambda x: x["created_at"], reverse=True)

    def switch_to_version(self, version_id: str) -> bool:
        """Switch to a specific model version"""
        try:
            if version_id not in self.model_versions:
                self.logger.error(f"Version {version_id} not found")
                return False

            # Update current version
            self.previous_version = self.current_version
            self.current_version = version_id

            self.logger.info(f"Switched to version: {version_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to switch to version {version_id}: {e}")
            return False

    def rollback_to_previous(self, failed_version_id: str) -> Dict[str, Any]:
        """Rollback to previous model version"""
        try:
            self.logger.info(f"Rolling back from failed version {failed_version_id}")

            # Remove failed version
            if failed_version_id in self.model_versions:
                del self.model_versions[failed_version_id]

            # Update current version to previous
            if self.previous_version:
                self.current_version = self.previous_version
                self.logger.info(f"Rolled back to version: {self.current_version}")

            return {
                "success": False,
                "rollback": True,
                "current_version": self.current_version,
                "failed_version": failed_version_id,
                "message": "Model performance below threshold, rolled back to previous version",
            }

        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            raise
