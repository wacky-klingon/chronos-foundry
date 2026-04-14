"""
Tests for CheckpointManager functionality

Tests checkpoint saving, loading, and progress tracking.
"""

import pytest
import tempfile
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from chronos_trainer.training.checkpoint_manager import CheckpointManager


def _minimal_autogluon_predictor_dir(parent: Path) -> str:
    """Directory layout required by CheckpointManager.save_checkpoint copy + validation."""
    d = parent / "fake_predictor"
    d.mkdir(parents=True)
    (d / "predictor.pkl").write_bytes(b"x")
    (d / "learner.pkl").write_bytes(b"x")
    (d / "models").mkdir()
    (d / "models" / "trainer.pkl").write_bytes(b"x")
    return str(d)


class TestCheckpointManager:
    """Test CheckpointManager core functionality"""

    @pytest.fixture
    def checkpoint_dir(self):
        """Create temporary checkpoint directory"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def checkpoint_manager(self, checkpoint_dir):
        """Create CheckpointManager instance"""
        return CheckpointManager(checkpoint_dir)

    @pytest.fixture
    def mock_predictor(self, tmp_path):
        """Mock predictor with a real on-disk path (checkpoint copies artifacts)."""
        mock = MagicMock()
        mock.path = _minimal_autogluon_predictor_dir(tmp_path)
        mock.save = MagicMock()
        return mock

    def test_checkpoint_save_and_load(self, checkpoint_manager, mock_predictor):
        """Test saving and loading checkpoints with TimeSeriesPredictor"""
        data_stats = {
            "record_count": 100,
            "columns": ["timestamp", "target", "item_id"],
            "memory_usage_mb": 1.5,
        }

        training_state = {
            "start_date": "2020-01-01",
            "end_date": "2020-02-28",
            "processed_files": [
                {"file_path": "test.parquet", "year": 2020, "month": 1}
            ],
        }

        # Save checkpoint
        success = checkpoint_manager.save_checkpoint(
            year=2020,
            month=1,
            model=mock_predictor,
            data_stats=data_stats,
            training_state=training_state,
        )

        assert success, "Checkpoint save should succeed"

        # Verify checkpoint file exists
        checkpoint_file = (
            checkpoint_manager.checkpoints_dir / "checkpoint_2020_01.json"
        )
        assert checkpoint_file.exists(), "Checkpoint file should exist"

        # Verify checkpoint JSON structure
        with open(checkpoint_file, "r") as f:
            checkpoint_data = json.load(f)

        assert checkpoint_data["year"] == 2020
        assert checkpoint_data["month"] == 1
        assert checkpoint_data["data_stats"]["record_count"] == 100
        assert checkpoint_data["training_state"]["start_date"] == "2020-01-01"

        # Load checkpoint (without loading model since it's mocked)
        with patch("chronos_trainer.training.checkpoint_manager.TimeSeriesPredictor") as mock_load:
            loaded_checkpoint = checkpoint_manager.load_checkpoint(2020, 1)

            # Should return checkpoint data even if model load fails
            assert loaded_checkpoint is not None
            assert loaded_checkpoint["year"] == 2020
            assert loaded_checkpoint["month"] == 1
            assert loaded_checkpoint["data_stats"]["record_count"] == 100

    def test_get_last_checkpoint_none(self, checkpoint_manager):
        """Test getting last checkpoint when none exist"""
        last_checkpoint = checkpoint_manager.get_last_checkpoint()
        assert last_checkpoint is None, "Should return None when no checkpoints"

    @patch(
        "chronos_trainer.training.checkpoint_manager.TimeSeriesPredictor.load",
        return_value=MagicMock(),
    )
    def test_get_last_checkpoint_single(self, mock_load, checkpoint_manager, mock_predictor):
        """Test getting last checkpoint when one exists"""
        data_stats = {"record_count": 100}
        training_state = {"start_date": "2020-01-01", "end_date": "2020-01-31"}

        checkpoint_manager.save_checkpoint(
            year=2020,
            month=1,
            model=mock_predictor,
            data_stats=data_stats,
            training_state=training_state,
        )

        last_checkpoint = checkpoint_manager.get_last_checkpoint()
        assert last_checkpoint is not None
        assert last_checkpoint["year"] == 2020
        assert last_checkpoint["month"] == 1

    @patch(
        "chronos_trainer.training.checkpoint_manager.TimeSeriesPredictor.load",
        return_value=MagicMock(),
    )
    def test_get_last_checkpoint_multiple(self, mock_load, checkpoint_manager, mock_predictor):
        """Test getting most recent checkpoint when multiple exist"""
        data_stats = {"record_count": 100}
        training_state = {"start_date": "2020-01-01", "end_date": "2020-03-31"}

        # Save multiple checkpoints with slight delay
        checkpoint_manager.save_checkpoint(
            year=2020,
            month=1,
            model=mock_predictor,
            data_stats=data_stats,
            training_state=training_state,
        )

        import time

        time.sleep(0.1)  # Ensure different timestamps

        checkpoint_manager.save_checkpoint(
            year=2020,
            month=2,
            model=mock_predictor,
            data_stats=data_stats,
            training_state=training_state,
        )

        last_checkpoint = checkpoint_manager.get_last_checkpoint()
        assert last_checkpoint is not None
        assert last_checkpoint["month"] == 2, "Should return most recent checkpoint"

    def test_training_progress_not_started(self, checkpoint_manager):
        """Test training progress when no checkpoints exist"""
        progress = checkpoint_manager.get_training_progress()

        assert progress["status"] == "not_started"
        assert progress["last_processed"] is None
        assert progress["total_checkpoints"] == 0

    @patch(
        "chronos_trainer.training.checkpoint_manager.TimeSeriesPredictor.load",
        return_value=MagicMock(),
    )
    def test_training_progress_in_progress(self, mock_load, checkpoint_manager, mock_predictor):
        """Test training progress when checkpoints exist"""
        data_stats = {"record_count": 100}
        training_state = {"start_date": "2020-01-01", "end_date": "2020-02-28"}

        checkpoint_manager.save_checkpoint(
            year=2020,
            month=1,
            model=mock_predictor,
            data_stats=data_stats,
            training_state=training_state,
        )

        progress = checkpoint_manager.get_training_progress()

        assert progress["status"] == "in_progress"
        assert progress["last_processed"] == "2020-01"
        assert progress["total_checkpoints"] == 1
        assert "last_checkpoint_time" in progress

    def test_remove_temp_directory_idempotent(self, checkpoint_manager, checkpoint_dir):
        """temp/ is removed when present; missing temp is a no-op."""
        temp = Path(checkpoint_dir) / "temp"
        temp.mkdir(parents=True)
        (temp / "stub.txt").write_text("x", encoding="utf-8")
        checkpoint_manager.remove_temp_directory()
        assert not temp.exists()
        checkpoint_manager.remove_temp_directory()

    def test_prune_model_checkpoints_keeps_last_n(self, checkpoint_manager, checkpoint_dir):
        """Keep N most recent model_YYYY_MM dirs; remove older."""
        mdir = Path(checkpoint_dir) / "model_checkpoints"
        for y, m in [(2006, 1), (2006, 2), (2006, 3)]:
            d = mdir / f"model_{y:04d}_{m:02d}"
            d.mkdir(parents=True)
            (d / "marker.txt").write_text(str(m), encoding="utf-8")

        checkpoint_manager.prune_model_checkpoints(1)
        assert (mdir / "model_2006_03").exists()
        assert not (mdir / "model_2006_01").exists()
        assert not (mdir / "model_2006_02").exists()

    def test_prune_model_checkpoints_keep_two(self, checkpoint_manager, checkpoint_dir):
        mdir = Path(checkpoint_dir) / "model_checkpoints"
        for y, m in [(2006, 1), (2006, 2), (2006, 3)]:
            d = mdir / f"model_{y:04d}_{m:02d}"
            d.mkdir(parents=True)
            (d / "marker.txt").write_text(str(m), encoding="utf-8")

        checkpoint_manager.prune_model_checkpoints(2)
        assert (mdir / "model_2006_02").exists()
        assert (mdir / "model_2006_03").exists()
        assert not (mdir / "model_2006_01").exists()

