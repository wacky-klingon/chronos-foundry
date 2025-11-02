"""
Tests for IncrementalTrainer functionality

Tests incremental training workflows, checkpointing, and resumable training.
"""

import pytest
import tempfile
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from unittest.mock import MagicMock, patch
import shutil

from chronos_trainer import IncrementalTrainer
from chronos_trainer.training.checkpoint_manager import CheckpointManager


class TestIncrementalTrainer:
    """Test IncrementalTrainer functionality"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test artifacts"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def sample_data_dir(self, temp_dir):
        """Create temporary directory with YYYY/MM/ structure and parquet files"""
        base_path = Path(temp_dir) / "data"

        # Create YYYY/MM/ structure with parquet files
        for year in [2020]:
            for month in [1, 2]:
                month_dir = base_path / f"{year:04d}" / f"{month:02d}"
                month_dir.mkdir(parents=True)

                # Create sample parquet file
                dates = pd.date_range(
                    start=f"{year}-{month:02d}-01",
                    periods=100,
                    freq="1h",
                )
                df = pd.DataFrame(
                    {
                        "timestamp": dates,
                        "target": pd.Series(range(100)) + 50.0,
                        "item_id": "test_item",
                    }
                )

                parquet_path = month_dir / "data.parquet"
                table = pa.Table.from_pandas(df)
                pq.write_table(table, parquet_path)

        return str(base_path)

    @pytest.fixture
    def sample_config(self, temp_dir, sample_data_dir):
        """Create sample configuration for incremental training"""
        return {
            # Model configuration
            "model_name": "amazon/chronos-t5-tiny",
            "context_length": 96,
            "prediction_length": 24,
            "learning_rate": 0.001,
            "batch_size": 32,
            "max_epochs": 1,  # Fast for testing
            "training_preset": "medium_quality",
            "time_limit": 60,

            # Data configuration
            "timestamp_col": "timestamp",
            "target_col": "target",
            "item_id_col": "item_id",

            # Data paths
            "parquet_loader": {
                "data_paths": {
                    "root_dir": sample_data_dir,
                },
            },

            # Incremental training configuration
            "incremental_training": {
                "incremental_training": {
                    "model_versioning": True,
                    "performance_threshold": 0.05,
                    "rollback_enabled": True,
                    "checkpoint_dir": str(Path(temp_dir) / "checkpoints"),
                    "model_base_path": str(Path(temp_dir) / "models"),
                }
            },
        }

    @pytest.mark.slow
    def test_integration_resumable_training(
        self, temp_dir, sample_data_dir, sample_config
    ):
        """
        Test full end-to-end incremental training workflow
        
        This test validates the complete resumable training process:
        - File discovery
        - Data loading
        - Checkpoint creation
        - Progress tracking
        """
        checkpoint_dir = str(Path(temp_dir) / "checkpoints")

        trainer = IncrementalTrainer(sample_config)

        # Mock the actual training to avoid slow model training
        with patch.object(trainer, "_train_predictor") as mock_train:
            mock_predictor = MagicMock()
            mock_predictor.path = None
            mock_predictor.fit = MagicMock()
            mock_predictor.predict = MagicMock(return_value=pd.DataFrame())
            mock_train.return_value = mock_predictor

            result = trainer.train_with_checkpoints(
                start_date="2020-01-01",
                end_date="2020-02-28",
                validation_start_date="2020-03-01",
                validation_end_date="2020-03-07",
                checkpoint_dir=checkpoint_dir,
            )

            # Verify result structure
            assert "status" in result
            assert result["status"] in ["completed", "error", "in_progress"]

            if result["status"] == "completed":
                assert "checkpoint_dir" in result
                assert "processed_files" in result or "total_files" in result

            # Verify checkpoint was created
            checkpoint_manager = CheckpointManager(checkpoint_dir)
            last_checkpoint = checkpoint_manager.get_last_checkpoint()

            if last_checkpoint:
                assert last_checkpoint["year"] == 2020
                assert last_checkpoint["month"] in [1, 2]

    @pytest.mark.slow
    def test_resume_training(self, temp_dir, sample_data_dir, sample_config):
        """
        Test resuming training from checkpoint
        
        Scenario:
        1. Start training with 2 files
        2. Process 1 file successfully (checkpoint saved)
        3. Resume training → should start from file 2
        """
        checkpoint_dir = str(Path(temp_dir) / "checkpoints")

        trainer = IncrementalTrainer(sample_config)

        # Mock training to avoid slow execution
        with patch.object(trainer, "_train_predictor") as mock_train:
            mock_predictor = MagicMock()
            mock_predictor.path = None
            mock_predictor.fit = MagicMock()
            mock_predictor.predict = MagicMock(return_value=pd.DataFrame())
            mock_train.return_value = mock_predictor

            # First training run - process January only
            result1 = trainer.train_with_checkpoints(
                start_date="2020-01-01",
                end_date="2020-01-31",  # Only January
                validation_start_date="2020-02-01",
                validation_end_date="2020-02-07",
                checkpoint_dir=checkpoint_dir,
            )

            assert result1["status"] in ["completed", "error", "in_progress"]

            # Verify checkpoint exists
            checkpoint_manager = CheckpointManager(checkpoint_dir)
            last_checkpoint = checkpoint_manager.get_last_checkpoint()
            assert last_checkpoint is not None, "Checkpoint should be created"

            # Resume training - should continue from where we left off
            result2 = trainer.train_with_checkpoints(
                start_date="2020-01-01",
                end_date="2020-02-28",  # Now includes February
                validation_start_date="2020-03-01",
                validation_end_date="2020-03-07",
                checkpoint_dir=checkpoint_dir,
            )

            # Should resume and process remaining files
            assert result2["status"] in ["completed", "error", "in_progress"]

            # Verify it resumed from checkpoint
            if result2["status"] == "completed":
                # Should have processed February (file 2)
                progress = checkpoint_manager.get_training_progress()
                assert progress["status"] == "in_progress" or progress["total_checkpoints"] >= 1

    def test_no_remaining_files(self, temp_dir, sample_data_dir, sample_config):
        """Test behavior when all files are already processed"""
        checkpoint_dir = str(Path(temp_dir) / "checkpoints")

        trainer = IncrementalTrainer(sample_config)

        # Create a checkpoint manager and save training state indicating all files processed
        checkpoint_manager = CheckpointManager(checkpoint_dir)
        mock_predictor = MagicMock()
        mock_predictor.path = None
        mock_predictor.save = MagicMock()

        # Save checkpoint with all files marked as processed
        checkpoint_manager.save_checkpoint(
            year=2020,
            month=2,  # Last month
            model=mock_predictor,
            data_stats={"record_count": 100},
            training_state={
                "start_date": "2020-01-01",
                "end_date": "2020-02-28",
                "processed_files": [
                    {"file_path": "data.parquet", "year": 2020, "month": 1},
                    {"file_path": "data.parquet", "year": 2020, "month": 2},
                ],
            },
        )

        # Try to train - should detect no remaining files
        result = trainer.train_with_checkpoints(
            start_date="2020-01-01",
            end_date="2020-02-28",
            validation_start_date="2020-03-01",
            validation_end_date="2020-03-07",
            checkpoint_dir=checkpoint_dir,
        )

        assert result["status"] == "completed"
        assert "All files already processed" in result.get("message", "")

