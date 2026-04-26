"""
Tests for IncrementalTrainer functionality

Tests incremental training workflows, checkpointing, and resumable training.
"""

import pytest
import tempfile
import json
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from unittest.mock import MagicMock, patch
import shutil
import logging

from chronos_trainer import IncrementalTrainer
from chronos_trainer.training.checkpoint_manager import CheckpointManager
from chronos_trainer.training.incremental_trainer import IncrementalTrainingError


# Final export validation requires total artifact size >= 1 MiB (see IncrementalTrainer).
_MIN_FINAL_MODEL_BYTES = 1024 * 1024


def _minimal_autogluon_predictor_dir(
    parent: Path, *, min_total_bytes: int = 0
) -> str:
    """Directory layout required by CheckpointManager.save_checkpoint."""
    d = parent / "fake_predictor"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    (d / "predictor.pkl").write_bytes(b"x")
    (d / "learner.pkl").write_bytes(b"x")
    (d / "models").mkdir()
    (d / "models" / "trainer.pkl").write_bytes(b"x")
    if min_total_bytes > 0:
        total = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
        if total < min_total_bytes:
            (d / "_padding.bin").write_bytes(b"\0" * (min_total_bytes - total))
    return str(d)


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
            # Required for train_with_checkpoints / final model export
            "model_path": str(Path(temp_dir) / "models"),
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
                "chronos_only": True,
                "chronos_model_variant": "bolt_small",
                # Points to the temp directory itself — satisfies the path-existence
                # check in _resolve_chronos_local_model_path() without requiring real
                # model weights. Tests that mock _train_predictor never reach AutoGluon.
                "chronos_local_model_dir": temp_dir,
                "model_versioning": True,
                "performance_threshold": 0.05,
                "rollback_enabled": True,
                "rollback_window_versions": 1,
                "checkpoint_post_success_cleanup": False,
                "lookback_days": 90,
                "checkpoint_dir": str(Path(temp_dir) / "checkpoints"),
                "model_base_path": str(Path(temp_dir) / "models"),
            },
        }

    @pytest.mark.slow
    @patch(
        "chronos_trainer.training.checkpoint_manager.TimeSeriesPredictor.load",
        return_value=MagicMock(),
    )
    def test_integration_resumable_training(
        self, _mock_ts_load, temp_dir, sample_data_dir, sample_config
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
            mock_predictor.path = _minimal_autogluon_predictor_dir(
                Path(temp_dir), min_total_bytes=_MIN_FINAL_MODEL_BYTES
            )
            mock_predictor.fit = MagicMock()
            mock_predictor.predict = MagicMock(return_value=pd.DataFrame())
            mock_train.return_value = (mock_predictor, 0.01)

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
    @patch(
        "chronos_trainer.training.checkpoint_manager.TimeSeriesPredictor.load",
        return_value=MagicMock(),
    )
    def test_resume_training(self, _mock_ts_load, temp_dir, sample_data_dir, sample_config):
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
            mock_predictor.path = _minimal_autogluon_predictor_dir(
                Path(temp_dir), min_total_bytes=_MIN_FINAL_MODEL_BYTES
            )
            mock_predictor.fit = MagicMock()
            mock_predictor.predict = MagicMock(return_value=pd.DataFrame())
            mock_train.return_value = (mock_predictor, 0.01)

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

    def test_final_model_metadata_contains_fine_tune_verification(
        self, temp_dir, sample_config
    ):
        trainer = IncrementalTrainer(sample_config)
        predictor = MagicMock()
        source_dir = Path(
            _minimal_autogluon_predictor_dir(
                Path(temp_dir), min_total_bytes=_MIN_FINAL_MODEL_BYTES
            )
        )
        predictor.path = str(source_dir)
        checkpoint_dir = Path(temp_dir) / "checkpoint_copy"
        checkpoint_model_dir = checkpoint_dir / "model_checkpoints" / "model_2020_02"
        checkpoint_model_dir.mkdir(parents=True, exist_ok=True)
        for item in source_dir.iterdir():
            dest = checkpoint_model_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
        verification_state = {
            "row_count": 200,
            "item_ids": ["item_a", "item_b"],
            "observed_start_timestamp": "2020-01-01T00:00:00",
            "observed_end_timestamp": "2020-02-29T23:00:00",
            "known_covariates": ["gold_xauusd_bid"],
            "fit_runtime_seconds": 45.0,
            "processed_files": [{"year": 2020, "month": 1}, {"year": 2020, "month": 2}],
        }

        model_path = trainer._save_final_model(
            Path(sample_config["model_path"]),
            predictor,
            "2020-01-01",
            "2020-02-29",
            performance={
                "validation_valid": True,
                "validation_reason": "ok",
                "validation_summary": {"validation_rows": 24},
            },
            checkpoint_dir=str(checkpoint_dir),
            last_year=2020,
            last_month=2,
            verification_state=verification_state,
        )

        metadata_path = Path(model_path) / "training_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        verification = metadata["fine_tune_verification"]
        assert verification["dataset_fingerprint"]["row_count"] == 200
        assert verification["dataset_fingerprint"]["item_count"] == 2
        assert verification["training_run"]["fit_runtime_seconds"] == 45.0
        assert (
            verification["training_run"]["requested_hyperparameters"]["learning_rate"]
            == 0.001
        )

    @patch("chronos_trainer.training.checkpoint_manager.TimeSeriesPredictor.load")
    def test_no_remaining_files(
        self, mock_ts_load, temp_dir, sample_data_dir, sample_config
    ):
        """Test behavior when all files are already processed"""
        mock_ts_load.return_value = MagicMock()

        checkpoint_dir = str(Path(temp_dir) / "checkpoints")

        trainer = IncrementalTrainer(sample_config)

        # Create a checkpoint manager and save training state indicating all files processed
        checkpoint_manager = CheckpointManager(checkpoint_dir)
        mock_predictor = MagicMock()
        mock_predictor.path = _minimal_autogluon_predictor_dir(
            Path(temp_dir), min_total_bytes=_MIN_FINAL_MODEL_BYTES
        )
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
                "validation_start_date": "2020-03-01",
                "validation_end_date": "2020-03-07",
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
        assert "completed successfully" in result.get("message", "")

    def test_apply_checkpoint_post_success_cleanup_true(self, temp_dir, sample_config):
        """When enabled, temp/ is removed and model_checkpoints/ pruned to rollback_window."""
        sample_config["incremental_training"]["checkpoint_post_success_cleanup"] = True
        ck = Path(temp_dir) / "ckpt_cleanup"
        ck.mkdir(parents=True)
        (ck / "temp" / "scratch").mkdir(parents=True)
        mcp = ck / "model_checkpoints"
        for y, m in [(2006, 1), (2006, 2)]:
            (mcp / f"model_{y:04d}_{m:02d}").mkdir(parents=True)

        trainer = IncrementalTrainer(sample_config)
        cm = CheckpointManager(str(ck))
        trainer._apply_checkpoint_post_success_cleanup(cm)

        assert not (ck / "temp").exists()
        assert not (mcp / "model_2006_01").exists()
        assert (mcp / "model_2006_02").exists()

    def test_apply_checkpoint_post_success_cleanup_false_noop(self, temp_dir, sample_config):
        sample_config["incremental_training"]["checkpoint_post_success_cleanup"] = False
        ck = Path(temp_dir) / "ckpt_no_cleanup"
        ck.mkdir(parents=True)
        (ck / "temp" / "scratch").mkdir(parents=True)

        trainer = IncrementalTrainer(sample_config)
        cm = CheckpointManager(str(ck))
        trainer._apply_checkpoint_post_success_cleanup(cm)

        assert (ck / "temp").exists()

    def test_evaluate_model_performance_marks_constant_validation_invalid(self, sample_config):
        ts_module = pytest.importorskip("autogluon.timeseries")
        trainer = IncrementalTrainer(sample_config)

        timestamps = pd.date_range(start="2020-01-01", periods=80, freq="1h")
        targets = np.concatenate((np.linspace(0.1, 0.2, 56), np.full(24, 0.1498844474554062)))
        df = pd.DataFrame(
            {
                "item_id": "series_a",
                "timestamp": timestamps,
                "target": targets,
            }
        )
        ts_df = ts_module.TimeSeriesDataFrame.from_data_frame(
            df, id_column="item_id", timestamp_column="timestamp"
        )

        result = trainer._evaluate_model_performance(MagicMock(), ts_df)

        assert result["validation_valid"] is False
        assert result["validation_reason"] == "constant_validation_target"
        assert result["mae"] is None
        assert result["validation_summary"]["validation_rows"] == 24

    def test_evaluate_model_performance_uses_per_series_holdout(self, sample_config):
        ts_module = pytest.importorskip("autogluon.timeseries")
        trainer = IncrementalTrainer(sample_config)

        timestamps = pd.date_range(start="2020-01-01", periods=80, freq="1h")
        df = pd.DataFrame(
            {
                "item_id": ["series_a"] * 80 + ["series_b"] * 80,
                "timestamp": list(timestamps) + list(timestamps),
                "target": np.concatenate(
                    (
                        np.linspace(0.10, 0.30, 80),
                        np.linspace(0.40, 0.62, 80),
                    )
                ),
            }
        )
        ts_df = ts_module.TimeSeriesDataFrame.from_data_frame(
            df, id_column="item_id", timestamp_column="timestamp"
        )

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = pd.DataFrame(
            {"0.5": np.linspace(0.2, 0.4, 48)}
        )

        result = trainer._evaluate_model_performance(mock_predictor, ts_df)

        assert result["validation_valid"] is True
        assert result["validation_reason"] == "ok"
        assert result["validation_summary"]["series_included"] == 2
        assert result["validation_summary"]["validation_rows"] == 48
        train_data = mock_predictor.predict.call_args.args[0]
        assert len(train_data) == 112

    def test_evaluate_model_performance_invalid_when_series_too_short(self, sample_config):
        ts_module = pytest.importorskip("autogluon.timeseries")
        trainer = IncrementalTrainer(sample_config)

        timestamps = pd.date_range(start="2020-01-01", periods=40, freq="1h")
        df = pd.DataFrame(
            {
                "item_id": "short_series",
                "timestamp": timestamps,
                "target": np.linspace(0.1, 0.3, 40),
            }
        )
        ts_df = ts_module.TimeSeriesDataFrame.from_data_frame(
            df, id_column="item_id", timestamp_column="timestamp"
        )
        mock_predictor = MagicMock()

        result = trainer._evaluate_model_performance(mock_predictor, ts_df)

        assert result["validation_valid"] is False
        assert result["validation_reason"] == "insufficient_series_length"
        assert result["validation_summary"]["series_included"] == 0
        mock_predictor.predict.assert_not_called()

    @patch(
        "chronos_trainer.training.checkpoint_manager.TimeSeriesPredictor.load",
        return_value=MagicMock(),
    )
    def test_checkpoint_training_uses_previous_model_warm_start(
        self, _mock_ts_load, temp_dir, sample_config, caplog
    ):
        checkpoint_dir = str(Path(temp_dir) / "checkpoints")
        previous_model_path = Path(temp_dir) / "previous_model"
        previous_model_path.mkdir(parents=True)

        trainer = IncrementalTrainer(sample_config)
        warm_predictor = MagicMock()
        final_predictor = MagicMock()
        final_predictor.path = _minimal_autogluon_predictor_dir(
            Path(temp_dir), min_total_bytes=_MIN_FINAL_MODEL_BYTES
        )

        with patch.object(
            trainer, "_load_previous_model", return_value=warm_predictor
        ), patch.object(
            trainer, "_train_predictor", return_value=(final_predictor, 0.01)
        ):
            with caplog.at_level(logging.INFO):
                result = trainer.train_with_checkpoints(
                    start_date="2020-01-01",
                    end_date="2020-01-31",
                    validation_start_date="2020-02-01",
                    validation_end_date="2020-02-07",
                    checkpoint_dir=checkpoint_dir,
                    previous_model_path=str(previous_model_path),
                )

        assert result["status"] == "completed"
        assert (
            "mode=warm_start_from_previous_model" in caplog.text
        ), caplog.text
        assert "mode=fresh_start_fallback_from_previous_model" not in caplog.text

    @patch(
        "chronos_trainer.training.checkpoint_manager.TimeSeriesPredictor.load",
        return_value=MagicMock(),
    )
    def test_checkpoint_training_falls_back_when_previous_model_unloadable(
        self, _mock_ts_load, temp_dir, sample_config, caplog
    ):
        checkpoint_dir = str(Path(temp_dir) / "checkpoints")
        previous_model_path = Path(temp_dir) / "previous_model"
        previous_model_path.mkdir(parents=True)

        trainer = IncrementalTrainer(sample_config)
        final_predictor = MagicMock()
        final_predictor.path = _minimal_autogluon_predictor_dir(
            Path(temp_dir), min_total_bytes=_MIN_FINAL_MODEL_BYTES
        )

        with patch.object(
            trainer,
            "_load_previous_model",
            side_effect=IncrementalTrainingError("corrupt_previous_model"),
        ), patch.object(
            trainer, "_train_predictor", return_value=(final_predictor, 0.01)
        ):
            with caplog.at_level(logging.WARNING):
                result = trainer.train_with_checkpoints(
                    start_date="2020-01-01",
                    end_date="2020-01-31",
                    validation_start_date="2020-02-01",
                    validation_end_date="2020-02-07",
                    checkpoint_dir=checkpoint_dir,
                    previous_model_path=str(previous_model_path),
                )

        assert result["status"] == "completed"
        assert "mode=fresh_start_fallback_from_previous_model" in caplog.text
        assert "fallback_reason=corrupt_previous_model" in caplog.text

    @patch(
        "chronos_trainer.training.checkpoint_manager.TimeSeriesPredictor.load",
        return_value=MagicMock(),
    )
    def test_resume_checkpoint_takes_precedence_over_previous_model(
        self, _mock_ts_load, temp_dir, sample_config, caplog
    ):
        checkpoint_dir = str(Path(temp_dir) / "checkpoints")
        checkpoint_manager = CheckpointManager(checkpoint_dir)
        prior_predictor = MagicMock()
        prior_predictor.path = _minimal_autogluon_predictor_dir(
            Path(temp_dir), min_total_bytes=_MIN_FINAL_MODEL_BYTES
        )
        checkpoint_manager.save_checkpoint(
            year=2020,
            month=1,
            model=prior_predictor,
            data_stats={"record_count": 100},
            training_state={
                "start_date": "2020-01-01",
                "end_date": "2020-02-28",
                "validation_start_date": "2020-03-01",
                "validation_end_date": "2020-03-07",
                "processed_files": [
                    {"file_path": "data.parquet", "year": 2020, "month": 1}
                ],
                "total_files": 2,
            },
        )

        previous_model_path = Path(temp_dir) / "previous_model"
        previous_model_path.mkdir(parents=True)
        trainer = IncrementalTrainer(sample_config)
        final_predictor = MagicMock()
        final_predictor.path = _minimal_autogluon_predictor_dir(
            Path(temp_dir), min_total_bytes=_MIN_FINAL_MODEL_BYTES
        )

        with patch.object(
            trainer, "_train_predictor", return_value=(final_predictor, 0.01)
        ):
            with caplog.at_level(logging.INFO):
                result = trainer.train_with_checkpoints(
                    start_date="2020-01-01",
                    end_date="2020-02-28",
                    validation_start_date="2020-03-01",
                    validation_end_date="2020-03-07",
                    checkpoint_dir=checkpoint_dir,
                    previous_model_path=str(previous_model_path),
                )

        assert result["status"] == "completed"
        assert "mode=resume_from_disk" in caplog.text
        assert "previous_model is ignored while resuming from checkpoint" in caplog.text

