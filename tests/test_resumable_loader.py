"""
Tests for ResumableDataLoader functionality

Tests file discovery, loading, and checkpoint integration.
"""

import pytest
import tempfile
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
import shutil

from chronos_trainer.data import ResumableDataLoader
from chronos_trainer.training.checkpoint_manager import CheckpointManager
from unittest.mock import MagicMock


class TestResumableDataLoader:
    """Test ResumableDataLoader core functionality"""

    @pytest.fixture
    def sample_data_dir(self):
        """Create temporary directory with YYYY/MM/ structure and parquet files"""
        temp = tempfile.mkdtemp()
        base_path = Path(temp)

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

        yield str(base_path)

        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def checkpoint_manager(self):
        """Create CheckpointManager for testing"""
        temp = tempfile.mkdtemp()
        manager = CheckpointManager(temp)
        yield manager
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def resumable_loader(self, sample_data_dir, checkpoint_manager):
        """Create ResumableDataLoader instance"""
        return ResumableDataLoader(sample_data_dir, checkpoint_manager)

    def test_get_parquet_files(self, resumable_loader):
        """Test getting parquet files in date range"""
        files = resumable_loader.get_parquet_files("2020-01-01", "2020-02-28")

        assert len(files) == 2, "Should find 2 files (Jan and Feb)"
        assert all(len(f) == 3 for f in files), "Each file should be (path, year, month) tuple"

        # Check chronological order
        assert files[0][1] == 2020 and files[0][2] == 1, "First file should be January"
        assert files[1][1] == 2020 and files[1][2] == 2, "Second file should be February"

    def test_load_parquet_file(self, resumable_loader):
        """Test loading a single parquet file"""
        files = resumable_loader.get_parquet_files("2020-01-01", "2020-01-31")

        assert len(files) > 0, "Should find at least one file"

        file_path, year, month = files[0]
        df = resumable_loader.load_parquet_file(file_path, year, month)

        assert df is not None, "Should load parquet file"
        assert len(df) == 100, "Should have 100 records"
        assert "_year" in df.columns, "Should add _year metadata column"
        assert "_month" in df.columns, "Should add _month metadata column"
        assert df["_year"].iloc[0] == 2020, "Year metadata should match"
        assert df["_month"].iloc[0] == 1, "Month metadata should match"

    def test_convert_to_timeseries_dataframe(self, resumable_loader):
        """Test converting DataFrame to TimeSeriesDataFrame"""
        files = resumable_loader.get_parquet_files("2020-01-01", "2020-01-31")
        file_path, year, month = files[0]

        df = resumable_loader.load_parquet_file(file_path, year, month)

        config = {
            "timestamp_col": "timestamp",
            "target_col": "target",
            "item_id_col": "item_id",
        }

        ts_df = resumable_loader.convert_to_timeseries_dataframe(df, config)

        assert ts_df is not None, "Should convert to TimeSeriesDataFrame"
        # TimeSeriesDataFrame has different structure - just verify it's not None

    def test_resumable_loader_remaining_files(self, sample_data_dir, checkpoint_manager):
        """Test getting remaining files based on checkpoint state"""
        loader = ResumableDataLoader(sample_data_dir, checkpoint_manager)

        # Initially all files should be remaining
        remaining = loader.get_remaining_files("2020-01-01", "2020-02-28")
        assert len(remaining) == 2, "All files should be remaining initially"

        # Save a checkpoint for 2020-01
        mock_predictor = MagicMock()
        mock_predictor.path = None
        mock_predictor.save = MagicMock()

        checkpoint_manager.save_checkpoint(
            year=2020,
            month=1,
            model=mock_predictor,
            data_stats={"record_count": 100},
            training_state={
                "start_date": "2020-01-01",
                "end_date": "2020-02-28",
                "processed_files": [
                    {"file_path": "data.parquet", "year": 2020, "month": 1}
                ],
            },
        )

        # Now only 2020-02 should be remaining
        remaining = loader.get_remaining_files("2020-01-01", "2020-02-28")
        assert len(remaining) == 1, "Only unprocessed file should remain"
        assert remaining[0][2] == 2, "Remaining file should be February"

    def test_get_data_stats(self, resumable_loader):
        """Test data statistics generation"""
        files = resumable_loader.get_parquet_files("2020-01-01", "2020-01-31")
        file_path, year, month = files[0]

        df = resumable_loader.load_parquet_file(file_path, year, month)
        stats = resumable_loader.get_data_stats(df)

        assert "record_count" in stats
        assert "columns" in stats
        assert "memory_usage_mb" in stats
        assert stats["record_count"] == 100
        assert "timestamp" in stats["columns"]
        assert "target" in stats["columns"]

    def test_get_remaining_files_no_checkpoint_manager(self, sample_data_dir):
        """Test get_remaining_files when no checkpoint manager provided"""
        loader = ResumableDataLoader(sample_data_dir, checkpoint_manager=None)

        remaining = loader.get_remaining_files("2020-01-01", "2020-02-28")
        # Without checkpoint manager, should return all files
        assert len(remaining) == 2

