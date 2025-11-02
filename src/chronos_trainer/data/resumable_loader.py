"""
Resumable Data Loader for checkpoint-based incremental training

This module provides functionality to load parquet files organized in YYYY/MM/ structure
with integration to CheckpointManager for resumable training workflows.
"""

import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import glob

try:
    from autogluon.timeseries import TimeSeriesDataFrame
except ImportError:
    TimeSeriesDataFrame = None
    logging.warning("AutoGluon not available - TimeSeriesDataFrame conversion will fail")


class ResumableDataLoader:
    """
    Loads parquet files organized in YYYY/MM/ directory structure with checkpoint support.

    Integrates with CheckpointManager to track which files have been processed and
    supports resumable training workflows.
    """

    def __init__(self, base_data_path: str, checkpoint_manager=None):
        """
        Initialize resumable data loader

        Args:
            base_data_path: Root directory containing YYYY/MM/ subdirectories with parquet files
            checkpoint_manager: Optional CheckpointManager instance for tracking processed files
        """
        self.base_path = Path(base_data_path)
        if not self.base_path.exists():
            raise ValueError(f"Data directory does not exist: {base_data_path}")

        self.checkpoint_manager = checkpoint_manager
        self.logger = logging.getLogger(__name__)

    def get_parquet_files(
        self, start_date: str, end_date: str
    ) -> List[Tuple[str, int, int]]:
        """
        Get all parquet files in the specified date range

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            List of (file_path, year, month) tuples, sorted chronologically
        """
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            files = []

            # Iterate through all months in range
            current = start_dt.replace(day=1)  # Start of month
            while current <= end_dt:
                year = current.year
                month = current.month

                # Look for parquet files in YYYY/MM/ directory
                month_dir = self.base_path / f"{year:04d}" / f"{month:02d}"
                if month_dir.exists():
                    # Find all parquet files in this month directory
                    parquet_files = list(month_dir.glob("*.parquet"))

                    # If multiple files, sort by name
                    parquet_files.sort()

                    # Add all files for this month
                    for parquet_file in parquet_files:
                        files.append((str(parquet_file), year, month))

                # Move to next month
                if month == 12:
                    current = current.replace(year=year + 1, month=1)
                else:
                    current = current.replace(month=month + 1)

            self.logger.info(f"Found {len(files)} parquet files in range {start_date} to {end_date}")
            return files

        except Exception as e:
            self.logger.error(f"Failed to get parquet files: {e}")
            return []

    def get_remaining_files(
        self, start_date: str, end_date: str
    ) -> List[Tuple[str, int, int]]:
        """
        Get files that haven't been processed yet (based on checkpoint state)

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            List of (file_path, year, month) tuples for unprocessed files
        """
        # Get all files in range
        all_files = self.get_parquet_files(start_date, end_date)

        # If no checkpoint manager, return all files
        if self.checkpoint_manager is None:
            self.logger.warning("No checkpoint manager provided - returning all files")
            return all_files

        # Get training state to see which files are processed
        training_state = self.checkpoint_manager.load_training_state()
        processed_files = set()

        if training_state and "processed_files" in training_state:
            # Create set of (year, month) pairs for quick lookup
            for file_info in training_state["processed_files"]:
                processed_files.add((file_info.get("year"), file_info.get("month")))

        # Filter to only unprocessed files
        remaining = [
            (file_path, year, month)
            for file_path, year, month in all_files
            if (year, month) not in processed_files
        ]

        self.logger.info(
            f"Found {len(remaining)} remaining files out of {len(all_files)} total"
        )
        return remaining

    def load_parquet_file(
        self, file_path: str, year: int, month: int
    ) -> Optional[pd.DataFrame]:
        """
        Load a single parquet file and add metadata columns

        Args:
            file_path: Path to parquet file
            year: Year of the data (for metadata)
            month: Month of the data (for metadata)

        Returns:
            DataFrame with loaded data plus _year and _month columns, or None if failed
        """
        try:
            parquet_path = Path(file_path)
            if not parquet_path.exists():
                self.logger.error(f"Parquet file does not exist: {file_path}")
                return None

            # Load parquet file
            df = pd.read_parquet(parquet_path)

            if df.empty:
                self.logger.warning(f"Parquet file is empty: {file_path}")
                return None

            # Add metadata columns
            df["_year"] = year
            df["_month"] = month

            self.logger.debug(f"Loaded {len(df)} records from {file_path}")
            return df

        except Exception as e:
            self.logger.error(f"Failed to load parquet file {file_path}: {e}")
            return None

    def convert_to_timeseries_dataframe(
        self, df: pd.DataFrame, config: Dict[str, Any]
    ) -> Optional[TimeSeriesDataFrame]:
        """
        Convert pandas DataFrame to AutoGluon TimeSeriesDataFrame

        Args:
            df: Input DataFrame with time series data
            config: Configuration dictionary with column mappings

        Returns:
            TimeSeriesDataFrame ready for training, or None if conversion fails
        """
        if TimeSeriesDataFrame is None:
            self.logger.error("AutoGluon not available - cannot create TimeSeriesDataFrame")
            return None

        try:
            # Get column mappings from config
            timestamp_col = config.get("timestamp_col", "timestamp")
            target_col = config.get("target_col", "target")
            item_id_col = config.get("item_id_col", "item_id")

            # Check if columns exist, if not try common aliases
            column_mapping = {}

            # Map timestamp column
            if timestamp_col not in df.columns:
                # Try common aliases
                for alias in ["ds", "date", "datetime", "timestamp"]:
                    if alias in df.columns:
                        column_mapping[alias] = "timestamp"
                        timestamp_col = "timestamp"
                        break
                else:
                    self.logger.error(f"Timestamp column not found. Expected: {timestamp_col}")
                    return None
            elif timestamp_col != "timestamp":
                column_mapping[timestamp_col] = "timestamp"
                timestamp_col = "timestamp"

            # Map target column
            if target_col not in df.columns:
                # Try common aliases
                for alias in ["value", "target", "y"]:
                    if alias in df.columns:
                        column_mapping[alias] = "target"
                        target_col = "target"
                        break
                else:
                    self.logger.error(f"Target column not found. Expected: {target_col}")
                    return None
            elif target_col != "target":
                column_mapping[target_col] = "target"
                target_col = "target"

            # Map item_id column
            if item_id_col not in df.columns:
                # Try to create item_id from other columns or use default
                if "item_id" not in df.columns:
                    # Create default item_id if none exists
                    df = df.copy()
                    df["item_id"] = "default_item"
                    item_id_col = "item_id"
                else:
                    item_id_col = "item_id"
            elif item_id_col != "item_id":
                column_mapping[item_id_col] = "item_id"
                item_id_col = "item_id"

            # Apply column mappings
            if column_mapping:
                df = df.rename(columns=column_mapping)

            # Ensure required columns exist
            required_cols = ["item_id", "timestamp", "target"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                return None

            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Create TimeSeriesDataFrame
            ts_df = TimeSeriesDataFrame.from_data_frame(
                df, id_column="item_id", timestamp_column="timestamp"
            )

            self.logger.info(
                f"Converted DataFrame to TimeSeriesDataFrame: {len(ts_df)} records"
            )
            return ts_df

        except Exception as e:
            self.logger.error(f"Failed to convert to TimeSeriesDataFrame: {e}")
            return None

    def get_data_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about a DataFrame

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with data statistics
        """
        try:
            stats = {
                "record_count": len(df),
                "columns": list(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            }

            # Add time range if timestamp column exists
            for col in ["timestamp", "ds", "date", "datetime"]:
                if col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        stats["start_time"] = str(df[col].min())
                        stats["end_time"] = str(df[col].max())
                        break

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get data stats: {e}")
            return {"record_count": 0, "columns": [], "memory_usage_mb": 0.0}

