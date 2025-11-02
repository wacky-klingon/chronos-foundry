"""
Base Data Converter for transforming data formats

Provides base class for data conversion operations.
"""

from typing import Dict, Any, Optional
import logging


class BaseDataConverter:
    """
    Base class for data conversion operations.

    Provides common functionality for converting data between different formats
    (pandas DataFrame, TimeSeriesDataFrame, etc.)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data converter

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def convert(self, data: Any, **kwargs) -> Any:
        """
        Convert data to target format

        Args:
            data: Input data
            **kwargs: Additional conversion parameters

        Returns:
            Converted data

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement convert() method")

    def validate(self, data: Any) -> bool:
        """
        Validate data format

        Args:
            data: Data to validate

        Returns:
            True if data is valid, False otherwise
        """
        raise NotImplementedError("Subclasses must implement validate() method")

    def get_schema(self) -> Dict[str, Any]:
        """
        Get expected data schema

        Returns:
            Dictionary describing expected schema
        """
        return {
            "timestamp_column": self.config.get("timestamp_col", "timestamp"),
            "target_column": self.config.get("target_col", "target"),
            "item_id_column": self.config.get("item_id_col", "item_id"),
        }

