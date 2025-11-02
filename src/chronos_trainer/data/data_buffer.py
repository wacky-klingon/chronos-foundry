"""
Data Buffer for efficient data handling during training

Provides buffering capabilities for managing data during incremental training.
"""

from typing import List, Dict
import logging
import pandas as pd


class DataUnavailableError(Exception):
    """Raised when requested data is not available in the buffer"""


class DataBuffer:
    """
    Buffer for managing data during training workflows.

    Provides caching and efficient data access patterns for incremental training.
    """

    def __init__(self, max_size: int = 10):
        """
        Initialize data buffer

        Args:
            max_size: Maximum number of data chunks to keep in buffer
        """
        self.max_size = max_size
        self.buffer: Dict[str, pd.DataFrame] = {}
        self.access_order: List[str] = []
        self.logger = logging.getLogger(__name__)

    def add(self, key: str, data: pd.DataFrame) -> None:
        """
        Add data to buffer with LRU eviction

        Args:
            key: Unique identifier for the data chunk
            data: DataFrame to store
        """
        # Remove if already exists (update access order)
        if key in self.buffer:
            self.access_order.remove(key)

        # Add to buffer
        self.buffer[key] = data
        self.access_order.append(key)

        # Evict oldest if over limit
        while len(self.buffer) > self.max_size:
            oldest_key = self.access_order.pop(0)
            del self.buffer[oldest_key]
            self.logger.debug("Evicted %s from buffer", oldest_key)

    def get(self, key: str) -> pd.DataFrame:
        """
        Get data from buffer

        Args:
            key: Unique identifier for the data chunk

        Returns:
            DataFrame from buffer

        Raises:
            DataUnavailableError: If key is not in buffer
        """
        if key not in self.buffer:
            raise DataUnavailableError(f"Data not available in buffer: {key}")

        # Update access order (move to end)
        self.access_order.remove(key)
        self.access_order.append(key)

        return self.buffer[key]

    def has(self, key: str) -> bool:
        """
        Check if data is in buffer

        Args:
            key: Unique identifier for the data chunk

        Returns:
            True if data is in buffer, False otherwise
        """
        return key in self.buffer

    def clear(self) -> None:
        """Clear all data from buffer"""
        self.buffer.clear()
        self.access_order.clear()
        self.logger.debug("Buffer cleared")

    def size(self) -> int:
        """
        Get current buffer size

        Returns:
            Number of items in buffer
        """
        return len(self.buffer)

    def keys(self) -> List[str]:
        """
        Get all keys in buffer

        Returns:
            List of buffer keys
        """
        return list(self.buffer.keys())

