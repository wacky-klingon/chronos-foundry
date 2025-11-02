"""
Chronos Trainer - Production time series forecasting framework

A production-grade framework for Chronos models with:
- Covariate integration
- Incremental training with versioning
- Model lifecycle management
- State management and checkpointing
"""

__version__ = "0.1.0"

# Re-export main classes for easy importing
from chronos_trainer.training import (
    ChronosTrainer,
    CovariateTrainer,
    IncrementalTrainer,
    CheckpointManager,
)
from chronos_trainer.data import (
    ResumableDataLoader,
    DataBuffer,
)
from chronos_trainer.core import (
    ConfigProvider,
)

__all__ = [
    "__version__",
    "ChronosTrainer",
    "CovariateTrainer",
    "IncrementalTrainer",
    "CheckpointManager",
    "ResumableDataLoader",
    "DataBuffer",
    "ConfigProvider",
]

