# Test Coverage TODO for chronos-foundry

This document lists tests that should be added to chronos-foundry's test suite. These tests currently exist in the-heisenberg-engine but are testing chronos-foundry's internal APIs.

## Status
- ✅ Current: 24 tests total (12 in `test_basic.py` + 12 new tests)
- ✅ Completed: Phase 1 tests (checkpoint_manager, resumable_loader)
- ⏭️ To Add: Phase 2 tests (training_progress, remaining_files) - Medium priority
- ⏭️ To Add: Phase 3 tests (checkpoint_cleanup, data_stats) - Low priority
- 🎯 Target: Full coverage of public APIs

---

## 1. CheckpointManager Tests

**Module:** `chronos_trainer.training.checkpoint_manager`

### test_checkpoint_save_and_load
**Priority:** HIGH
**Source:** `the-heisenberg-engine/tests/test_resumable_training.py:92`

**What it tests:**
- Saving checkpoints with TimeSeriesPredictor object
- Loading checkpoints and restoring model state
- Checkpoint data structure (year, month, model_path, data_stats, training_state)

**Key Requirements:**
- Must use real or properly mocked `TimeSeriesPredictor` object
- Test checkpoint file creation in `checkpoints_dir`
- Test model saving in `model_checkpoints_dir`
- Verify checkpoint JSON structure

**Implementation Notes:**
```python
def test_checkpoint_save_and_load(checkpoint_manager):
    # Create a real TimeSeriesPredictor or MagicMock with .path and .save()
    from unittest.mock import MagicMock
    mock_model = MagicMock(spec=['path', 'save', 'load'])
    mock_model.path = None

    data_stats = {"record_count": 100, "columns": ["ds", "target"]}
    training_state = {"start_date": "2020-01-01", "processed_files": [...]}

    success = checkpoint_manager.save_checkpoint(
        year=2020, month=1,
        model=mock_model,
        data_stats=data_stats,
        training_state=training_state
    )

    assert success
    # Test loading...
```

---

### test_get_last_checkpoint
**Priority:** HIGH
**Source:** `the-heisenberg-engine/tests/test_resumable_training.py:132`

**What it tests:**
- Getting the most recent checkpoint
- Returns None when no checkpoints exist
- Returns correct checkpoint data when checkpoints exist
- Checkpoint sorting by timestamp

**Key Scenarios:**
1. No checkpoints → returns None
2. Single checkpoint → returns that checkpoint
3. Multiple checkpoints → returns most recent

---

### test_training_progress
**Priority:** MEDIUM
**Source:** `the-heisenberg-engine/tests/test_resumable_training.py:155`

**What it tests:**
- `get_training_progress()` method
- Progress status states: "not_started", "in_progress", "completed"
- Tracking last processed file
- Progress percentage calculation

**Expected Output:**
```python
{
    "status": "in_progress",
    "last_processed": {"year": 2020, "month": 1},
    "total_checkpoints": 5,
    "progress_pct": 20.0
}
```

---

### test_checkpoint_cleanup
**Priority:** LOW
**Source:** `the-heisenberg-engine/tests/test_resumable_training.py:305`

**What it tests:**
- Old checkpoint deletion
- Keeping only N most recent checkpoints (e.g., max_versions=5)
- Cleanup of both checkpoint JSON and model files

---

## 2. ResumableDataLoader Tests

**Module:** `chronos_trainer.data.resumable_loader`

### test_resumable_loader_remaining_files
**Priority:** MEDIUM
**Source:** `the-heisenberg-engine/tests/test_resumable_training.py:192`

**What it tests:**
- `get_remaining_files()` method
- Filter files based on checkpoint state
- Integration with CheckpointManager to determine which files are already processed

**Scenario:**
```python
# Total: 12 monthly files (2020-01 to 2020-12)
# Processed: 3 files (2020-01, 2020-02, 2020-03)
# Remaining: 9 files (2020-04 to 2020-12)
```

---

### test_resumable_loader_data_stats
**Priority:** LOW
**Source:** `the-heisenberg-engine/tests/test_resumable_training.py:234`

**What it tests:**
- Data statistics generation
- Expected stats format from chronos-foundry

**Current Format Issue:**
The test expects:
```python
{
    "record_count": 100,
    "has_target": True,  # ← This field doesn't exist in chronos-foundry
    "columns": [...],
    "start_time": "2020-01-01",
    "end_time": "2020-01-31"
}
```

**Action Required:**
Either update test to match chronos-foundry's actual stats format, or add `has_target` field to ResumableDataLoader.

---

## 3. IncrementalTrainer Tests

**Module:** `chronos_trainer.training.incremental_trainer`

### test_integration_resumable_training
**Priority:** HIGH
**Source:** `the-heisenberg-engine/tests/test_resumable_training.py:251`

**What it tests:**
- Full end-to-end incremental training workflow
- Multiple file processing
- Checkpoint creation after each file
- Model versioning
- Quality gates

**Blocked By:**
- Missing `model_name` in test config
- Need to create proper config with all required fields:
  - `model_name`, `context_length`, `prediction_length`
  - `learning_rate`, `batch_size`, `max_epochs`

**Implementation Notes:**
```python
config = {
    "model_name": "amazon/chronos-bolt-tiny",
    "context_length": 96,
    "prediction_length": 24,
    "learning_rate": 0.001,
    "batch_size": 32,
    "max_epochs": 1,  # Fast for testing
    "training_preset": "medium_quality",  # Fast for testing
    "parquet_loader": {...},
    "incremental": {
        "versioning_enabled": True,
        "quality_gate_enabled": True,
        "performance_threshold": 0.05
    }
}
```

---

### test_resume_training
**Priority:** HIGH
**Source:** `the-heisenberg-engine/tests/test_resumable_training.py:279`

**What it tests:**
- Training interruption and resume
- Loading from checkpoint
- Continuing from last processed file
- Not re-processing already completed files

**Scenario:**
1. Start training with 12 files
2. Process 5 files successfully (checkpoints saved)
3. Simulate interruption/crash
4. Resume training → should start from file 6

---

## 4. Current chronos-foundry Test Coverage

**File:** `tests/test_basic.py`

**Existing Tests (12 total):**
- ✅ test_imports (1)
- ✅ test_config_provider (1)
- ✅ test_checkpoint_manager_initialization (1)
- ✅ test_data_generation (1)
- ✅ test_custom_trainer_instantiation (1)
- ✅ test_minimal_training_workflow (1)
- ✅ test_data_validation (1)
- ✅ test_checkpoint_save_load (1)
- ✅ test_incremental_checkpointing (1)
- ✅ test_model_versioning_save_load (1)
- ✅ test_model_versioning_quality_gate (1)
- ✅ test_model_versioning_rollback (1)

**Note:** Some tests above may overlap with what needs to be added. Review and consolidate.

---

## 5. Implementation Priority

### Phase 1: Critical (Add First) ✅ COMPLETED
1. ✅ `test_checkpoint_save_and_load` - Core functionality (IMPLEMENTED in test_checkpoint_manager.py)
2. ✅ `test_get_last_checkpoint` - Core functionality (IMPLEMENTED - includes none/single/multiple scenarios)
3. ✅ `test_integration_resumable_training` - End-to-end validation (IMPLEMENTED in test_incremental_trainer.py)
4. ✅ `test_resume_training` - Core use case (IMPLEMENTED in test_incremental_trainer.py)

### Phase 2: Important
5. ✅ `test_training_progress` - User-facing feature (IMPLEMENTED in test_checkpoint_manager.py)
6. ✅ `test_resumable_loader_remaining_files` - Efficiency feature (IMPLEMENTED in test_resumable_loader.py)

### Phase 3: Nice to Have
7. ⏭️ `test_checkpoint_cleanup` - Maintenance feature
8. ⏭️ `test_resumable_loader_data_stats` - Utility feature

---

## 6. Test Strategy

### For CheckpointManager:
```python
# Use MagicMock for TimeSeriesPredictor
from unittest.mock import MagicMock

mock_predictor = MagicMock()
mock_predictor.path = None
mock_predictor.save = MagicMock()
mock_predictor.load = MagicMock(return_value=mock_predictor)
```

### For IncrementalTrainer:
```python
# Use tiny model and small dataset for fast testing
config = {
    "model_name": "amazon/chronos-bolt-tiny",
    "training_preset": "medium_quality",
    "max_epochs": 1,
    "time_limit": 60,
    # ... other required fields
}
```

### For ResumableDataLoader:
```python
# Create temporary parquet files with synthetic data
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as temp_dir:
    # Create test data structure
    base_path = Path(temp_dir) / "data"
    base_path.mkdir()
    # ... create YYYY/MM structure with parquet files
```

---

## 7. Action Items

- [x] Review existing `test_basic.py` for overlaps
- [x] Create `tests/test_checkpoint_manager.py` (6 tests - COMPLETED)
- [x] Create `tests/test_resumable_loader.py` (6 tests - COMPLETED)
- [x] Create `tests/test_incremental_trainer.py` (3 tests - COMPLETED)
- [x] Move relevant tests from heisenberg-engine (with proper mocking)
- [ ] Update test documentation in [testing guide](../user-guides/testing.md) - IN PROGRESS
- [x] Ensure all tests use synthetic data (no external dependencies)
- [ ] Add CI/CD integration (GitHub Actions) - FUTURE

---

## 8. Notes

**Why These Tests Matter:**
- chronos-foundry is a public library - needs comprehensive test coverage
- Users depend on CheckpointManager and IncrementalTrainer APIs
- Tests document expected behavior
- Prevent regressions when updating code

**Test Data Strategy:**
- Use synthetic data only (no external datasets required)
- Keep test execution fast (<5 seconds per test)
- Use small models (`chronos-bolt-tiny`)
- Mock heavy operations where appropriate

**Documentation:**
Each test should include:
- Clear docstring explaining what's being tested
- Comments for non-obvious assertions
- Example usage patterns for library users

