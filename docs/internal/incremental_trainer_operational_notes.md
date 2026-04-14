# Incremental Trainer: Operational Notes

Design notes for `IncrementalTrainer` behavior: rolling history window, model exclusions, and data loading fixes. Implementation lives under `src/chronos_trainer/`.

## Root cause: O(N^2) training growth

Each month, `_train_predictor` in `training/incremental_trainer.py` calls `_load_previous_training_data(processed_files)`, which re-reads all prior monthly parquets, concatenates them, and fits a new `TimeSeriesPredictor`. As `processed_files` grows, work scales as O(N^2).

Predictor logs showed TiDE dominated runtime (majority of wall time per month).

## Configuration levers (Heisenberg app config)

Keys belong in the application `incremental_training_config.yaml` (see the-heisenberg-engine), for example:

- `lookback_days`: cap how much prior history is reload each iteration (required for bounded work).
- `excluded_model_types`: e.g. skip `TemporalFusionTransformer` and `TiDE` when they dominate runtime or fail repeatedly.
- `known_covariates`: macro columns available at forecast time for covariate-aware models.

## Code integration

In `_train_predictor`, use config-driven `excluded_model_types`, `known_covariates_names`, and optional `lookback_days` to slice `processed_files` before `_load_previous_training_data`.

## Data loader: covariate columns dropped

In `data/resumable_loader.py`, `load_parquet_file` should avoid AutoGluon dropping columns as non-numeric: drop unused date-string helper columns if needed, and coerce feature columns to numeric so covariates are retained.

## Trade-offs

- Excluding heavy models changes ensemble composition; validate metrics after changes.
- A rolling window discards very old data; tune `lookback_days` if quality regresses.

## Related

- End-to-end incremental training and AWS orchestration are documented in the consuming application (for example the-heisenberg-engine `docs/architecture/INCREMENTAL-TRAINING.md` and `docs/aws/`).
