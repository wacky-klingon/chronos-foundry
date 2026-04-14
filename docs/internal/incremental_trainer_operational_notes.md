# Incremental Trainer: Operational Notes

Design notes for `IncrementalTrainer` behavior: rolling history window, Chronos-only incremental policy, artifact integrity validation, and data loading fixes. Implementation lives under `src/chronos_trainer/`.

## Root cause: O(N^2) training growth

Each month, `_train_predictor` in `training/incremental_trainer.py` calls `_load_previous_training_data(processed_files)`, which re-reads all prior monthly parquets, concatenates them, and fits a new `TimeSeriesPredictor`. As `processed_files` grows, work scales as O(N^2).

Predictor logs showed TiDE dominated runtime (majority of wall time per month).

## Configuration levers (Heisenberg app config)

Keys belong in the application `incremental_training_config.yaml` (see the-heisenberg-engine), for example:

- `lookback_days`: cap how much prior history is reload each iteration (required for bounded work).
- `chronos_only`: must be `true` for incremental training.
- `chronos_model_variant`: one of `bolt_tiny`, `bolt_mini`, `bolt_small`, `bolt_base`.
- `known_covariates`: macro columns available at forecast time for covariate-aware models.

## Code integration

In `_train_predictor`, use Chronos-only hyperparameters and `known_covariates_names`. For bounded history, use `lookback_days` to slice `processed_files` before `_load_previous_training_data`.

## Checkpoint and final artifact contract

The incremental pipeline now enforces explicit artifact contracts to prevent publishing pointers to non-loadable models.

- Canonical checkpoint path: `model_checkpoints/model_YYYY_MM/`
- Backward compatibility: load path resolver accepts legacy `model_YYYY_MM.pkl` directories
- Required model files for checkpoint/final validation:
  - `predictor.pkl`
  - `learner.pkl`
  - `models/trainer.pkl`
- Final model directory is expected to be larger than 1 MB

If required artifacts are missing, saving/loading fails fast and the caller surfaces a run failure.

## Structured observability events

Artifact lifecycle and pointer gates are traced with structured logs:

- `checkpoint_event` (checkpoint manager)
- `artifact_event` (incremental trainer)
- `WRAPPER_EVENT` (AWS training wrapper in consuming app)

These events should be used as the first-line signal during AWS run triage.

## Data loader: covariate columns dropped

In `data/resumable_loader.py`, `load_parquet_file` should avoid AutoGluon dropping columns as non-numeric: drop unused date-string helper columns if needed, and coerce feature columns to numeric so covariates are retained.

## Trade-offs

- Chronos-only mode removes mixed-model search and ensemble candidates; validate forecast quality when switching variants.
- A rolling window discards very old data; tune `lookback_days` if quality regresses.

## Related

- End-to-end incremental training and AWS orchestration are documented in the consuming application (for example the-heisenberg-engine `docs/architecture/INCREMENTAL-TRAINING.md` and `docs/aws/`).
