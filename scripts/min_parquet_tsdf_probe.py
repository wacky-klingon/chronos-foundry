#!/usr/bin/env python3
"""Minimal load + TimeSeriesDataFrame conversion probe for a single cached parquet file."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC = SCRIPT_DIR.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402


def load_resumable_loader_module():
    path = SRC / "chronos_trainer" / "data" / "resumable_loader.py"
    name = "chronos_trainer_data_resumable_loader_standalone"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_parquet_path() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser().resolve()
    for p in (
        Path(r"C:\dev\projects\cached_datasets\2024\02\USDCAD_1min_h15_2024_02_bef847556b9fa046.parquet"),
        Path("/mnt/c/dev/projects/cached_datasets/2024/02/USDCAD_1min_h15_2024_02_bef847556b9fa046.parquet"),
    ):
        if p.exists():
            return p.resolve()
    raise SystemExit(
        "No parquet path given and no default file found. Pass path as argv[1]. "
        r"Defaults: C:\dev\projects\cached_datasets\... and /mnt/c/dev/projects/cached_datasets\..."
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("min_parquet_tsdf_probe")

    parquet_path = resolve_parquet_path()
    data_root = parquet_path.parent.parent.parent
    if not data_root.is_dir():
        raise SystemExit(f"Expected cached_datasets root missing: {data_root}")

    log.info("parquet=%s", parquet_path)
    log.info("data_root=%s", data_root)

    raw = pd.read_parquet(parquet_path)
    cols = list(raw.columns)
    log.info("parquet_only rows=%s n_cols=%s", len(raw), len(cols))
    log.info("parquet_only columns=%s", cols)
    log.info(
        "parquet_only target_close=%s ds=%s item_id=%s target_name_count=%s dup_labels=%s",
        "target_close" in raw.columns,
        "ds" in raw.columns,
        "item_id" in raw.columns,
        cols.count("target"),
        bool(raw.columns.duplicated().any()),
    )
    if "target" in raw.columns and "target_close" in raw.columns:
        log.warning(
            "Schema risk: both literal columns 'target' and 'target_close' exist. "
            "With target_col=target_close the loader renames target_close -> 'target', "
            "which typically yields duplicate column label 'target' (odd layout for AutoGluon)."
        )

    rl = load_resumable_loader_module()
    ResumableDataLoader = rl.ResumableDataLoader
    log_autogluon_timeseries_dataframe_probe = rl.log_autogluon_timeseries_dataframe_probe
    if rl.TimeSeriesDataFrame is None:
        log.warning(
            "autogluon not installed: skipping ResumableDataLoader.convert and "
            "TimeSeriesPredictor.fit. Install autogluon for full probe."
        )
        return 0

    loader = ResumableDataLoader(str(data_root), checkpoint_manager=None)
    year, month = int(parquet_path.parent.parent.name), int(parquet_path.parent.name)
    df = loader.load_parquet_file(str(parquet_path), year, month)
    if df is None:
        log.error("load_parquet_file returned None")
        return 1

    log.info("loader rows=%s cols=%s", len(df), list(df.columns))

    config = {
        "timestamp_col": "ds",
        "target_col": "target_close",
        "item_id_col": "item_id",
    }
    ts_df = loader.convert_to_timeseries_dataframe(df, config)
    if ts_df is None:
        log.error("convert_to_timeseries_dataframe returned None")
        return 1

    log.info("converted len(ts_df)=%s", len(ts_df))
    log_autogluon_timeseries_dataframe_probe(ts_df, log, phase="min_script_post_convert")

    try:
        from autogluon.timeseries import TimeSeriesPredictor

        pred = TimeSeriesPredictor(
            target="target",
            prediction_length=64,
            path=str(SCRIPT_DIR / "_min_probe_autogluon_models"),
            verbosity=0,
        )
        pred.fit(ts_df, hyperparameters={"Naive": {}})
        log.info("TimeSeriesPredictor.fit completed (Naive smoke).")
    except Exception as exc:
        log.exception("TimeSeriesPredictor.fit failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
