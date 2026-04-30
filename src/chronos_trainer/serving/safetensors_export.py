# RM - future-fixme: This module is a one-time copy from the donor file
# chronos-finetuning/src/tachyon_model_downloader/fine_tune_and_export.py @ git ac19b47.
# Consolidation target: tachyon-core/.tbd/post_offline_work.md section F4.
# Any changes to checkpoint discovery or copy logic should be reviewed against
# the donor file for drift. Do not add a runtime dependency on chronos-finetuning.
"""Deterministic safetensors checkpoint discovery, copy, and artifact purge.

Lifted from chronos-finetuning fine_tune_and_export.py (donor commit ac19b47).
Changes from donor:
  - sleep(0.5) removed (WSL cross-mount magic constant, not needed here).
  - AG-internal reflection helpers dropped; deterministic checkpoint-dir copy is
    the only code path that shipped artifacts in chronos-finetuning's export manifest.
  - Private names promoted to public API.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .model_validation import FORBIDDEN_ARTIFACT_SUFFIXES


def find_finetuned_checkpoint_dir(predictor_dir: Path) -> Path | None:
    """Return the newest ``fine-tuned-ckpt`` directory containing both
    ``config.json`` and ``model.safetensors``, or ``None`` if not found.
    """
    models_dir = predictor_dir / "models"
    if not models_dir.exists() or not models_dir.is_dir():
        return None
    candidates: list[Path] = []
    for candidate in models_dir.rglob("fine-tuned-ckpt"):
        if not candidate.is_dir():
            continue
        if (candidate / "config.json").exists() and (candidate / "model.safetensors").exists():
            candidates.append(candidate)
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def copy_checkpoint_to_export(
    predictor_dir: Path,
    export_dir: Path,
) -> tuple[bool, str, Path | None]:
    """Copy ``config.json`` and ``model.safetensors`` from the newest fine-tuned
    checkpoint directory inside ``predictor_dir`` into ``export_dir``.

    Returns:
        (success, message, checkpoint_dir) where ``checkpoint_dir`` is the source
        path used, or ``None`` on failure.
    """
    checkpoint_dir = find_finetuned_checkpoint_dir(predictor_dir=predictor_dir)
    if checkpoint_dir is None:
        return (
            False,
            "No fine-tuned checkpoint directory with config.json + model.safetensors "
            "was found under predictor models/.",
            None,
        )

    shutil.copy2(checkpoint_dir / "config.json", export_dir / "config.json")
    shutil.copy2(checkpoint_dir / "model.safetensors", export_dir / "model.safetensors")
    return (
        True,
        f"Copied fine-tuned checkpoint artifacts from {checkpoint_dir}.",
        checkpoint_dir,
    )


def purge_forbidden_artifacts(export_dir: Path) -> list[Path]:
    """Remove any pickle-based artifacts from ``export_dir``.

    Returns the list of paths that were deleted.
    """
    purged: list[Path] = []
    for path in export_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in FORBIDDEN_ARTIFACT_SUFFIXES:
            purged.append(path)
            path.unlink()
    return purged
