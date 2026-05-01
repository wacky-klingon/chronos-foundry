"""Facade for extracting safetensors artifacts from an AutoGluon predictor directory."""

from __future__ import annotations

import json
import shutil
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autogluon.timeseries import TimeSeriesPredictor
from .model_validation import validate_safetensors_only_model_dir
from .safetensors_export import copy_checkpoint_to_export, purge_forbidden_artifacts
from .schemas import ExportManifest, ExportResult, SafetensorsExtractionError


def _reset_export_dir(export_dir: Path) -> None:
    """Ensure export_dir starts empty to avoid stale artifact mixing."""
    if export_dir.exists():
        for path in export_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    else:
        export_dir.mkdir(parents=True, exist_ok=True)


def _extract_by_path(root: Any, attribute_path: tuple[str, ...]) -> Any | None:
    current = root
    for part in attribute_path:
        if current is None or not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _iter_objects_for_export(root: Any) -> list[Any]:
    queue: deque[Any] = deque([root])
    visited: set[int] = set()
    discovered: list[Any] = []

    while queue:
        current = queue.popleft()
        current_id = id(current)
        if current_id in visited:
            continue
        visited.add(current_id)
        discovered.append(current)

        if isinstance(current, dict):
            queue.extend(current.values())
            continue
        if isinstance(current, (list, tuple, set)):
            queue.extend(current)
            continue

        for attr_name in (
            "model",
            "_model",
            "backbone",
            "network",
            "module",
            "pipeline",
            "tokenizer",
            "_tokenizer",
            "processor",
            "_processor",
            "trainer",
            "_trainer",
            "learner",
            "_learner",
            "models",
            "_models",
            "model_best",
            "_model_best",
        ):
            if hasattr(current, attr_name):
                queue.append(getattr(current, attr_name))

    return discovered


def _is_save_pretrained_object(candidate: Any) -> bool:
    return hasattr(candidate, "save_pretrained") and callable(candidate.save_pretrained)


def _find_exportable_model(root: Any) -> Any | None:
    search_roots: list[Any] = [root]
    for path in (
        ("_learner",),
        ("_learner", "trainer"),
        ("_learner", "trainer", "model"),
        ("_learner", "trainer", "models"),
        ("_learner", "model"),
        ("_learner", "model", "model"),
        ("_learner", "model", "backbone"),
    ):
        extracted = _extract_by_path(root, path)
        if extracted is not None:
            search_roots.append(extracted)

    for candidate_root in search_roots:
        for candidate in _iter_objects_for_export(candidate_root):
            if _is_save_pretrained_object(candidate):
                return candidate
    return None


def _find_exportable_tokenizer(root: Any) -> Any | None:
    for candidate in _iter_objects_for_export(root):
        for attr_name in ("tokenizer", "_tokenizer", "processor", "_processor"):
            if hasattr(candidate, attr_name):
                item = getattr(candidate, attr_name)
                if _is_save_pretrained_object(item):
                    return item
        if _is_save_pretrained_object(candidate):
            type_name = type(candidate).__name__.lower()
            if "tokenizer" in type_name or "processor" in type_name:
                return candidate
    return None


def extract_safetensors_from_predictor(
    predictor_dir: Path,
    export_dir: Path,
    role_label: str = "exported",
) -> ExportResult:
    """Extract safetensors model artifacts from an AutoGluon predictor directory.

    Locates the newest ``fine-tuned-ckpt`` directory under
    ``predictor_dir/models/``, copies ``config.json`` and
    ``model.safetensors`` into ``export_dir``, purges any forbidden pickle
    artifacts, validates the result, and writes ``export_manifest.json``.

    Args:
        predictor_dir: Root directory of the AutoGluon TimeSeriesPredictor.
        export_dir: Destination directory (created if absent).
        role_label: Human-readable label used in validation error messages.

    Returns:
        ExportResult describing the completed extraction.

    Raises:
        SafetensorsExtractionError: If checkpoint discovery, copy, validation,
            or manifest write fails for any reason.
    """
    try:
        _reset_export_dir(export_dir)
        predictor = TimeSeriesPredictor.load(str(predictor_dir))
        checkpoint_dir = None
        copied_files: list[Path] = []

        model_obj = _find_exportable_model(predictor)
        if model_obj is not None:
            try:
                model_obj.save_pretrained(str(export_dir), safe_serialization=True)
                tokenizer_or_processor = _find_exportable_tokenizer(predictor)
                if tokenizer_or_processor is not None:
                    tokenizer_or_processor.save_pretrained(str(export_dir))
                if (export_dir / "config.json").exists():
                    copied_files.append(export_dir / "config.json")
                if (export_dir / "model.safetensors").exists():
                    copied_files.append(export_dir / "model.safetensors")
            except Exception:
                success, message, checkpoint_dir = copy_checkpoint_to_export(
                    predictor_dir=predictor_dir,
                    export_dir=export_dir,
                )
                if not success:
                    raise SafetensorsExtractionError(
                        f"Safetensors extraction failed for {role_label!r}: {message}"
                    )
                copied_files = [export_dir / "config.json", export_dir / "model.safetensors"]
        else:
            success, message, checkpoint_dir = copy_checkpoint_to_export(
                predictor_dir=predictor_dir,
                export_dir=export_dir,
            )
            if not success:
                raise SafetensorsExtractionError(
                    f"Safetensors extraction failed for {role_label!r}: {message}"
                )
            copied_files = [export_dir / "config.json", export_dir / "model.safetensors"]

        purged = purge_forbidden_artifacts(export_dir)

        validate_safetensors_only_model_dir(export_dir, role_label=role_label)
        exported_at = datetime.now(timezone.utc)
        manifest = ExportManifest.create(
            role_label=role_label,
            predictor_dir=predictor_dir,
            export_dir=export_dir,
            checkpoint_dir=checkpoint_dir,
            copied_files=copied_files,
            purged_files=purged,
            exported_at=exported_at,
        )

        manifest_path = export_dir / "export_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest.model_dump(), indent=2),
            encoding="utf-8",
        )

        return ExportResult(
            success=True,
            export_dir=str(export_dir),
            manifest=manifest,
        )

    except SafetensorsExtractionError:
        raise
    except Exception as exc:
        raise SafetensorsExtractionError(
            f"Unexpected error during safetensors extraction for {role_label!r}: {exc}"
        ) from exc
