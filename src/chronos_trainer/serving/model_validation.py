# Donor: chronos-finetuning model_validation.py @ git ac19b47
"""Validate offline model directories follow the safetensors-only policy.

The project supports full offline model artifacts only. A valid model directory
must contain ``config.json`` and ``model.safetensors``, and must not contain
pickle-based artifacts such as ``.pkl``, ``.pickle``, or ``.bin`` files.
"""

from __future__ import annotations

from pathlib import Path

REQUIRED_MODEL_FILES: tuple[str, ...] = ("config.json", "model.safetensors")
FORBIDDEN_ARTIFACT_SUFFIXES: tuple[str, ...] = (".pkl", ".pickle", ".bin")


def validate_safetensors_only_model_dir(model_dir: Path, role_label: str) -> None:
    """Validate ``model_dir`` for safetensors-only offline mode.

    Raises:
        FileNotFoundError: if directory or required files are missing.
        NotADirectoryError: if the path is not a directory.
        ValueError: if any forbidden pickle-based artifact is present.
    """
    if not model_dir.exists():
        raise FileNotFoundError(f"{role_label} model directory does not exist: {model_dir}")
    if not model_dir.is_dir():
        raise NotADirectoryError(f"{role_label} model path is not a directory: {model_dir}")

    for required_filename in REQUIRED_MODEL_FILES:
        candidate = model_dir / required_filename
        if not candidate.exists():
            raise FileNotFoundError(
                f"{role_label} model directory missing required safetensors artifact: {candidate}"
            )

    forbidden_files: list[Path] = []
    for path in model_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in FORBIDDEN_ARTIFACT_SUFFIXES:
            forbidden_files.append(path)
    if forbidden_files:
        forbidden_listing = ", ".join(
            str(file_path.relative_to(model_dir)) for file_path in forbidden_files[:10]
        )
        raise ValueError(
            f"{role_label} model directory contains pickle-based artifacts which are "
            f"not supported by the safetensors-only offline mode. "
            f"Forbidden suffixes: {FORBIDDEN_ARTIFACT_SUFFIXES}. "
            f"Found: {forbidden_listing}"
        )
