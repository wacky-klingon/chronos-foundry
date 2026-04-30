"""Pydantic schemas and exception class for safetensors export results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel


class SafetensorsExtractionError(Exception):
    """Raised when safetensors extraction from a predictor directory fails."""


class ExportManifest(BaseModel):
    role_label: str
    predictor_dir: str
    export_dir: str
    checkpoint_dir: Optional[str]
    copied_files: List[str]
    purged_files: List[str]
    exported_at: str

    @classmethod
    def create(
        cls,
        role_label: str,
        predictor_dir: Path,
        export_dir: Path,
        checkpoint_dir: Optional[Path],
        copied_files: List[Path],
        purged_files: List[Path],
        exported_at: datetime,
    ) -> "ExportManifest":
        return cls(
            role_label=role_label,
            predictor_dir=str(predictor_dir),
            export_dir=str(export_dir),
            checkpoint_dir=str(checkpoint_dir) if checkpoint_dir is not None else None,
            copied_files=[str(p) for p in copied_files],
            purged_files=[str(p) for p in purged_files],
            exported_at=exported_at.isoformat(),
        )


class ExportResult(BaseModel):
    success: bool
    export_dir: str
    manifest: ExportManifest
