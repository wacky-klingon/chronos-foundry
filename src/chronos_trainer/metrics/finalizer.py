"""
MetricsFinalizer: flush, compress, and manifest metrics artifacts.

Called unconditionally in the training entry-point finally block so artifacts
are always finalized on success, failure, and interrupted runs. The finalizer
does not perform S3 operations; it only prepares local artifacts for publish.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .local_sink import LocalMetricsSink
from .recorder import MetricsRecorder
from .schemas import ManifestEntry, PublishManifest, RunSummary

logger = logging.getLogger(__name__)

_JSONL_PUBLISH_NAMES = ("phase_events.jsonl", "resource_samples.jsonl")
_JSON_PUBLISH_NAMES = ("run_summary.json", "transfer_counters.json")


class MetricsFinalizer:
    """
    Drives the finalize sequence:

        stop sampler -> flush summary -> compress JSONL ->
        create archive -> build manifest -> write manifest.json
    """

    def __init__(self, recorder: MetricsRecorder, sink: LocalMetricsSink) -> None:
        self._recorder = recorder
        self._sink = sink
        self._run_dir = sink.run_dir

    def finalize(self, training_status: str) -> PublishManifest:
        """
        Stop sampling, write final summary, compress artifacts, and return
        a PublishManifest. Always runs to completion regardless of internal
        errors; individual step failures are logged but do not raise.
        """
        ended_at = datetime.now(timezone.utc).isoformat()

        try:
            self._recorder.set_training_status(training_status)
            self._recorder.stop_sampler()
        except Exception as exc:
            logger.warning("finalizer: stop_sampler error (ignored): %s", exc)

        summary: RunSummary
        try:
            summary = self._recorder.build_run_summary(ended_at=ended_at)
            self._sink.write_run_summary(summary)
        except Exception as exc:
            logger.warning("finalizer: write_run_summary error (ignored): %s", exc)
            summary = RunSummary(
                run_id=self._recorder.run_id,
                training_status=training_status,
                started_at=ended_at,
                ended_at=ended_at,
            )

        try:
            self._compress_jsonl_files()
        except Exception as exc:
            logger.warning("finalizer: compress error (ignored): %s", exc)

        try:
            self._create_archive()
        except Exception as exc:
            logger.warning("finalizer: archive error (ignored): %s", exc)

        manifest = self._build_manifest(self._recorder.run_id)

        try:
            manifest_path = self._run_dir / "manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest.model_dump(), f, indent=2, default=str)
        except Exception as exc:
            logger.warning("finalizer: write manifest error (ignored): %s", exc)

        return manifest

    def _compress_jsonl_files(self) -> None:
        for name in _JSONL_PUBLISH_NAMES:
            src = self._run_dir / name
            if not src.exists():
                continue
            dst = self._run_dir / (name + ".gz")
            with open(src, "rb") as f_in, gzip.open(dst, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            logger.debug("finalizer: compressed %s -> %s", src.name, dst.name)

    def _create_archive(self) -> None:
        archive_path = self._run_dir / "archive.tar.gz"
        excluded = {"archive.tar.gz"}
        with tarfile.open(archive_path, "w:gz") as tar:
            for path in sorted(self._run_dir.glob("*")):
                if path.name in excluded or not path.is_file():
                    continue
                tar.add(path, arcname=path.name)
        logger.debug("finalizer: archive created at %s", archive_path)

    def _build_manifest(self, run_id: str) -> PublishManifest:
        publish_targets: List[str] = (
            [n + ".gz" for n in _JSONL_PUBLISH_NAMES]
            + list(_JSON_PUBLISH_NAMES)
            + ["archive.tar.gz"]
        )
        entries: List[ManifestEntry] = []
        for name in publish_targets:
            path = self._run_dir / name
            if not path.exists():
                continue
            entries.append(
                ManifestEntry(
                    filename=name,
                    s3_key=f"{run_id}/{name}",
                    size_bytes=path.stat().st_size,
                    sha256=self._sha256(path),
                )
            )
        return PublishManifest(run_id=run_id, entries=entries)

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
