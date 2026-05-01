"""
S3MetricsPublisher: idempotent upload of finalized metrics artifacts.

Each file is skipped if an identical object already exists in S3 (checked
by sha256 stored in object metadata). A failed individual upload is recorded
but does not abort remaining uploads. The caller decides whether to treat
partial publish as failure.

The training failure status is always the primary run outcome; publish errors
are secondary diagnostics captured in PublishResult.error.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import boto3
import botocore.exceptions

from .schemas import ManifestEntry, PublishManifest, PublishResult, RunSummary
from .local_sink import LocalMetricsSink

logger = logging.getLogger(__name__)


class S3MetricsPublisher:
    """
    Uploads entries listed in a PublishManifest to S3 under:
        s3://<bucket>/<prefix>/<run_id>/<filename>

    Idempotency: if an object already exists with a matching sha256 metadata
    header, the upload is skipped without error.
    """

    def __init__(self, bucket: str, prefix: str) -> None:
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._client = boto3.client("s3")

    def publish(
        self,
        manifest: PublishManifest,
        run_dir: Path,
        sink: LocalMetricsSink,
    ) -> PublishResult:
        """
        Upload all manifest entries. Always attempts every file even if earlier
        uploads fail. Returns a PublishResult that the caller merges into
        run_summary.json.
        """
        published_at = datetime.now(timezone.utc).isoformat()
        errors: List[str] = []
        uploaded: int = 0

        for entry in manifest.entries:
            local_path = run_dir / entry.filename
            if not local_path.exists():
                errors.append(f"missing_local: {entry.filename}")
                logger.warning("publish: local file missing, skipping: %s", entry.filename)
                continue
            s3_key = f"{self._prefix}/{entry.s3_key}"
            try:
                self._put_idempotent(local_path, s3_key, entry.sha256)
                uploaded += 1
            except Exception as exc:
                msg = f"{entry.filename}: {exc}"
                errors.append(msg)
                logger.warning("publish: upload failed key=%s error=%s", s3_key, exc)

        success = len(errors) == 0
        error_str = "; ".join(errors) if errors else None
        s3_prefix_uri = f"s3://{self._bucket}/{self._prefix}/{manifest.run_id}"

        result = PublishResult(
            success=success,
            s3_prefix=s3_prefix_uri,
            published_at=published_at,
            error=error_str,
            files_uploaded=uploaded,
        )

        self._write_publish_result_to_summary(
            sink=sink,
            manifest=manifest,
            result=result,
            run_dir=run_dir,
        )

        return result

    def _put_idempotent(self, local_path: Path, s3_key: str, sha256: str) -> None:
        try:
            resp = self._client.head_object(Bucket=self._bucket, Key=s3_key)
            existing_sha = resp.get("Metadata", {}).get("sha256", "")
            if existing_sha == sha256:
                logger.debug("publish: skip existing key=%s", s3_key)
                return
        except botocore.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] not in ("404", "NoSuchKey"):
                raise

        self._client.upload_file(
            str(local_path),
            self._bucket,
            s3_key,
            ExtraArgs={"Metadata": {"sha256": sha256}},
        )
        logger.info(
            "publish: uploaded key=%s bytes=%d",
            s3_key,
            local_path.stat().st_size,
        )

    @staticmethod
    def _write_publish_result_to_summary(
        sink: LocalMetricsSink,
        manifest: PublishManifest,
        result: PublishResult,
        run_dir: Path,
    ) -> None:
        """Merge publish outcome back into run_summary.json."""
        summary_path = run_dir / "run_summary.json"
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            raw = {}

        raw["publish_status"] = "success" if result.success else "failed"
        raw["published_at"] = result.published_at
        raw["s3_prefix"] = result.s3_prefix
        raw["publish_error"] = result.error

        import os

        tmp = summary_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, default=str)
        os.replace(tmp, summary_path)

        # Also patch manifest.json with publish outcome
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    mraw = json.load(f)
                mraw["publish_status"] = "success" if result.success else "failed"
                mraw["published_at"] = result.published_at
                mraw["publish_error"] = result.error
                tmp_m = manifest_path.with_suffix(".tmp")
                with open(tmp_m, "w", encoding="utf-8") as f:
                    json.dump(mraw, f, indent=2, default=str)
                os.replace(tmp_m, manifest_path)
            except Exception as exc:
                logger.warning("publish: manifest patch error (ignored): %s", exc)
