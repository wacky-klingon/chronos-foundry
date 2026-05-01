"""
Pydantic schemas for offline metrics artifacts.

All event, sample, summary, and publish result types are defined here.
No business logic; pure data contracts.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"

Phase = Literal["boot", "data_download", "model_acquire", "train", "validate", "cleanup"]
EventType = Literal["start", "end", "fail"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid4() -> str:
    return str(uuid.uuid4())


class PhaseEvent(BaseModel):
    event_id: str = Field(default_factory=_uuid4)
    run_id: str
    phase: Phase
    event_type: EventType
    ts_utc: str = Field(default_factory=_utc_now)
    duration_ms: Optional[int] = None
    status: str
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    context: Optional[Dict[str, str]] = None


class ResourceSample(BaseModel):
    sample_id: str = Field(default_factory=_uuid4)
    run_id: str
    ts_utc: str = Field(default_factory=_utc_now)
    cpu_pct: float
    mem_used_mb: float
    gpu_util_pct: Optional[float] = None
    gpu_mem_mb: Optional[float] = None
    disk_read_bytes: int
    disk_write_bytes: int
    net_rx_bytes: int
    net_tx_bytes: int


class TransferCounters(BaseModel):
    run_id: str
    bytes_in: int = 0
    bytes_out: int = 0
    updated_at: str = Field(default_factory=_utc_now)


class RunSummary(BaseModel):
    run_id: str
    parent_run_id: Optional[str] = None
    schema_version: str = SCHEMA_VERSION
    training_status: str = "unknown"
    started_at: str
    ended_at: Optional[str] = None
    total_duration_ms: Optional[int] = None
    phase_durations_ms: Dict[str, int] = Field(default_factory=dict)
    bytes_in: int = 0
    bytes_out: int = 0
    model_type: str = ""
    processed_files: int = 0
    total_files: int = 0
    publish_status: Optional[str] = None
    published_at: Optional[str] = None
    s3_prefix: Optional[str] = None
    publish_error: Optional[str] = None


class ManifestEntry(BaseModel):
    filename: str
    s3_key: str
    size_bytes: int
    sha256: str


class PublishManifest(BaseModel):
    run_id: str
    schema_version: str = SCHEMA_VERSION
    created_at: str = Field(default_factory=_utc_now)
    entries: List[ManifestEntry] = Field(default_factory=list)
    publish_status: str = "pending"
    published_at: Optional[str] = None
    publish_error: Optional[str] = None


class PublishResult(BaseModel):
    success: bool
    s3_prefix: str
    published_at: Optional[str] = None
    error: Optional[str] = None
    files_uploaded: int = 0
