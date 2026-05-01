"""
LocalMetricsSink: filesystem writer for metrics artifacts.

JSONL files are opened in append mode for event streams.
JSON documents (summary, counters) are written atomically via
temp file + os.replace so a half-written file is never exposed.
All methods are thread-safe.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict

from .schemas import PhaseEvent, ResourceSample, RunSummary, TransferCounters


class LocalMetricsSink:
    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._phase_events_path = run_dir / "phase_events.jsonl"
        self._resource_samples_path = run_dir / "resource_samples.jsonl"
        self._run_summary_path = run_dir / "run_summary.json"
        self._transfer_counters_path = run_dir / "transfer_counters.json"
        self._lock = threading.Lock()

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def append_phase_event(self, event: PhaseEvent) -> None:
        self._append_jsonl(self._phase_events_path, event.model_dump())

    def append_resource_sample(self, sample: ResourceSample) -> None:
        self._append_jsonl(self._resource_samples_path, sample.model_dump())

    def write_run_summary(self, summary: RunSummary) -> None:
        self._atomic_write_json(self._run_summary_path, summary.model_dump())

    def write_transfer_counters(self, counters: TransferCounters) -> None:
        self._atomic_write_json(self._transfer_counters_path, counters.model_dump())

    def _append_jsonl(self, path: Path, record: Dict[str, Any]) -> None:
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")

    def _atomic_write_json(self, path: Path, data: Dict[str, Any]) -> None:
        tmp = path.with_suffix(".tmp")
        with self._lock:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp, path)
