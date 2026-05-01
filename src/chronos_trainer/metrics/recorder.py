"""
MetricsRecorder: single interface for training code to emit metrics events.

Training code calls start_phase / end_phase / fail_phase at lifecycle boundaries
and add_bytes_in / add_bytes_out for transfer accounting. All storage concerns
are delegated to LocalMetricsSink; this class holds only timing state and counters.

NullMetricsRecorder provides a no-op implementation of the same interface so
callers that receive an Optional recorder need not guard every call site.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Dict, Optional

from .local_sink import LocalMetricsSink
from .resource_sampler import ResourceSampler
from .schemas import PhaseEvent, RunSummary, TransferCounters


class NullMetricsRecorder:
    """No-op recorder used when metrics collection is disabled."""

    def start_phase(self, phase: str, context: Optional[Dict[str, str]] = None) -> None:
        pass

    def end_phase(self, phase: str, context: Optional[Dict[str, str]] = None) -> None:
        pass

    def fail_phase(
        self,
        phase: str,
        error: Exception,
        context: Optional[Dict[str, str]] = None,
    ) -> None:
        pass

    def add_bytes_in(self, n: int) -> None:
        pass

    def add_bytes_out(self, n: int) -> None:
        pass

    def set_file_counts(self, processed: int, total: int) -> None:
        pass

    def set_training_status(self, status: str) -> None:
        pass

    def stop_sampler(self) -> None:
        pass

    def flush_summary(self, ended_at: Optional[str] = None) -> None:
        pass


class MetricsRecorder(NullMetricsRecorder):
    """
    Active recorder backed by a LocalMetricsSink.

    Lifecycle: create -> (training code calls phase/counter methods) ->
    caller must call stop_sampler() + flush_summary() in its finally block,
    then hand off to MetricsFinalizer.
    """

    def __init__(
        self,
        run_id: str,
        sink: LocalMetricsSink,
        model_type: str = "",
        parent_run_id: Optional[str] = None,
        resource_sample_interval_s: float = 5.0,
    ) -> None:
        self._run_id = run_id
        self._sink = sink
        self._model_type = model_type
        self._parent_run_id = parent_run_id
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._run_start_mono: float = time.monotonic()
        self._phase_starts: Dict[str, float] = {}
        self._phase_durations_ms: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._bytes_in: int = 0
        self._bytes_out: int = 0
        self._processed_files: int = 0
        self._total_files: int = 0
        self._training_status: str = "running"
        self._sampler = ResourceSampler(run_id, sink, interval_s=resource_sample_interval_s)
        self._sampler.start()
        self._flush_summary_internal()

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def sink(self) -> LocalMetricsSink:
        return self._sink

    def start_phase(self, phase: str, context: Optional[Dict[str, str]] = None) -> None:
        with self._lock:
            self._phase_starts[phase] = time.monotonic()
        event = PhaseEvent(
            run_id=self._run_id,
            phase=phase,  # type: ignore[arg-type]
            event_type="start",
            status="started",
            context=context,
        )
        self._sink.append_phase_event(event)

    def end_phase(self, phase: str, context: Optional[Dict[str, str]] = None) -> None:
        duration_ms = self._record_duration(phase)
        event = PhaseEvent(
            run_id=self._run_id,
            phase=phase,  # type: ignore[arg-type]
            event_type="end",
            status="completed",
            duration_ms=duration_ms,
            context=context,
        )
        self._sink.append_phase_event(event)

    def fail_phase(
        self,
        phase: str,
        error: Exception,
        context: Optional[Dict[str, str]] = None,
    ) -> None:
        duration_ms = self._record_duration(phase)
        event = PhaseEvent(
            run_id=self._run_id,
            phase=phase,  # type: ignore[arg-type]
            event_type="fail",
            status="failed",
            duration_ms=duration_ms,
            error_type=type(error).__name__,
            error_message=str(error)[:2000],
            context=context,
        )
        self._sink.append_phase_event(event)

    def add_bytes_in(self, n: int) -> None:
        with self._lock:
            self._bytes_in += n

    def add_bytes_out(self, n: int) -> None:
        with self._lock:
            self._bytes_out += n

    def set_file_counts(self, processed: int, total: int) -> None:
        with self._lock:
            self._processed_files = processed
            self._total_files = total

    def set_training_status(self, status: str) -> None:
        with self._lock:
            self._training_status = status

    def stop_sampler(self) -> None:
        self._sampler.stop()

    def flush_summary(self, ended_at: Optional[str] = None) -> None:
        self._flush_summary_internal(ended_at=ended_at)

    def build_run_summary(self, ended_at: Optional[str] = None) -> RunSummary:
        """Return a RunSummary snapshot for use by the finalizer."""
        with self._lock:
            elapsed_ms = int((time.monotonic() - self._run_start_mono) * 1000)
            return RunSummary(
                run_id=self._run_id,
                parent_run_id=self._parent_run_id,
                training_status=self._training_status,
                started_at=self._started_at,
                ended_at=ended_at or datetime.now(timezone.utc).isoformat(),
                total_duration_ms=elapsed_ms,
                phase_durations_ms=dict(self._phase_durations_ms),
                bytes_in=self._bytes_in,
                bytes_out=self._bytes_out,
                model_type=self._model_type,
                processed_files=self._processed_files,
                total_files=self._total_files,
            )

    def _record_duration(self, phase: str) -> Optional[int]:
        with self._lock:
            start = self._phase_starts.get(phase)
            if start is not None:
                dur = int((time.monotonic() - start) * 1000)
                self._phase_durations_ms[phase] = dur
                return dur
        return None

    def _flush_summary_internal(self, ended_at: Optional[str] = None) -> None:
        summary = self.build_run_summary(ended_at=ended_at)
        self._sink.write_run_summary(summary)
        counters = TransferCounters(
            run_id=self._run_id,
            bytes_in=summary.bytes_in,
            bytes_out=summary.bytes_out,
        )
        self._sink.write_transfer_counters(counters)
