"""
ResourceSampler: background thread for periodic system resource collection.

Runs as a daemon thread so it never blocks process exit. Errors in the
collection loop are silently swallowed to avoid disrupting training.
If psutil is not installed the sampler is disabled and start() is a no-op.
GPU metrics are collected via pynvml when available; torch.cuda memory is
used as a fallback if pynvml is absent.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

from .schemas import ResourceSample
from .local_sink import LocalMetricsSink

try:
    import psutil as _psutil

    _PSUTIL_AVAILABLE = True
except ImportError:
    _psutil = None  # type: ignore[assignment]
    _PSUTIL_AVAILABLE = False


class ResourceSampler:
    def __init__(
        self,
        run_id: str,
        sink: LocalMetricsSink,
        interval_s: float = 5.0,
    ) -> None:
        self._run_id = run_id
        self._sink = sink
        self._interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._baseline_disk: Any = None
        self._baseline_net: Any = None

    def start(self) -> None:
        if not _PSUTIL_AVAILABLE:
            return
        self._baseline_disk = _psutil.disk_io_counters()
        self._baseline_net = _psutil.net_io_counters()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="metrics-resource-sampler",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self._interval_s * 2, 10.0))

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            try:
                sample = self._collect()
                self._sink.append_resource_sample(sample)
            except Exception:
                pass

    def _collect(self) -> ResourceSample:
        cpu_pct: float = _psutil.cpu_percent(interval=None)
        mem = _psutil.virtual_memory()
        mem_used_mb: float = (mem.total - mem.available) / (1024 * 1024)

        disk_read_bytes: int = 0
        disk_write_bytes: int = 0
        try:
            disk_now = _psutil.disk_io_counters()
            if self._baseline_disk is not None and disk_now is not None:
                disk_read_bytes = max(
                    0, disk_now.read_bytes - self._baseline_disk.read_bytes
                )
                disk_write_bytes = max(
                    0, disk_now.write_bytes - self._baseline_disk.write_bytes
                )
        except Exception:
            pass

        net_rx_bytes: int = 0
        net_tx_bytes: int = 0
        try:
            net_now = _psutil.net_io_counters()
            if self._baseline_net is not None and net_now is not None:
                net_rx_bytes = max(
                    0, net_now.bytes_recv - self._baseline_net.bytes_recv
                )
                net_tx_bytes = max(
                    0, net_now.bytes_sent - self._baseline_net.bytes_sent
                )
        except Exception:
            pass

        gpu_util_pct: Optional[float] = None
        gpu_mem_mb: Optional[float] = None
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_util_pct = float(util.gpu)
            gpu_mem_mb = mem_info.used / (1024 * 1024)
        except Exception:
            try:
                import torch

                if torch.cuda.is_available():
                    gpu_mem_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
            except Exception:
                pass

        return ResourceSample(
            run_id=self._run_id,
            ts_utc=datetime.now(timezone.utc).isoformat(),
            cpu_pct=cpu_pct,
            mem_used_mb=mem_used_mb,
            gpu_util_pct=gpu_util_pct,
            gpu_mem_mb=gpu_mem_mb,
            disk_read_bytes=disk_read_bytes,
            disk_write_bytes=disk_write_bytes,
            net_rx_bytes=net_rx_bytes,
            net_tx_bytes=net_tx_bytes,
        )
