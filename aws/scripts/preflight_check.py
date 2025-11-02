#!/usr/bin/env python3
"""
GPU and environment preflight checks for EC2 training instances.

Validates:
- NVIDIA driver loaded
- CUDA available via PyTorch
- GPU count >= 1
- Python virtual environment valid

Exit codes:
- 0: All checks passed
- 1: One or more critical checks failed
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any


def check_nvidia_driver() -> bool:
    """Verify NVIDIA driver is loaded via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=10, check=False
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"ERROR: nvidia-smi check failed: {e}", file=sys.stderr)
        return False


def check_cuda() -> bool:
    """Verify CUDA is available via PyTorch."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError as e:
        print(f"ERROR: PyTorch import failed: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"ERROR: CUDA check failed: {e}", file=sys.stderr)
        return False


def get_gpu_count() -> int:
    """Get number of available GPUs."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return torch.cuda.device_count()
    except Exception:
        return 0


def get_gpu_names() -> list:
    """Get names of available GPUs."""
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def check_venv(venv_path: str = "/opt/venv") -> bool:
    """Verify Python virtual environment exists and is valid."""
    venv = Path(venv_path)

    # Check critical paths exist
    required_paths = [
        venv / "bin" / "python3",
        venv / "bin" / "pip",
        venv / "lib",
    ]

    for path in required_paths:
        if not path.exists():
            print(f"ERROR: Missing venv component: {path}", file=sys.stderr)
            return False

    return True


def check_disk_space(min_gb: int = 20) -> Dict[str, Any]:
    """Check available disk space on /data mount."""
    try:
        import shutil

        stat = shutil.disk_usage("/data")
        available_gb = stat.free / (1024**3)
        return {
            "available_gb": round(available_gb, 2),
            "sufficient": available_gb >= min_gb,
            "threshold_gb": min_gb,
        }
    except Exception as e:
        return {"available_gb": 0, "sufficient": False, "error": str(e)}


def main():
    """Run all preflight checks and report results."""
    print("=" * 60)
    print("GPU Preflight Checks")
    print("=" * 60)

    # Run all checks
    checks = {
        "nvidia_driver": check_nvidia_driver(),
        "cuda_available": check_cuda(),
        "gpu_count": get_gpu_count(),
        "gpu_names": get_gpu_names(),
        "venv_valid": check_venv(),
        "disk_space": check_disk_space(),
    }

    # Print results
    print(json.dumps(checks, indent=2))

    # Determine pass/fail
    critical_checks = [
        checks["nvidia_driver"],
        checks["cuda_available"],
        checks["venv_valid"],
        checks["disk_space"]["sufficient"],
    ]

    if not all(critical_checks):
        print("\n" + "=" * 60, file=sys.stderr)
        print("PREFLIGHT FAILED - Critical checks did not pass", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        if not checks["nvidia_driver"]:
            print("- NVIDIA driver not loaded", file=sys.stderr)
        if not checks["cuda_available"]:
            print("- CUDA not available", file=sys.stderr)
        if not checks["venv_valid"]:
            print("- Python venv invalid or missing", file=sys.stderr)
        if not checks["disk_space"]["sufficient"]:
            print(
                f"- Insufficient disk space: {checks['disk_space']['available_gb']}GB",
                file=sys.stderr,
            )

        sys.exit(1)

    if checks["gpu_count"] < 1:
        print("\n" + "=" * 60, file=sys.stderr)
        print("PREFLIGHT FAILED - No GPUs detected", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        sys.exit(1)

    # Success
    print("\n" + "=" * 60)
    print(f"PREFLIGHT PASSED - {checks['gpu_count']} GPU(s) ready")
    for i, name in enumerate(checks["gpu_names"]):
        print(f"  GPU {i}: {name}")
    print("=" * 60)

    sys.exit(0)


if __name__ == "__main__":
    main()
