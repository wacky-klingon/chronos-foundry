#!/usr/bin/env python3
"""
Test training script for AWS infrastructure validation.

This is a TEMPORARY script used for testing EC2 orchestration
before integrating the actual training CLI.

Usage:
    test_trainer.py --config <config.yaml> --output <output_dir>
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime


def check_cuda():
    """Check if CUDA is available."""
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking CUDA availability..."
    )

    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CUDA available: {gpu_count} GPU(s)"
            )
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   GPU {i}: {gpu_name}"
                )
            return True
        else:
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CUDA not available",
                file=sys.stderr,
            )
            return False
    except ImportError:
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] PyTorch not installed",
            file=sys.stderr,
        )
        return False
    except Exception as e:
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CUDA check failed: {e}",
            file=sys.stderr,
        )
        return False


def simulate_training(config_path: str, output_dir: Path):
    """Simulate training execution with logging."""

    print("=" * 80)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TEST TRAINER STARTED")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    # Run CUDA check
    if not check_cuda():
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] WARNING: CUDA not available, continuing anyway..."
        )

    print()

    # Simulate training loop
    for epoch in range(1, 4):  # 3 iterations
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting epoch {epoch}/3..."
        )
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   Loading data...")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   Training model...")
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   Loss: {1.0 / epoch:.4f}"
        )

        # Sleep for 4 seconds
        time.sleep(4)

        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch}/3 completed"
        )
        print()

    # Create dummy output files
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a dummy model file
    dummy_model = output_dir / "test_model.pkl"
    with open(dummy_model, "w") as f:
        f.write("# Dummy model file for testing\n")
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Created dummy model: {dummy_model}"
    )

    # Create a dummy log file
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    dummy_log = log_dir / "training.log"
    with open(dummy_log, "w") as f:
        f.write(f"Training started: {datetime.now()}\n")
        f.write("Epoch 1: loss=1.0000\n")
        f.write("Epoch 2: loss=0.5000\n")
        f.write("Epoch 3: loss=0.3333\n")
        f.write(f"Training completed: {datetime.now()}\n")
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Created dummy log: {dummy_log}"
    )

    print()
    print("=" * 80)
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TEST TRAINER COMPLETED SUCCESSFULLY"
    )
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test training script for AWS validation"
    )
    parser.add_argument("--config", required=True, help="Path to training config")
    parser.add_argument("--output", required=True, help="Output directory")

    args = parser.parse_args()

    try:
        simulate_training(args.config, Path(args.output))
        sys.exit(0)
    except Exception as e:
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
