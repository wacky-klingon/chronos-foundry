#!/usr/bin/env python3
"""
Training execution wrapper with state management and metrics collection.

Responsibilities:
- Update S3 state file via boto3
- Execute training via existing src/ code
- Collect metrics from training logs
- Generate training.json artifact
- Handle errors and update state accordingly

Usage:
    training_wrapper.py --config <config.yaml> --output <output_dir>
"""

import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import boto3
from botocore.exceptions import ClientError


class TrainingWrapper:
    """Wrapper for executing training with state management."""

    def __init__(self, bucket: str, environment: str, run_id: str):
        self.bucket = bucket
        self.environment = environment
        self.run_id = run_id
        self.s3_client = boto3.client("s3")
        self.state_key = f"{environment}/system-state.json"
        self.start_time = datetime.utcnow().isoformat()
        self.start_timestamp = time.time()

    def update_state(
        self,
        status: str,
        step: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update S3 state file with current status."""
        try:
            # Download current state
            response = self.s3_client.get_object(Bucket=self.bucket, Key=self.state_key)
            state = json.loads(response["Body"].read())

            # Update fields
            state["status"] = status
            if step:
                state["current_step"] = step
            if error_message:
                state["error_message"] = error_message
            state["timestamps"]["last_update"] = datetime.utcnow().isoformat()

            # Upload atomically (write to tmp, then copy)
            tmp_key = f"{self.state_key}.tmp"
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=tmp_key,
                Body=json.dumps(state, indent=2),
                ContentType="application/json",
            )

            # Copy tmp to final (atomic on S3)
            self.s3_client.copy_object(
                Bucket=self.bucket,
                CopySource={"Bucket": self.bucket, "Key": tmp_key},
                Key=self.state_key,
            )

            # Delete tmp
            self.s3_client.delete_object(Bucket=self.bucket, Key=tmp_key)

            print(f"State updated: status={status}, step={step}")

        except ClientError as e:
            print(f"WARNING: Failed to update state: {e}", file=sys.stderr)

    def execute_training(self, config_path: str, output_dir: Path) -> bool:
        """
        Execute training using existing chronos_trainer CLI.

        Returns:
            True if training succeeded, False otherwise
        """
        try:
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)

            # Execute training using chronos_trainer CLI
            cmd = [
                "/opt/venv/bin/python3",
                "-m",
                "chronos_trainer.cli",
                "train",
                "--config-file",
                config_path,
                "--model-path",
                str(output_dir),
                # Note: start-date and end-date should come from config file
                # or be passed as additional parameters if needed
            ]

            print(f"Executing training: {' '.join(cmd)}")

            # Run training with live output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Stream output
            for line in process.stdout:
                print(line, end="")

            # Wait for completion
            return_code = process.wait()

            if return_code != 0:
                print(
                    f"ERROR: Training failed with exit code {return_code}",
                    file=sys.stderr,
                )
                return False

            print("Training completed successfully")
            return True

        except Exception as e:
            print(f"ERROR: Training execution failed: {e}", file=sys.stderr)
            return False

    def collect_metrics(self, output_dir: Path) -> Dict[str, Any]:
        """
        Collect training metrics from output directory.

        Looks for:
        - model files
        - log files
        - training artifacts
        """
        metrics = {
            "run_id": self.run_id,
            "start_time": self.start_time,
            "end_time": datetime.utcnow().isoformat(),
            "duration_seconds": round(time.time() - self.start_timestamp, 2),
        }

        # Check for model files
        model_files = list(output_dir.glob("**/*.pkl")) + list(
            output_dir.glob("**/*.safetensors")
        )
        if model_files:
            metrics["model_files"] = [
                str(f.relative_to(output_dir)) for f in model_files
            ]
            metrics["model_count"] = len(model_files)

        # Check for log files
        log_files = list(output_dir.glob("**/*.log")) + list(
            output_dir.glob("**/*.txt")
        )
        if log_files:
            metrics["log_files"] = [str(f.relative_to(output_dir)) for f in log_files]

        # Try to extract metrics from logs (simple parsing)
        try:
            predictor_log = output_dir / "logs" / "predictor_log.txt"
            if predictor_log.exists():
                with open(predictor_log) as f:
                    log_content = f.read()
                    # Simple metric extraction (customize based on actual log format)
                    if "loss" in log_content.lower():
                        metrics["contains_loss_metrics"] = True
        except Exception:
            pass

        return metrics

    def save_training_json(self, output_dir: Path, metrics: Dict[str, Any]) -> None:
        """Save training.json artifact."""
        training_json = output_dir / "training.json"
        with open(training_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved training metrics: {training_json}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Training execution wrapper")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument(
        "--output", required=True, help="Output directory for models/logs"
    )
    parser.add_argument(
        "--bucket", required=False, help="S3 bucket name (from env if not provided)"
    )
    parser.add_argument(
        "--environment", required=False, help="Environment (from env if not provided)"
    )
    parser.add_argument(
        "--run-id", required=False, help="Run ID (from env if not provided)"
    )

    args = parser.parse_args()

    # Get configuration from args or environment
    import os

    bucket = args.bucket or os.environ.get("BUCKET_NAME")
    environment = args.environment or os.environ.get("ENVIRONMENT", "dev")
    run_id = args.run_id or os.environ.get("RUN_ID")

    if not bucket:
        print("ERROR: BUCKET_NAME not provided", file=sys.stderr)
        sys.exit(1)

    if not run_id:
        print("ERROR: RUN_ID not provided", file=sys.stderr)
        sys.exit(1)

    # Initialize wrapper
    wrapper = TrainingWrapper(bucket, environment, run_id)
    output_dir = Path(args.output)

    try:
        # Update state: starting training
        wrapper.update_state("training", "epoch_0")

        # Execute training
        success = wrapper.execute_training(args.config, output_dir)

        if not success:
            wrapper.update_state(
                "failed",
                "training_error",
                "Training process returned non-zero exit code",
            )
            sys.exit(1)

        # Collect metrics
        print("\nCollecting training metrics...")
        metrics = wrapper.collect_metrics(output_dir)

        # Save training.json
        wrapper.save_training_json(output_dir, metrics)

        # Update state: training completed
        wrapper.update_state("training", "completed")

        print("\n" + "=" * 60)
        print("Training wrapper completed successfully")
        print("=" * 60)

        sys.exit(0)

    except Exception as e:
        print(f"\nERROR: Training wrapper failed: {e}", file=sys.stderr)
        wrapper.update_state("failed", "training_error", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
