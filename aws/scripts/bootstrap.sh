#!/bin/bash
#
# bootstrap.sh - EC2 Instance Bootstrap and Training Execution
#
# This script runs on the EC2 instance after launch (via user data).
# It performs system setup, downloads dependencies, runs training,
# syncs results, and triggers cleanup/termination.
#
# Expected Environment Variables:
#   BUCKET_NAME      - S3 bucket name
#   ENVIRONMENT      - dev, stage, or prod
#   RUN_ID           - Unique run identifier (YYYYMMDD_HHMMSS)
#   INSTANCE_ID      - EC2 instance ID
#   AWS_REGION       - AWS region
#
# Exit Codes:
#   0   - Success (normal path, will self-terminate)
#   1   - Fatal error (will attempt cleanup and terminate)

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/bootstrap.log"
VENV_PATH="/opt/venv"
REPO_PATH="/opt/chronos-foundry"
DATA_PATH="/data"
CONFIG_PATH="${DATA_PATH}/config.yaml"
OUTPUT_PATH="${DATA_PATH}/output"

# Source state helpers (assuming uploaded to /opt/scripts/)
if [[ -f "/opt/scripts/lib/state_helpers.sh" ]]; then
    source /opt/scripts/lib/state_helpers.sh
else
    echo "ERROR: state_helpers.sh not found" >&2
    exit 1
fi

# ============================================================================
# Logging Setup
# ============================================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "$LOG_FILE" >&2
}

# Redirect all output to log file (but also show on console)
exec > >(tee -a "$LOG_FILE")
exec 2>&1

# ============================================================================
# Main Bootstrap Function
# ============================================================================

main() {
    log "=========================================="
    log "Chronos Training Bootstrap"
    log "=========================================="
    log "Run ID: ${RUN_ID}"
    log "Environment: ${ENVIRONMENT}"
    log "Instance: ${INSTANCE_ID}"
    log "Bucket: ${BUCKET_NAME}"
    log "=========================================="

    # Update state: initialization
    atomic_write_state "status=running" "current_step=initialization"

    # Step 1: System Setup
    log "Step 1: System setup..."
    atomic_write_state "current_step=system_setup"
    system_setup

    # Step 2: Download Python Environment
    log "Step 2: Downloading Python environment..."
    atomic_write_state "current_step=venv_download"
    download_venv

    # Step 3: GPU Preflight Checks
    log "Step 3: Running GPU preflight checks..."
    atomic_write_state "current_step=gpu_preflight"
    run_preflight

    # Step 4: Sync Training Data
    log "Step 4: Syncing training data from S3..."
    atomic_write_state "current_step=data_sync"
    sync_data

    # Step 5: Execute Training
    log "Step 5: Executing training..."
    atomic_write_state "current_step=training"
    execute_training

    # Step 6: Sync Results
    log "Step 6: Syncing results to S3..."
    atomic_write_state "current_step=results_sync"
    sync_results

    # Step 7: Cleanup and Terminate
    log "Step 7: Running cleanup..."
    atomic_write_state "current_step=cleanup"
    run_cleanup

    log "=========================================="
    log "Bootstrap completed successfully"
    log "=========================================="
}

# ============================================================================
# Step Functions
# ============================================================================

system_setup() {
    log "Installing system packages..."

    # Amazon Linux 2023 uses dnf/yum
    # Update package list
    retry_with_backoff yum update -y

    # Install required packages
    retry_with_backoff yum install -y \
        python3.11 \
        python3-pip \
        awscli \
        jq \
        git \
        curl \
        unzip \
        tar \
        gzip

    # Check if NVIDIA driver is already installed
    if ! command -v nvidia-smi &> /dev/null; then
        log "Installing NVIDIA driver..."
        # For Amazon Linux 2023, NVIDIA drivers may come pre-installed on GPU instances
        # If not, install from EPEL or NVIDIA repositories
        # Note: On AL2023 with g4dn instances, drivers are typically available
        # via instance-specific AMIs or can be installed from NVIDIA's repository
        if yum list available nvidia-driver &>/dev/null; then
            retry_with_backoff yum install -y nvidia-driver
        else
            log "NVIDIA driver not available via yum. Checking if instance has GPU support..."
            # GPU instances typically have drivers pre-installed
            if lspci | grep -i nvidia > /dev/null; then
                log "NVIDIA GPU detected but driver not found. Attempting manual installation..."
                # This would require more complex driver installation
                # For MVP, we assume drivers are pre-installed on GPU instances
            fi
        fi

        # Restart nvidia-persistenced if installed
        if systemctl list-units --type=service | grep -q nvidia-persistenced; then
            systemctl restart nvidia-persistenced || true
        fi
    else
        log "NVIDIA driver already installed"
    fi

    # Create data directory
    mkdir -p "$DATA_PATH"
    mkdir -p "$OUTPUT_PATH"

    log "✓ System setup complete"
}

download_venv() {
    log "Downloading pre-built Python environment..."

    # Download venv tarball from S3
    local venv_s3_path="s3://${BUCKET_NAME}/cached-datasets/python-env/chronos-venv-3.11.13.tar.gz"
    local venv_local="/tmp/venv.tar.gz"

    retry_with_backoff aws s3 cp "$venv_s3_path" "$venv_local"

    # Extract to /opt/venv
    log "Extracting virtual environment..."
    mkdir -p "$VENV_PATH"
    tar -xzf "$venv_local" -C "$VENV_PATH"

    # Verify extraction
    if [[ ! -f "${VENV_PATH}/bin/python3" ]]; then
        log_error "Virtual environment extraction failed"
        atomic_write_state "status=failed" "error_type=terminal" "error_message=venv extraction failed"
        exit 1
    fi

    # Clean up tarball
    rm -f "$venv_local"

    log "✓ Virtual environment ready"
}

run_preflight() {
    log "Running GPU preflight checks..."

    # Download training scripts from S3
    local scripts_s3_path="s3://${BUCKET_NAME}/cached-datasets/scripts/"
    mkdir -p /opt/scripts
    retry_with_backoff aws s3 sync "$scripts_s3_path" /opt/scripts/
    chmod +x /opt/scripts/*.sh /opt/scripts/*.py

    # Run preflight check
    if ! "${VENV_PATH}/bin/python3" /opt/scripts/preflight_check.py; then
        log_error "GPU preflight checks failed"
        atomic_write_state "status=failed" "error_type=terminal" "error_message=GPU preflight failed"
        exit 1
    fi

    log "✓ GPU preflight passed"
}

sync_data() {
    log "Syncing cached datasets..."

    # Sync cached datasets (read-only)
    local datasets_s3_path="s3://${BUCKET_NAME}/cached-datasets/training-data/"
    retry_with_backoff aws s3 sync "$datasets_s3_path" "${DATA_PATH}/"

    # Download run-specific config
    local config_s3_path="s3://${BUCKET_NAME}/${ENVIRONMENT}/${RUN_ID}/config.yaml"
    if aws s3 ls "$config_s3_path" &>/dev/null; then
        retry_with_backoff aws s3 cp "$config_s3_path" "$CONFIG_PATH"
    else
        log "WARNING: No run-specific config found, using defaults"
        # Use default config if available
        if [[ -f "${DATA_PATH}/default_config.yaml" ]]; then
            cp "${DATA_PATH}/default_config.yaml" "$CONFIG_PATH"
        fi
    fi

    log "✓ Data sync complete"
}

execute_training() {
    log "Starting training execution..."

    # Export environment variables for training_wrapper.py
    export BUCKET_NAME
    export ENVIRONMENT
    export RUN_ID

    # Execute training wrapper
    # training_wrapper.py will call chronos_trainer.cli internally
    if ! "${VENV_PATH}/bin/python3" /opt/scripts/training_wrapper.py \
        --config "$CONFIG_PATH" \
        --output "$OUTPUT_PATH"; then

        log_error "Training execution failed"
        atomic_write_state "status=failed" "error_type=terminal" "error_message=Training execution failed"
        exit 1
    fi

    log "✓ Training execution complete"
}

sync_results() {
    log "Uploading results to S3..."

    # Upload models
    local models_s3_path="s3://${BUCKET_NAME}/${ENVIRONMENT}/${RUN_ID}/models/"
    if [[ -d "${OUTPUT_PATH}" ]]; then
        retry_with_backoff aws s3 sync "$OUTPUT_PATH" "$models_s3_path"
    fi

    # Upload logs
    local logs_s3_path="s3://${BUCKET_NAME}/${ENVIRONMENT}/logs/${RUN_ID}/"
    if [[ -f "$LOG_FILE" ]]; then
        retry_with_backoff aws s3 cp "$LOG_FILE" "${logs_s3_path}bootstrap.log"
    fi

    # Upload training.json if exists
    if [[ -f "${OUTPUT_PATH}/training.json" ]]; then
        retry_with_backoff aws s3 cp "${OUTPUT_PATH}/training.json" \
            "s3://${BUCKET_NAME}/${ENVIRONMENT}/${RUN_ID}/training.json"
    fi

    log "✓ Results sync complete"
}

run_cleanup() {
    log "Running cleanup script..."

    # Execute cleanup.sh if it exists
    if [[ -f "/opt/scripts/cleanup.sh" ]]; then
        bash /opt/scripts/cleanup.sh
    else
        log "WARNING: cleanup.sh not found, performing basic cleanup"

        # Delete state file
        aws s3 rm "s3://${BUCKET_NAME}/${ENVIRONMENT}/system-state.json" || true

        # Self-terminate instance
        log "Terminating instance ${INSTANCE_ID}..."
        aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION" || true
    fi

    log "✓ Cleanup complete"
}

# ============================================================================
# Error Handler
# ============================================================================

error_handler() {
    local exit_code=$?
    local line_no=$1

    log_error "Script failed at line $line_no with exit code $exit_code"

    # Attempt to update state
    atomic_write_state "status=failed" "error_type=terminal" \
        "error_message=Bootstrap failed at line $line_no"

    # Attempt cleanup
    run_cleanup || true

    exit "$exit_code"
}

trap 'error_handler $LINENO' ERR

# ============================================================================
# Entry Point
# ============================================================================

# Verify required environment variables
if [[ -z "${BUCKET_NAME:-}" ]]; then
    log_error "BUCKET_NAME not set"
    exit 1
fi

if [[ -z "${ENVIRONMENT:-}" ]]; then
    log_error "ENVIRONMENT not set"
    exit 1
fi

if [[ -z "${RUN_ID:-}" ]]; then
    log_error "RUN_ID not set"
    exit 1
fi

# Get instance metadata if not set
if [[ -z "${INSTANCE_ID:-}" ]]; then
    log "Fetching instance ID from metadata..."
    INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id || echo "")
    if [[ -z "$INSTANCE_ID" ]]; then
        log_error "INSTANCE_ID not set and could not fetch from metadata"
        exit 1
    fi
    export INSTANCE_ID
fi

# Get AWS region from metadata if not set
if [[ -z "${AWS_REGION:-}" ]]; then
    log "Fetching AWS region from metadata..."
    AWS_REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region || echo "us-east-1")
    export AWS_REGION
fi

# Run main bootstrap
main

exit 0
