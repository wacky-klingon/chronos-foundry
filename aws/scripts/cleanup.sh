#!/bin/bash
#
# cleanup.sh - Resource Cleanup and Instance Termination
#
# This script runs on the EC2 instance after training completes (or fails).
# It performs cleanup operations and terminates the instance.
#
# IMPORTANT: This script should NEVER fail completely - always attempt
# to terminate the instance even if some cleanup steps fail.
#
# Expected Environment Variables:
#   BUCKET_NAME      - S3 bucket name
#   ENVIRONMENT      - dev, stage, or prod
#   RUN_ID           - Unique run identifier
#   INSTANCE_ID      - EC2 instance ID
#   AWS_REGION       - AWS region
#
# Exit Codes:
#   Always exits 0 (never blocks termination)

set -uo pipefail  # Note: NOT using -e (errexit) to continue on errors

# ============================================================================
# Configuration
# ============================================================================

LOG_FILE="/var/log/cleanup.log"
CLEANUP_STATUS_FILE="/tmp/cleanup_status.json"

# ============================================================================
# Logging Setup
# ============================================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "$LOG_FILE" >&2
}

# ============================================================================
# Cleanup Status Tracking
# ============================================================================

init_cleanup_status() {
    cat > "$CLEANUP_STATUS_FILE" <<EOF
{
  "run_id": "${RUN_ID:-unknown}",
  "instance_id": "${INSTANCE_ID:-unknown}",
  "cleanup_start": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "steps": {}
}
EOF
}

update_cleanup_status() {
    local step=$1
    local status=$2  # success, failed, skipped
    local message=${3:-""}

    # Use jq to update status file
    if command -v jq &> /dev/null; then
        local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        jq --arg step "$step" \
           --arg status "$status" \
           --arg msg "$message" \
           --arg ts "$timestamp" \
           '.steps[$step] = {"status": $status, "message": $msg, "timestamp": $ts}' \
           "$CLEANUP_STATUS_FILE" > "${CLEANUP_STATUS_FILE}.tmp"
        mv "${CLEANUP_STATUS_FILE}.tmp" "$CLEANUP_STATUS_FILE"
    fi
}

finalize_cleanup_status() {
    if command -v jq &> /dev/null; then
        jq --arg ts "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
           '.cleanup_end = $ts' \
           "$CLEANUP_STATUS_FILE" > "${CLEANUP_STATUS_FILE}.tmp"
        mv "${CLEANUP_STATUS_FILE}.tmp" "$CLEANUP_STATUS_FILE"
    fi
}

# ============================================================================
# Main Cleanup Function
# ============================================================================

main() {
    log "=========================================="
    log "Starting Cleanup"
    log "=========================================="
    log "Run ID: ${RUN_ID:-unknown}"
    log "Instance: ${INSTANCE_ID:-unknown}"
    log "=========================================="

    # Initialize cleanup status
    init_cleanup_status

    # Step 1: Upload cleanup status to S3 (before other cleanups)
    upload_cleanup_status_initial

    # Step 2: Verify EBS volume auto-deletion
    verify_ebs_deletion

    # Step 3: Delete state file from S3
    delete_state_file

    # Step 4: Upload final logs
    upload_final_logs

    # Step 5: Upload final cleanup status
    upload_cleanup_status_final

    # Step 6: Terminate instance
    terminate_instance

    log "=========================================="
    log "Cleanup sequence completed"
    log "=========================================="
}

# ============================================================================
# Cleanup Steps
# ============================================================================

upload_cleanup_status_initial() {
    log "Uploading initial cleanup status..."

    if [[ -z "${BUCKET_NAME:-}" ]] || [[ -z "${ENVIRONMENT:-}" ]] || [[ -z "${RUN_ID:-}" ]]; then
        log_error "Missing required environment variables for S3 upload"
        update_cleanup_status "upload_initial_status" "failed" "Missing env vars"
        return 1
    fi

    if aws s3 cp "$CLEANUP_STATUS_FILE" \
        "s3://${BUCKET_NAME}/${ENVIRONMENT}/logs/${RUN_ID}/cleanup_status.json" \
        2>&1 | tee -a "$LOG_FILE"; then
        log "✓ Initial cleanup status uploaded"
        update_cleanup_status "upload_initial_status" "success" ""
    else
        log_error "Failed to upload initial cleanup status (non-blocking)"
        update_cleanup_status "upload_initial_status" "failed" "S3 upload error"
    fi
}

verify_ebs_deletion() {
    log "Verifying EBS volume auto-deletion..."

    if [[ -z "${INSTANCE_ID:-}" ]] || [[ -z "${AWS_REGION:-}" ]]; then
        log_error "Missing INSTANCE_ID or AWS_REGION"
        update_cleanup_status "verify_ebs" "skipped" "Missing env vars"
        return 1
    fi

    # Check if DeleteOnTermination is set for root volume
    if aws ec2 describe-instances \
        --instance-ids "$INSTANCE_ID" \
        --region "$AWS_REGION" \
        --query 'Reservations[0].Instances[0].BlockDeviceMappings[0].Ebs.DeleteOnTermination' \
        --output text 2>&1 | grep -q "True"; then
        log "✓ EBS volume configured for auto-deletion"
        update_cleanup_status "verify_ebs" "success" "DeleteOnTermination=true"
    else
        log_error "WARNING: EBS volume may not auto-delete (manual cleanup may be needed)"
        update_cleanup_status "verify_ebs" "failed" "DeleteOnTermination=false or check failed"
    fi
}

delete_state_file() {
    log "Deleting state file from S3..."

    if [[ -z "${BUCKET_NAME:-}" ]] || [[ -z "${ENVIRONMENT:-}" ]]; then
        log_error "Missing BUCKET_NAME or ENVIRONMENT"
        update_cleanup_status "delete_state" "skipped" "Missing env vars"
        return 1
    fi

    local state_file="s3://${BUCKET_NAME}/${ENVIRONMENT}/system-state.json"

    if aws s3 rm "$state_file" 2>&1 | tee -a "$LOG_FILE"; then
        log "✓ State file deleted"
        update_cleanup_status "delete_state" "success" ""
    else
        log_error "Failed to delete state file (non-blocking)"
        update_cleanup_status "delete_state" "failed" "S3 delete error"
    fi
}

upload_final_logs() {
    log "Uploading final logs to S3..."

    if [[ -z "${BUCKET_NAME:-}" ]] || [[ -z "${ENVIRONMENT:-}" ]] || [[ -z "${RUN_ID:-}" ]]; then
        log_error "Missing required environment variables"
        update_cleanup_status "upload_logs" "skipped" "Missing env vars"
        return 1
    fi

    local logs_s3_path="s3://${BUCKET_NAME}/${ENVIRONMENT}/logs/${RUN_ID}/"

    # Upload cleanup log
    if [[ -f "$LOG_FILE" ]]; then
        aws s3 cp "$LOG_FILE" "${logs_s3_path}cleanup.log" 2>&1 | tee -a "$LOG_FILE" || true
    fi

    # Upload bootstrap log if exists
    if [[ -f "/var/log/bootstrap.log" ]]; then
        aws s3 cp "/var/log/bootstrap.log" "${logs_s3_path}bootstrap.log" 2>&1 | tee -a "$LOG_FILE" || true
    fi

    log "✓ Logs uploaded (best effort)"
    update_cleanup_status "upload_logs" "success" ""
}

upload_cleanup_status_final() {
    log "Uploading final cleanup status..."

    # Finalize status file
    finalize_cleanup_status

    if [[ -z "${BUCKET_NAME:-}" ]] || [[ -z "${ENVIRONMENT:-}" ]] || [[ -z "${RUN_ID:-}" ]]; then
        log_error "Missing required environment variables"
        update_cleanup_status "upload_final_status" "skipped" "Missing env vars"
        return 1
    fi

    if aws s3 cp "$CLEANUP_STATUS_FILE" \
        "s3://${BUCKET_NAME}/${ENVIRONMENT}/logs/${RUN_ID}/cleanup_status.json" \
        2>&1 | tee -a "$LOG_FILE"; then
        log "✓ Final cleanup status uploaded"
        update_cleanup_status "upload_final_status" "success" ""
    else
        log_error "Failed to upload final cleanup status (non-blocking)"
        update_cleanup_status "upload_final_status" "failed" "S3 upload error"
    fi
}

terminate_instance() {
    log "Terminating EC2 instance..."

    if [[ -z "${INSTANCE_ID:-}" ]] || [[ -z "${AWS_REGION:-}" ]]; then
        log_error "CRITICAL: Cannot terminate - INSTANCE_ID or AWS_REGION not set"
        log_error "MANUAL CLEANUP REQUIRED: Instance ${INSTANCE_ID:-unknown} in ${AWS_REGION:-unknown}"
        update_cleanup_status "terminate_instance" "failed" "Missing env vars - MANUAL CLEANUP REQUIRED"
        return 1
    fi

    log "Requesting termination of instance ${INSTANCE_ID}..."

    if aws ec2 terminate-instances \
        --instance-ids "$INSTANCE_ID" \
        --region "$AWS_REGION" \
        2>&1 | tee -a "$LOG_FILE"; then
        log "✓ Termination request sent successfully"
        log "Instance ${INSTANCE_ID} will terminate shortly"
        update_cleanup_status "terminate_instance" "success" "Termination requested"
    else
        log_error "CRITICAL: Failed to terminate instance"
        log_error "MANUAL CLEANUP REQUIRED: Terminate ${INSTANCE_ID} via AWS Console"
        update_cleanup_status "terminate_instance" "failed" "AWS API error - MANUAL CLEANUP REQUIRED"
        return 1
    fi
}

# ============================================================================
# Entry Point
# ============================================================================

# Note: We don't trap errors or use errexit - we want to continue cleanup
# even if individual steps fail

log "Cleanup script started"

# Verify we're running on EC2 (optional safety check)
if ! curl -s -m 5 http://169.254.169.254/latest/meta-data/instance-id &>/dev/null; then
    log "WARNING: Not running on EC2 (metadata service unavailable)"
fi

# Set defaults if env vars missing (for safety)
: "${INSTANCE_ID:=$(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "unknown")}"
: "${AWS_REGION:=$(curl -s http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null || echo "us-east-1")}"

# Run main cleanup
main

# Always exit 0 (even if cleanup steps failed)
log "Cleanup script exiting"
exit 0

