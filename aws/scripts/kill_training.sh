#!/bin/bash
#
# kill_training.sh - Emergency Training Termination
#
# Terminates a running training job with cleanup verification.
# Runs on dev machine (not EC2).
#
# Usage:
#   ./kill_training.sh
#   ./kill_training.sh --force
#   ENVIRONMENT=stage ./kill_training.sh
#
# Environment Variables:
#   ENVIRONMENT       - dev, stage, or prod (default: dev)
#   BUCKET_NAME       - S3 bucket name (from .env if not set)
#   AWS_PROFILE       - AWS CLI profile (default: trainer-runtime)

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

ENVIRONMENT="${ENVIRONMENT:-dev}"
AWS_PROFILE="${AWS_PROFILE:-trainer-runtime}"
FORCE_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)
            FORCE_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--force]" >&2
            exit 1
            ;;
    esac
done

# Load environment variables from .env if available
if [[ -f "$(dirname "$0")/../cdk/.env" ]]; then
    export $(grep -v '^#' "$(dirname "$0")/../cdk/.env" | xargs)
fi

if [[ -z "${BUCKET_NAME:-}" ]]; then
    echo "ERROR: BUCKET_NAME not set. Set in .env or environment." >&2
    exit 1
fi

if [[ -z "${AWS_REGION:-}" ]]; then
    AWS_REGION=$(aws configure get region --profile "$AWS_PROFILE" 2>/dev/null || echo "us-east-1")
fi

STATE_FILE="s3://${BUCKET_NAME}/${ENVIRONMENT}/system-state.json"

# ============================================================================
# Color Codes
# ============================================================================

if [[ -t 1 ]]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
else
    GREEN=''
    RED=''
    YELLOW=''
    BLUE=''
    NC=''
fi

# ============================================================================
# Helper Functions
# ============================================================================

log() {
    echo -e "$*"
}

log_success() {
    echo -e "${GREEN}✓${NC} $*"
}

log_error() {
    echo -e "${RED}✗${NC} $*" >&2
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $*"
}

log_info() {
    echo -e "${BLUE}ℹ${NC} $*"
}

format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))

    if [[ $hours -gt 0 ]]; then
        printf "%dh %dm %ds" "$hours" "$minutes" "$secs"
    elif [[ $minutes -gt 0 ]]; then
        printf "%dm %ds" "$minutes" "$secs"
    else
        printf "%ds" "$secs"
    fi
}

calculate_elapsed() {
    local start_time=$1
    local current_time=$(date -u +%s)
    local start_epoch=$(date -d "$start_time" +%s 2>/dev/null || echo "$current_time")
    local elapsed=$((current_time - start_epoch))
    echo "$elapsed"
}

# ============================================================================
# State File Operations
# ============================================================================

fetch_state() {
    local state_json
    if ! state_json=$(aws s3 cp "$STATE_FILE" - --profile "$AWS_PROFILE" 2>/dev/null); then
        return 1
    fi
    echo "$state_json"
}

update_state_killed() {
    local state_json=$1

    # Update status to killed
    local updated_state=$(echo "$state_json" | jq \
        --arg ts "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
        '.status = "killed" |
         .timestamps.last_update = $ts |
         .error_message = "Terminated by user"')

    # Upload to S3
    echo "$updated_state" | aws s3 cp - "$STATE_FILE" --profile "$AWS_PROFILE"
}

# ============================================================================
# EC2 Operations
# ============================================================================

terminate_instance() {
    local instance_id=$1

    log_info "Terminating instance ${instance_id}..."

    if aws ec2 terminate-instances \
        --instance-ids "$instance_id" \
        --region "$AWS_REGION" \
        --profile "$AWS_PROFILE" \
        --output json > /dev/null; then
        log_success "Termination request sent"
        return 0
    else
        log_error "Failed to terminate instance"
        return 1
    fi
}

wait_for_termination() {
    local instance_id=$1
    local timeout=300  # 5 minutes
    local elapsed=0

    log_info "Waiting for instance to terminate (timeout: ${timeout}s)..."

    while [[ $elapsed -lt $timeout ]]; do
        local state=$(aws ec2 describe-instances \
            --instance-ids "$instance_id" \
            --region "$AWS_REGION" \
            --profile "$AWS_PROFILE" \
            --query 'Reservations[0].Instances[0].State.Name' \
            --output text 2>/dev/null || echo "unknown")

        if [[ "$state" == "terminated" ]]; then
            log_success "Instance terminated"
            return 0
        elif [[ "$state" == "unknown" ]]; then
            log_warning "Cannot query instance state (may already be terminated)"
            return 0
        fi

        echo -n "."
        sleep 10
        elapsed=$((elapsed + 10))
    done

    echo ""
    log_warning "Timeout waiting for termination (instance may still be terminating)"
    return 1
}

# ============================================================================
# Main Termination Logic
# ============================================================================

main() {
    log "${RED}========================================"
    log "Emergency Training Termination"
    log "========================================${NC}"
    log ""

    # Check if state file exists
    if ! aws s3 ls "$STATE_FILE" --profile "$AWS_PROFILE" &>/dev/null; then
        log_error "No training run found in ${ENVIRONMENT} environment"
        log_info "State file not found: ${STATE_FILE}"
        exit 1
    fi

    # Fetch current state
    local state_json
    if ! state_json=$(fetch_state); then
        log_error "Failed to fetch state file"
        exit 1
    fi

    # Parse state
    local run_id=$(echo "$state_json" | jq -r '.run_id // "unknown"')
    local status=$(echo "$state_json" | jq -r '.status // "unknown"')
    local instance_id=$(echo "$state_json" | jq -r '.instance_id // "unknown"')
    local start_time=$(echo "$state_json" | jq -r '.timestamps.start // "unknown"')

    # Display current state
    log "Run ID: ${run_id}"
    log "Instance: ${instance_id}"
    log "Current Status: ${status}"

    if [[ "$start_time" != "unknown" ]]; then
        local elapsed=$(calculate_elapsed "$start_time")
        log "Elapsed Time: $(format_duration $elapsed)"
    fi

    log ""

    # Check if already terminated
    if [[ "$status" == "killed" ]] || [[ "$status" == "failed" ]] || [[ "$status" == "cleanup" ]]; then
        log_warning "Training is already in terminal state: ${status}"
        read -p "Continue with termination anyway? (yes/no): " confirm
        if [[ "$confirm" != "yes" ]]; then
            log_info "Termination cancelled"
            exit 0
        fi
    fi

    # Confirm termination
    if [[ "$FORCE_MODE" == false ]]; then
        log_warning "WARNING: This will terminate the training run."
        log_warning "All unsaved progress will be lost."
        log ""
        read -p "Are you sure? (yes/no): " confirm

        if [[ "$confirm" != "yes" ]]; then
            log_info "Termination cancelled"
            exit 0
        fi
    fi

    log ""

    # Update state to killed
    log_info "Updating state to 'killed'..."
    if update_state_killed "$state_json"; then
        log_success "State updated"
    else
        log_error "Failed to update state (continuing anyway)"
    fi

    # Terminate instance
    if terminate_instance "$instance_id"; then
        # Wait for termination
        wait_for_termination "$instance_id"
    else
        log_error "Failed to terminate instance"
        log_info "You may need to terminate manually:"
        log "  aws ec2 terminate-instances --instance-ids ${instance_id} --region ${AWS_REGION} --profile ${AWS_PROFILE}"
        exit 1
    fi

    # Wait for cleanup
    log ""
    log_info "Waiting for cleanup to complete..."
    sleep 10

    # Check cleanup status
    local cleanup_status_file="s3://${BUCKET_NAME}/${ENVIRONMENT}/logs/${run_id}/cleanup_status.json"
    if aws s3 ls "$cleanup_status_file" --profile "$AWS_PROFILE" &>/dev/null; then
        log_success "Cleanup status available"
        aws s3 cp "$cleanup_status_file" - --profile "$AWS_PROFILE" 2>/dev/null | jq '.' || true
    else
        log_warning "Cleanup status not yet available"
    fi

    # Summary
    log ""
    log "${GREEN}========================================"
    log "Termination Complete"
    log "========================================${NC}"

    if [[ "$start_time" != "unknown" ]]; then
        local final_elapsed=$(calculate_elapsed "$start_time")
        log "Total runtime: $(format_duration $final_elapsed)"

        # Rough cost estimate (g4dn.2xlarge at $0.752/hour)
        local hours=$(echo "scale=2; $final_elapsed / 3600" | bc)
        local estimated_cost=$(echo "scale=2; $hours * 0.752" | bc)
        log "Estimated cost: \$${estimated_cost}"
    fi

    log ""
    log_info "Training run ${run_id} has been terminated"
}

# ============================================================================
# Entry Point
# ============================================================================

main

