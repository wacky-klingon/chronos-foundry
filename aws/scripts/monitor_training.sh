#!/bin/bash
#
# monitor_training.sh - Training Progress Monitor
#
# Polls S3 state file and displays real-time training progress.
# Runs on dev machine (not EC2).
#
# Usage:
#   ./monitor_training.sh
#   ENVIRONMENT=stage ./monitor_training.sh
#   REFRESH_INTERVAL=60 ./monitor_training.sh
#
# Environment Variables:
#   ENVIRONMENT       - dev, stage, or prod (default: dev)
#   BUCKET_NAME       - S3 bucket name (from .env if not set)
#   AWS_PROFILE       - AWS CLI profile (default: trainer-runtime)
#   REFRESH_INTERVAL  - Seconds between polls (default: 30)

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

ENVIRONMENT="${ENVIRONMENT:-dev}"
AWS_PROFILE="${AWS_PROFILE:-trainer-runtime}"
REFRESH_INTERVAL="${REFRESH_INTERVAL:-30}"

# Load environment variables from .env if available
if [[ -f "$(dirname "$0")/../cdk/.env" ]]; then
    export $(grep -v '^#' "$(dirname "$0")/../cdk/.env" | xargs)
fi

if [[ -z "${BUCKET_NAME:-}" ]]; then
    echo "ERROR: BUCKET_NAME not set. Set in .env or environment." >&2
    exit 1
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

check_state_file_exists() {
    if aws s3 ls "$STATE_FILE" --profile "$AWS_PROFILE" &>/dev/null; then
        return 0
    else
        return 1
    fi
}

fetch_state() {
    local state_json
    if ! state_json=$(aws s3 cp "$STATE_FILE" - --profile "$AWS_PROFILE" 2>/dev/null); then
        return 1
    fi
    echo "$state_json"
}

parse_state() {
    local state_json=$1
    local field=$2
    echo "$state_json" | jq -r ".$field // \"unknown\""
}

# ============================================================================
# Display Functions
# ============================================================================

display_header() {
    clear
    log "${BLUE}========================================"
    log "Training Monitor - ${ENVIRONMENT} environment"
    log "========================================${NC}"
    log ""
}

display_state() {
    local state_json=$1

    local run_id=$(parse_state "$state_json" "run_id")
    local status=$(parse_state "$state_json" "status")
    local current_step=$(parse_state "$state_json" "current_step")
    local instance_id=$(parse_state "$state_json" "instance_id")
    local start_time=$(parse_state "$state_json" "timestamps.start")
    local last_update=$(parse_state "$state_json" "timestamps.last_update")
    local error_msg=$(parse_state "$state_json" "error_message")

    # Display based on status
    case "$status" in
        running)
            log_success "Status: ${GREEN}RUNNING${NC}"
            ;;
        failed)
            log_error "Status: ${RED}FAILED${NC}"
            ;;
        cleanup)
            log_info "Status: ${BLUE}CLEANUP${NC}"
            ;;
        *)
            log "Status: ${status}"
            ;;
    esac

    log "Run ID: ${run_id}"
    log "Step: ${current_step}"
    log "Instance: ${instance_id}"
    log ""

    # Display timestamps
    log "Started: ${start_time}"
    log "Last Update: ${last_update}"

    # Calculate elapsed time
    if [[ "$start_time" != "unknown" ]]; then
        local elapsed=$(calculate_elapsed "$start_time")
        log "Elapsed: $(format_duration $elapsed)"
    fi

    log ""

    # Display error if present
    if [[ "$error_msg" != "null" ]] && [[ "$error_msg" != "unknown" ]] && [[ -n "$error_msg" ]]; then
        log_error "Error: ${error_msg}"
        log ""
    fi
}

display_footer() {
    local status=$1

    if [[ "$status" == "failed" ]] || [[ "$status" == "cleanup" ]]; then
        log_info "Training has finished. Press Ctrl+C to exit."
    else
        log_info "Refreshing in ${REFRESH_INTERVAL}s... (Ctrl+C to exit)"
    fi
}

# ============================================================================
# Main Monitoring Loop
# ============================================================================

main() {
    log_info "Starting training monitor..."
    log_info "Environment: ${ENVIRONMENT}"
    log_info "Bucket: ${BUCKET_NAME}"
    log_info "Profile: ${AWS_PROFILE}"
    log ""

    # Check if state file exists
    if ! check_state_file_exists; then
        log_error "No training run found in ${ENVIRONMENT} environment"
        log_info "State file not found: ${STATE_FILE}"
        exit 1
    fi

    local previous_status=""
    local loop_count=0

    while true; do
        # Fetch state
        local state_json
        if ! state_json=$(fetch_state); then
            display_header
            log_error "Failed to fetch state file"
            log_info "Training may have completed and cleaned up"
            log ""
            log_info "Retrying in ${REFRESH_INTERVAL}s... (Ctrl+C to exit)"
            sleep "$REFRESH_INTERVAL"

            # Check if state file still doesn't exist after retry
            if ! check_state_file_exists; then
                log_info "State file no longer exists - training completed"
                exit 0
            fi
            continue
        fi

        # Parse status
        local status=$(parse_state "$state_json" "status")

        # Display state
        display_header
        display_state "$state_json"
        display_footer "$status"

        # Detect status change
        if [[ -n "$previous_status" ]] && [[ "$previous_status" != "$status" ]]; then
            log ""
            log_info "Status changed: ${previous_status} → ${status}"
        fi
        previous_status="$status"

        # Check for terminal states
        if [[ "$status" == "failed" ]]; then
            log ""
            log_error "Training failed. Check logs for details:"
            local run_id=$(parse_state "$state_json" "run_id")
            log "  aws s3 ls s3://${BUCKET_NAME}/${ENVIRONMENT}/logs/${run_id}/ --profile ${AWS_PROFILE}"
            sleep 5
            exit 1
        fi

        # Check if state file was deleted (training completed)
        if [[ "$status" == "cleanup" ]] && [[ $loop_count -gt 2 ]]; then
            sleep 5
            if ! check_state_file_exists; then
                log ""
                log_success "Training completed and cleaned up successfully!"
                exit 0
            fi
        fi

        # Sleep before next poll
        sleep "$REFRESH_INTERVAL"
        loop_count=$((loop_count + 1))
    done
}

# ============================================================================
# Signal Handlers
# ============================================================================

cleanup_on_exit() {
    log ""
    log_info "Monitoring stopped by user"
    exit 0
}

trap cleanup_on_exit SIGINT SIGTERM

# ============================================================================
# Entry Point
# ============================================================================

main

