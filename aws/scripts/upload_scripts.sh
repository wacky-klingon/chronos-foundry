#!/bin/bash
#
# upload_scripts.sh - Upload bootstrap scripts to S3
#
# Uploads all necessary scripts to S3 so they can be downloaded by EC2 instances.
# This should be run once before deploying, or whenever scripts are updated.
#
# Usage:
#   ./upload_scripts.sh
#   ENVIRONMENT=stage BUCKET_NAME=my-bucket ./upload_scripts.sh
#
# Environment Variables:
#   ENVIRONMENT     - dev, stage, or prod (default: dev)
#   BUCKET_NAME     - S3 bucket name (required)
#   AWS_PROFILE     - AWS CLI profile (default: trainer-runtime)

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVIRONMENT="${ENVIRONMENT:-dev}"
AWS_PROFILE="${AWS_PROFILE:-trainer-runtime}"

# Load environment variables from .env if available
if [[ -f "${SCRIPT_DIR}/../cdk/.env" ]]; then
    export $(grep -v '^#' "${SCRIPT_DIR}/../cdk/.env" | xargs)
fi

if [[ -z "${BUCKET_NAME:-}" ]]; then
    echo "ERROR: BUCKET_NAME not set. Set in .env or environment." >&2
    exit 1
fi

S3_SCRIPTS_BASE="s3://${BUCKET_NAME}/cached-datasets/scripts"

# ============================================================================
# Helper Functions
# ============================================================================

log_msg() {
    local level="$1"
    shift
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [$level] $*"
}

# ============================================================================
# Upload Functions
# ============================================================================

upload_script() {
    local local_path="$1"
    local s3_path="$2"
    
    if [[ ! -f "$local_path" ]]; then
        log_msg ERROR "Script not found: $local_path"
        return 1
    fi
    
    log_msg INFO "Uploading $local_path -> $s3_path"
    if aws s3 cp "$local_path" "$s3_path" --profile "$AWS_PROFILE"; then
        log_msg SUCCESS "Uploaded: $(basename "$local_path")"
        return 0
    else
        log_msg ERROR "Failed to upload: $local_path"
        return 1
    fi
}

# ============================================================================
# Main Upload Process
# ============================================================================

main() {
    echo ""
    echo "======================================================================"
    echo "  Upload Training Scripts to S3"
    echo "======================================================================"
    echo ""
    echo "Environment: $ENVIRONMENT"
    echo "Bucket: $BUCKET_NAME"
    echo "S3 Base Path: $S3_SCRIPTS_BASE"
    echo ""

    # Create S3 directory structure
    log_msg INFO "Ensuring S3 directory structure exists..."
    aws s3api put-object \
        --bucket "$BUCKET_NAME" \
        --key "cached-datasets/scripts/.keep" \
        --body /dev/null \
        --profile "$AWS_PROFILE" 2>/dev/null || true

    aws s3api put-object \
        --bucket "$BUCKET_NAME" \
        --key "cached-datasets/scripts/lib/.keep" \
        --body /dev/null \
        --profile "$AWS_PROFILE" 2>/dev/null || true

    # Upload main scripts
    echo ""
    log_msg INFO "Uploading main scripts..."

    upload_script "${SCRIPT_DIR}/bootstrap.sh" "${S3_SCRIPTS_BASE}/bootstrap.sh"
    upload_script "${SCRIPT_DIR}/training_wrapper.py" "${S3_SCRIPTS_BASE}/training_wrapper.py"
    upload_script "${SCRIPT_DIR}/preflight_check.py" "${S3_SCRIPTS_BASE}/preflight_check.py"
    upload_script "${SCRIPT_DIR}/cleanup.sh" "${S3_SCRIPTS_BASE}/cleanup.sh" || true  # Optional

    # Upload library files
    echo ""
    log_msg INFO "Uploading library files..."

    if [[ -f "${SCRIPT_DIR}/lib/state_helpers.sh" ]]; then
        upload_script "${SCRIPT_DIR}/lib/state_helpers.sh" "${S3_SCRIPTS_BASE}/lib/state_helpers.sh"
    else
        log_msg ERROR "state_helpers.sh not found at ${SCRIPT_DIR}/lib/state_helpers.sh"
        return 1
    fi

    echo ""
    echo "======================================================================"
    log_msg SUCCESS "Script upload complete!"
    echo "======================================================================"
    echo ""
    echo "Scripts are available at:"
    echo "  ${S3_SCRIPTS_BASE}/"
    echo ""
    echo "Next steps:"
    echo "  1. Verify scripts are accessible:"
    echo "     aws s3 ls ${S3_SCRIPTS_BASE}/ --profile $AWS_PROFILE"
    echo ""
    echo "  2. Launch training:"
    echo "     cd ${SCRIPT_DIR}"
    echo "     ./launch_training.sh"
    echo ""
}

# Run main function
main "$@"

