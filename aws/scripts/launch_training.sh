#!/bin/bash
# Launch Training Script
# Starts a new training run on EC2

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load state helpers
source "${SCRIPT_DIR}/lib/state_helpers.sh"

# ============================================================================
# CONFIGURATION
# ============================================================================

export ENVIRONMENT="${ENVIRONMENT:-dev}"
export AWS_PROFILE="${AWS_PROFILE:-trainer-runtime}"
export INSTANCE_TYPE="${INSTANCE_TYPE:-g4dn.2xlarge}"
export DATASET="${DATASET:-2010-01}"

# CloudFormation stack name pattern
STACK_NAME="HeisenbergTraining-${ENVIRONMENT}-Stack"

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

preflight_checks() {
    log_msg INFO "Starting pre-flight checks..."

    # Check dependencies
    if ! check_dependencies; then
        exit 1
    fi

    # Check AWS credentials
    if ! check_aws_credentials; then
        exit 1
    fi

    # Check for concurrent run
    if ! check_concurrent_run; then
        exit 1
    fi

    log_msg SUCCESS "Pre-flight checks passed"
}

# ============================================================================
# CLOUDFORMATION EXPORTS
# ============================================================================

get_cloudformation_exports() {
    log_msg INFO "Reading CloudFormation exports from stack: $STACK_NAME"

    # Get VPC ID
    VPC_ID=$(aws cloudformation list-exports \
        --query "Exports[?Name=='HeisenbergTraining-${ENVIRONMENT}-VPCId'].Value" \
        --output text \
        --profile "$AWS_PROFILE")

    # Get Subnet ID
    SUBNET_ID=$(aws cloudformation list-exports \
        --query "Exports[?Name=='HeisenbergTraining-${ENVIRONMENT}-PublicSubnetId'].Value" \
        --output text \
        --profile "$AWS_PROFILE")

    # Get Security Group ID
    SECURITY_GROUP_ID=$(aws cloudformation list-exports \
        --query "Exports[?Name=='HeisenbergTraining-${ENVIRONMENT}-SecurityGroupId'].Value" \
        --output text \
        --profile "$AWS_PROFILE")

    # Get IAM Instance Profile Name
    INSTANCE_PROFILE=$(aws cloudformation list-exports \
        --query "Exports[?Name=='HeisenbergTraining-${ENVIRONMENT}-InstanceProfileName'].Value" \
        --output text \
        --profile "$AWS_PROFILE")

    # Get S3 Bucket Name
    S3_BUCKET=$(aws cloudformation list-exports \
        --query "Exports[?Name=='HeisenbergTraining-${ENVIRONMENT}-S3BucketName'].Value" \
        --output text \
        --profile "$AWS_PROFILE")

    # Validate exports
    if [ -z "$VPC_ID" ] || [ -z "$SUBNET_ID" ] || [ -z "$SECURITY_GROUP_ID" ] || [ -z "$INSTANCE_PROFILE" ]; then
        log_msg ERROR "Failed to retrieve CloudFormation exports. Is the stack deployed?"
        echo ""
        echo "Expected stack: $STACK_NAME"
        echo "Run: cd aws/cdk && cdk deploy --profile trainer-infra"
        exit 1
    fi

    log_msg SUCCESS "CloudFormation exports retrieved"
    echo "  VPC ID: $VPC_ID"
    echo "  Subnet ID: $SUBNET_ID"
    echo "  Security Group: $SECURITY_GROUP_ID"
    echo "  Instance Profile: $INSTANCE_PROFILE"
    echo "  S3 Bucket: $S3_BUCKET"
}

# ============================================================================
# AMI SELECTION
# ============================================================================

get_latest_ami() {
    log_msg INFO "Finding latest Amazon Linux 2023 GPU AMI..."

    # Get latest AL2023 AMI with NVIDIA drivers
    # For MVP: Use standard AL2023, we'll install NVIDIA drivers in user data
    AMI_ID=$(aws ec2 describe-images \
        --owners amazon \
        --filters \
          "Name=name,Values=al2023-ami-2023.*-x86_64" \
          "Name=state,Values=available" \
        --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
        --output text \
        --profile "$AWS_PROFILE")

    if [ -z "$AMI_ID" ] || [ "$AMI_ID" == "None" ]; then
        log_msg ERROR "Failed to find suitable AMI"
        exit 1
    fi

    log_msg SUCCESS "AMI selected: $AMI_ID"
}

# ============================================================================
# USER DATA SCRIPT
# ============================================================================

generate_user_data() {
    local run_id="$1"

    log_msg INFO "Generating user data script..."

    cat > /tmp/user-data-${run_id}.sh <<'EOF'
#!/bin/bash
# EC2 User Data - Minimal Bootstrap Launcher
set -x
exec > >(tee /var/log/user-data.log) 2>&1

echo "Starting user data script at $(date)"

# Export environment variables (from launch script)
export RUN_ID="__RUN_ID__"
export ENVIRONMENT="__ENVIRONMENT__"
export BUCKET_NAME="__BUCKET__"
export DATASET="__DATASET__"
export AWS_DEFAULT_REGION="us-east-1"

# Get AWS region from metadata
export AWS_REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region || echo "us-east-1")

# Get instance ID from metadata
export INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)

echo "Run ID: ${RUN_ID}"
echo "Environment: ${ENVIRONMENT}"
echo "Bucket: ${BUCKET_NAME}"
echo "Instance: ${INSTANCE_ID}"
echo "Region: ${AWS_REGION}"

# Install minimal dependencies needed to download scripts
yum update -y -q
yum install -y -q aws-cli curl tar gzip

# Download bootstrap script and dependencies from S3
echo "Downloading bootstrap scripts from S3..."
BOOTSTRAP_DIR="/opt/scripts"
mkdir -p "${BOOTSTRAP_DIR}/lib"

# Download bootstrap.sh
aws s3 cp "s3://${BUCKET_NAME}/cached-datasets/scripts/bootstrap.sh" \
    "${BOOTSTRAP_DIR}/bootstrap.sh" || {
    echo "ERROR: Failed to download bootstrap.sh from S3"
    exit 1
}

# Download state_helpers.sh
aws s3 cp "s3://${BUCKET_NAME}/cached-datasets/scripts/lib/state_helpers.sh" \
    "${BOOTSTRAP_DIR}/lib/state_helpers.sh" || {
    echo "ERROR: Failed to download state_helpers.sh from S3"
    exit 1
}

# Download training_wrapper.py
aws s3 cp "s3://${BUCKET_NAME}/cached-datasets/scripts/training_wrapper.py" \
    "${BOOTSTRAP_DIR}/training_wrapper.py" || {
    echo "ERROR: Failed to download training_wrapper.py from S3"
    exit 1
}

# Download preflight_check.py
aws s3 cp "s3://${BUCKET_NAME}/cached-datasets/scripts/preflight_check.py" \
    "${BOOTSTRAP_DIR}/preflight_check.py" || {
    echo "ERROR: Failed to download preflight_check.py from S3"
    exit 1
}

# Make scripts executable
chmod +x "${BOOTSTRAP_DIR}/bootstrap.sh"
chmod +x "${BOOTSTRAP_DIR}/training_wrapper.py"
chmod +x "${BOOTSTRAP_DIR}/preflight_check.py"

# Set state file path for bootstrap
export STATE_FILE="s3://${BUCKET_NAME}/${ENVIRONMENT}/system-state.json"

# Execute bootstrap script
echo "Executing bootstrap.sh..."
cd "${BOOTSTRAP_DIR}"
exec bash bootstrap.sh
EOF

    # Replace placeholders
    sed -i "s/__RUN_ID__/$run_id/g" /tmp/user-data-${run_id}.sh
    sed -i "s/__ENVIRONMENT__/$ENVIRONMENT/g" /tmp/user-data-${run_id}.sh
    sed -i "s/__BUCKET__/$S3_BUCKET/g" /tmp/user-data-${run_id}.sh
    sed -i "s/__DATASET__/$DATASET/g" /tmp/user-data-${run_id}.sh

    log_msg SUCCESS "User data generated"
}

# ============================================================================
# EC2 LAUNCH
# ============================================================================

launch_ec2_instance() {
    local run_id="$1"

    log_msg INFO "Launching EC2 instance..."

    # Launch instance
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --subnet-id "$SUBNET_ID" \
        --security-group-ids "$SECURITY_GROUP_ID" \
        --iam-instance-profile "Name=$INSTANCE_PROFILE" \
        --user-data "file:///tmp/user-data-${run_id}.sh" \
        --tag-specifications \
          "ResourceType=instance,Tags=[
            {Key=Name,Value=HeisenbergTraining-${ENVIRONMENT}-${run_id}},
            {Key=Project,Value=Heisenberg-Engine},
            {Key=Environment,Value=${ENVIRONMENT}},
            {Key=CostCenter,Value=train},
            {Key=ManagedBy,Value=launch-script},
            {Key=RunID,Value=${run_id}}
          ]" \
          "ResourceType=volume,Tags=[
            {Key=Name,Value=HeisenbergTraining-${ENVIRONMENT}-${run_id}-volume},
            {Key=Project,Value=Heisenberg-Engine},
            {Key=Environment,Value=${ENVIRONMENT}},
            {Key=DeleteOnTermination,Value=true}
          ]" \
        --block-device-mappings "[
          {
            \"DeviceName\": \"/dev/xvda\",
            \"Ebs\": {
              \"VolumeSize\": 100,
              \"VolumeType\": \"gp3\",
              \"DeleteOnTermination\": true,
              \"Encrypted\": true
            }
          }
        ]" \
        --query 'Instances[0].InstanceId' \
        --output text \
        --profile "$AWS_PROFILE")

    if [ -z "$INSTANCE_ID" ]; then
        log_msg ERROR "Failed to launch EC2 instance"
        exit 1
    fi

    log_msg SUCCESS "EC2 instance launched: $INSTANCE_ID"

    # Wait for instance to enter running state
    log_msg INFO "Waiting for instance to reach running state..."
    aws ec2 wait instance-running \
        --instance-ids "$INSTANCE_ID" \
        --profile "$AWS_PROFILE"

    log_msg SUCCESS "Instance is running"

    # Clean up user data file
    rm -f /tmp/user-data-${run_id}.sh
}

# ============================================================================
# STATE FILE INITIALIZATION
# ============================================================================

initialize_state_file() {
    local run_id="$1"
    local instance_id="$2"

    log_msg INFO "Initializing state file..."

    init_state_file "$run_id" "$instance_id" "$INSTANCE_TYPE"

    # Add training config
    atomic_write_state \
        "training_config.dataset=$DATASET" \
        "training_config.git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

    log_msg SUCCESS "State file initialized: $STATE_FILE"
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    echo ""
    echo "======================================================================"
    echo "  Heisenberg Engine Training - Launch Script"
    echo "======================================================================"
    echo ""
    echo "Environment: $ENVIRONMENT"
    echo "Instance Type: $INSTANCE_TYPE"
    echo "Dataset: $DATASET"
    echo ""

    # Generate run ID
    RUN_ID=$(generate_run_id)
    export RUN_ID

    log_msg INFO "Run ID: $RUN_ID"
    echo ""

    # Pre-flight checks
    preflight_checks
    echo ""

    # Get CloudFormation exports
    get_cloudformation_exports
    echo ""

    # Get AMI
    get_latest_ami
    echo ""

    # Generate user data
    generate_user_data "$RUN_ID"
    echo ""

    # Launch EC2 instance
    launch_ec2_instance "$RUN_ID"
    echo ""

    # Initialize state file
    initialize_state_file "$RUN_ID" "$INSTANCE_ID"
    echo ""

    # Success message
    echo "======================================================================"
    log_msg SUCCESS "Training run launched successfully!"
    echo "======================================================================"
    echo ""
    echo "Run ID: $RUN_ID"
    echo "Instance ID: $INSTANCE_ID"
    echo "Instance Type: $INSTANCE_TYPE"
    echo "State File: $STATE_FILE"
    echo ""
    echo "Next steps:"
    echo "  1. Monitor progress:"
    echo "     ./monitor_training.sh"
    echo ""
    echo "  2. View logs in S3:"
    echo "     aws s3 ls s3://$S3_BUCKET/$ENVIRONMENT/logs/$RUN_ID/ --profile $AWS_PROFILE"
    echo ""
    echo "  3. Emergency stop:"
    echo "     ./kill_training.sh"
    echo ""
    echo "Estimated cost: ~\$2-4 for this run"
    echo ""
}

# Run main function
main "$@"

