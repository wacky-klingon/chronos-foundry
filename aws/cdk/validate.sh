#!/bin/bash
# AWS Infrastructure Validation Script
# Usage: ./validate.sh [pre|post]
#   pre  - Check state before teardown
#   post - Verify clean state after teardown

set -e

PROFILE="trainer-infra"
PROJECT_TAG="Heisenberg-Engine"
STACK_NAME="HeisenbergTraining-dev-Stack"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

MODE="${1:-pre}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AWS Infrastructure Validation${NC}"
echo -e "${BLUE}Mode: ${MODE}-teardown${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# ============================================================================
# PRE-TEARDOWN CHECKS
# ============================================================================
if [ "$MODE" = "pre" ]; then
    echo -e "${YELLOW}[PRE-TEARDOWN] Checking for blockers...${NC}"
    echo ""

    # 1. Check for running EC2 instances
    echo "1. Checking EC2 instances..."
    INSTANCES=$(aws ec2 describe-instances \
        --filters "Name=tag:Project,Values=${PROJECT_TAG}" "Name=instance-state-name,Values=running,stopped,stopping" \
        --query 'Reservations[*].Instances[*].[InstanceId,State.Name,Tags[?Key==`Name`].Value|[0]]' \
        --output text \
        --profile "$PROFILE" 2>/dev/null || echo "")

    if [ -z "$INSTANCES" ]; then
        echo -e "   ${GREEN}✓ No EC2 instances found${NC}"
    else
        echo -e "   ${RED}✗ WARNING: Found running/stopped instances:${NC}"
        echo "$INSTANCES" | while read line; do
            echo -e "   ${RED}  - $line${NC}"
        done
        echo -e "   ${RED}  Action: Terminate these instances before teardown${NC}"
    fi
    echo ""

    # 2. Check CloudFormation stack
    echo "2. Checking CloudFormation stack..."
    STACK_STATUS=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --query 'Stacks[0].StackStatus' \
        --output text \
        --profile "$PROFILE" 2>/dev/null || echo "NOT_FOUND")

    if [ "$STACK_STATUS" = "NOT_FOUND" ]; then
        echo -e "   ${YELLOW}⚠ Stack not found (already deleted or never created)${NC}"
    else
        echo -e "   ${GREEN}✓ Stack exists: ${STACK_STATUS}${NC}"
    fi
    echo ""

    # 3. Check VPCs
    echo "3. Checking VPCs..."
    VPCS=$(aws ec2 describe-vpcs \
        --filters "Name=tag:Project,Values=${PROJECT_TAG}" \
        --query 'Vpcs[*].[VpcId,Tags[?Key==`Name`].Value|[0]]' \
        --output text \
        --profile "$PROFILE" 2>/dev/null || echo "")

    if [ -z "$VPCS" ]; then
        echo -e "   ${YELLOW}⚠ No VPCs found${NC}"
    else
        echo -e "   ${GREEN}✓ Found VPCs:${NC}"
        echo "$VPCS" | while read line; do
            echo "     - $line"
        done
    fi
    echo ""

    # 4. Check Security Groups
    echo "4. Checking Security Groups..."
    SGS=$(aws ec2 describe-security-groups \
        --filters "Name=tag:Project,Values=${PROJECT_TAG}" \
        --query 'SecurityGroups[*].[GroupId,GroupName]' \
        --output text \
        --profile "$PROFILE" 2>/dev/null || echo "")

    if [ -z "$SGS" ]; then
        echo -e "   ${YELLOW}⚠ No security groups found${NC}"
    else
        echo -e "   ${GREEN}✓ Found security groups:${NC}"
        echo "$SGS" | while read line; do
            echo "     - $line"
        done
    fi
    echo ""

    # 5. Check IAM Roles
    echo "5. Checking IAM Roles..."
    ROLES=$(aws iam list-roles \
        --query "Roles[?contains(RoleName, 'HeisenbergTraining')].RoleName" \
        --output text \
        --profile "$PROFILE" 2>/dev/null || echo "")

    if [ -z "$ROLES" ]; then
        echo -e "   ${YELLOW}⚠ No IAM roles found (may take 30s to propagate)${NC}"
    else
        echo -e "   ${GREEN}✓ Found IAM roles:${NC}"
        echo "$ROLES" | tr '\t' '\n' | while read line; do
            [ -n "$line" ] && echo "     - $line"
        done
    fi
    echo ""

    # 6. Check EBS Volumes (orphaned)
    echo "6. Checking for orphaned EBS volumes..."
    VOLUMES=$(aws ec2 describe-volumes \
        --filters "Name=tag:Project,Values=${PROJECT_TAG}" "Name=status,Values=available" \
        --query 'Volumes[*].[VolumeId,Size,State]' \
        --output text \
        --profile "$PROFILE" 2>/dev/null || echo "")

    if [ -z "$VOLUMES" ]; then
        echo -e "   ${GREEN}✓ No orphaned volumes${NC}"
    else
        echo -e "   ${YELLOW}⚠ Found orphaned volumes:${NC}"
        echo "$VOLUMES" | while read line; do
            echo "     - $line GB"
        done
    fi
    echo ""

    # 7. Check S3 Bucket
    echo "7. Checking S3 bucket..."
    # Try list-buckets first (more permissive), fallback to head-bucket
    BUCKET_EXISTS=$(aws s3api list-buckets \
        --query "Buckets[?Name=='the-heisenberg-engine-tachyon-core'].Name" \
        --output text \
        --profile "$PROFILE" 2>/dev/null || echo "")

    if [ -n "$BUCKET_EXISTS" ]; then
        echo -e "   ${GREEN}✓ S3 bucket exists (will NOT be deleted by CDK)${NC}"
    else
        # Fallback to direct check
        BUCKET_EXISTS=$(aws s3 ls s3://the-heisenberg-engine-tachyon-core \
            --profile "$PROFILE" 2>/dev/null && echo "EXISTS" || echo "")
        if [ -n "$BUCKET_EXISTS" ]; then
            echo -e "   ${GREEN}✓ S3 bucket exists (will NOT be deleted by CDK)${NC}"
        else
            echo -e "   ${YELLOW}⚠ S3 bucket check inconclusive (may be permissions issue)${NC}"
        fi
    fi
    echo ""

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Pre-Teardown Summary${NC}"
    echo -e "${BLUE}========================================${NC}"
    if [ -z "$INSTANCES" ]; then
        echo -e "${GREEN}✓ Safe to proceed with teardown${NC}"
        echo ""
        echo "Run: export \$(grep -v '^#' .env | xargs) && cdk destroy --profile trainer-infra"
    else
        echo -e "${RED}✗ Blockers detected - terminate EC2 instances first${NC}"
    fi
fi

# ============================================================================
# POST-TEARDOWN VERIFICATION
# ============================================================================
if [ "$MODE" = "post" ]; then
    echo -e "${YELLOW}[POST-TEARDOWN] Verifying clean state...${NC}"
    echo ""

    ERRORS=0

    # 1. Verify CloudFormation stack is gone
    echo "1. Verifying CloudFormation stack deletion..."
    STACK_STATUS=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --query 'Stacks[0].StackStatus' \
        --output text \
        --profile "$PROFILE" 2>/dev/null || echo "NOT_FOUND")

    if [ "$STACK_STATUS" = "NOT_FOUND" ] || [ "$STACK_STATUS" = "DELETE_COMPLETE" ]; then
        echo -e "   ${GREEN}✓ Stack deleted successfully${NC}"
    else
        echo -e "   ${RED}✗ Stack still exists: ${STACK_STATUS}${NC}"
        ((ERRORS++))
    fi
    echo ""

    # 2. Verify no EC2 instances
    echo "2. Verifying no EC2 instances..."
    INSTANCES=$(aws ec2 describe-instances \
        --filters "Name=tag:Project,Values=${PROJECT_TAG}" "Name=instance-state-name,Values=running,stopped,stopping,pending" \
        --query 'Reservations[*].Instances[*].InstanceId' \
        --output text \
        --profile "$PROFILE" 2>/dev/null || echo "")

    if [ -z "$INSTANCES" ]; then
        echo -e "   ${GREEN}✓ No EC2 instances found${NC}"
    else
        echo -e "   ${RED}✗ Found instances: ${INSTANCES}${NC}"
        ((ERRORS++))
    fi
    echo ""

    # 3. Verify VPCs deleted
    echo "3. Verifying VPCs deleted..."
    VPCS=$(aws ec2 describe-vpcs \
        --filters "Name=tag:Project,Values=${PROJECT_TAG}" \
        --query 'Vpcs[*].VpcId' \
        --output text \
        --profile "$PROFILE" 2>/dev/null || echo "")

    if [ -z "$VPCS" ]; then
        echo -e "   ${GREEN}✓ No VPCs found${NC}"
    else
        echo -e "   ${RED}✗ Found VPCs: ${VPCS}${NC}"
        ((ERRORS++))
    fi
    echo ""

    # 4. Verify Security Groups deleted
    echo "4. Verifying Security Groups deleted..."
    SGS=$(aws ec2 describe-security-groups \
        --filters "Name=tag:Project,Values=${PROJECT_TAG}" \
        --query 'SecurityGroups[*].GroupId' \
        --output text \
        --profile "$PROFILE" 2>/dev/null || echo "")

    if [ -z "$SGS" ]; then
        echo -e "   ${GREEN}✓ No security groups found${NC}"
    else
        echo -e "   ${RED}✗ Found security groups: ${SGS}${NC}"
        ((ERRORS++))
    fi
    echo ""

    # 5. Verify IAM Roles deleted
    echo "5. Verifying IAM Roles deleted..."
    ROLES=$(aws iam list-roles \
        --query "Roles[?contains(RoleName, 'HeisenbergTraining')].RoleName" \
        --output text \
        --profile "$PROFILE" 2>/dev/null || echo "")

    if [ -z "$ROLES" ]; then
        echo -e "   ${GREEN}✓ No IAM roles found${NC}"
    else
        echo -e "   ${YELLOW}⚠ Found IAM roles (IAM deletion can take 1-2 min):${NC}"
        echo "$ROLES" | tr '\t' '\n' | while read line; do
            [ -n "$line" ] && echo "     - $line"
        done
    fi
    echo ""

    # 6. Verify S3 bucket still exists (should NOT be deleted)
    echo "6. Verifying S3 bucket (should remain)..."
    BUCKET_EXISTS=$(aws s3api list-buckets \
        --query "Buckets[?Name=='the-heisenberg-engine-tachyon-core'].Name" \
        --output text \
        --profile "$PROFILE" 2>/dev/null || echo "")

    if [ -n "$BUCKET_EXISTS" ]; then
        echo -e "   ${GREEN}✓ S3 bucket preserved (as expected)${NC}"
    else
        # Fallback check
        BUCKET_EXISTS=$(aws s3 ls s3://the-heisenberg-engine-tachyon-core \
            --profile "$PROFILE" 2>/dev/null && echo "EXISTS" || echo "")
        if [ -n "$BUCKET_EXISTS" ]; then
            echo -e "   ${GREEN}✓ S3 bucket preserved (as expected)${NC}"
        else
            echo -e "   ${YELLOW}⚠ S3 bucket check inconclusive${NC}"
        fi
    fi
    echo ""

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Post-Teardown Summary${NC}"
    echo -e "${BLUE}========================================${NC}"
    if [ $ERRORS -eq 0 ]; then
        echo -e "${GREEN}✓ Teardown successful - all resources cleaned up${NC}"
        echo ""
        echo "Ready for fresh deployment:"
        echo "  cd aws/cdk"
        echo "  npm run build"
        echo "  export \$(grep -v '^#' .env | xargs) && cdk deploy --profile trainer-infra"
    else
        echo -e "${RED}✗ Teardown incomplete - $ERRORS issues detected${NC}"
        echo "Check AWS Console for manual cleanup"
    fi
fi

echo ""

