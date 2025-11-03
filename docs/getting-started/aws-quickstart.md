# Quick Start: AWS EC2 Training in 5 Steps

Get your first Chronos training run on AWS EC2 in under 30 minutes.

## Prerequisites Checklist

Before you begin, ensure you have:

- [ ] AWS account with CLI configured (`aws configure`)
- [ ] Node.js 18+ installed (`node --version`)
- [ ] AWS CDK CLI installed (`npm install -g aws-cdk`)
- [ ] S3 bucket created for training artifacts
- [ ] Bash shell (Linux/macOS/WSL)

## Step 1: Deploy Infrastructure (5 minutes)

```bash
cd chronos-foundry/aws/cdk

# Create .env file
cat > .env <<EOF
S3_BUCKET_NAME=your-bucket-name
CDK_ENVIRONMENT=dev
PROJECT_NAME=Chronos-Training
COST_CENTER=train
EOF

# Install dependencies
npm install

# Bootstrap CDK (first time only)
cdk bootstrap

# Deploy
cdk deploy
```

**What this creates:**
- VPC with public subnet
- S3 Gateway Endpoint (zero cost)
- IAM role for EC2
- Security group

**Cost**: $0/month when idle

## Step 2: Build Python Environment (10 minutes)

```bash
cd chronos-foundry

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Package venv for EC2
tar -czf venv.tar.gz .venv/

# Upload to S3
export BUCKET=your-bucket-name
aws s3 cp venv.tar.gz s3://$BUCKET/runtime/python-env/chronos-venv-3.11.13.tar.gz
```

## Step 3: Upload Training Data (5 minutes)

```bash
# Prepare sample dataset (or use your own Parquet files)
# Organize by date: YYYY/MM/ structure
mkdir -p 2020/01
# ... add your .parquet files to 2020/01/ ...

# Upload to S3 (date-based structure)
aws s3 sync ./2020/ s3://$BUCKET/cached-datasets/2020/

# Upload scripts
cd aws/scripts
aws s3 sync . s3://$BUCKET/runtime/scripts/ --exclude "*.sh"
aws s3 sync . s3://$BUCKET/runtime/scripts/ --include "*.sh" --include "*.py"
```

## Step 4: Create Training Config (2 minutes)

```bash
cat > config.yaml <<EOF
prediction_length: 64
context_length: 512
model_id: "amazon/chronos-t5-small"
num_samples: 10000
batch_size: 32
learning_rate: 0.001
epochs: 10
EOF

# Upload config
aws s3 cp config.yaml s3://$BUCKET/dev/config.yaml
```

## Step 5: Launch Training (2 minutes + training time)

```bash
cd chronos-foundry/aws/scripts

# Set environment
export BUCKET=your-bucket-name
export ENVIRONMENT=dev

# Launch (dry-run first)
./launch_training.sh --dry-run

# Launch for real
./launch_training.sh
```

**Monitor progress:**
```bash
# Watch state file
watch -n 10 "aws s3 cp s3://$BUCKET/dev/system-state.json - | jq ."

# Or use monitor script
./monitor_training.sh
```

## Expected Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| EC2 Launch | 2-3 min | Instance startup |
| System Setup | 3-5 min | APT packages, NVIDIA driver |
| Venv Download | 1-2 min | Depends on venv size |
| Data Sync | 1-5 min | Depends on dataset size |
| Training | 30-120 min | Depends on data and model |
| Results Upload | 1-2 min | Model artifacts to S3 |
| Cleanup | 1 min | Self-termination |

**Total**: ~45-150 minutes for first run

## Cost for First Run

| Component | Cost |
|-----------|------|
| g4dn.2xlarge (2 hours) | ~$1.50 |
| EBS 100GB (2 hours) | ~$0.02 |
| S3 Storage (100MB) | ~$0.002 |
| **Total** | **~$1.52** |

## Verify Success

After training completes, check results:

```bash
# List outputs
RUN_ID=$(aws s3 ls s3://$BUCKET/dev/ | grep -oP '\d{4}-\d{2}-\d{2}-\d{6}' | tail -1)
aws s3 ls s3://$BUCKET/dev/$RUN_ID/ --recursive

# Download training metrics
aws s3 cp s3://$BUCKET/dev/$RUN_ID/training.json -
```

Expected output:
```json
{
  "run_id": "2025-01-15-143022",
  "start_time": "2025-01-15T14:30:22Z",
  "end_time": "2025-01-15T16:15:30Z",
  "duration_seconds": 6308,
  "model_files": ["model.pkl"],
  "model_count": 1
}
```

## Troubleshooting

### CDK Deploy Fails

**Error**: "S3 bucket not found"
```bash
# Verify bucket exists
aws s3 ls s3://your-bucket-name

# Or create it
aws s3 mb s3://your-bucket-name
```

### Launch Script Fails

**Error**: "CloudFormation exports not found"
```bash
# Verify CDK deployment
cdk list
cdk diff

# Re-deploy if needed
cd aws/cdk && cdk deploy
```

### Training Stuck at "data_sync"

**Error**: State file shows `"current_step": "data_sync"` for > 10 minutes
```bash
# Check dataset size (date-based structure)
aws s3 ls s3://$BUCKET/cached-datasets/2020/ --recursive --summarize

# Check instance console output
INSTANCE_ID=$(aws s3 cp s3://$BUCKET/dev/system-state.json - | jq -r .instance_id)
aws ec2 get-console-output --instance-id $INSTANCE_ID
```

### No Results After 2+ Hours

**Check state file:**
```bash
aws s3 cp s3://$BUCKET/dev/system-state.json - | jq .
```

**Possible states:**
- `"status": "failed"` - Check error_message
- `"status": "training"` - Still running (be patient)
- State file missing - Instance may have terminated prematurely

## Next Steps

Now that you've completed your first training run:

1. Read [System Architecture](../architecture/system-architecture.md) to understand system design
2. Customize `bootstrap.sh` for your domain-specific needs
3. Set up monitoring and alerting (CloudWatch, SNS)
4. Integrate with your CI/CD pipeline

## Clean Up (Optional)

To tear down infrastructure:

```bash
cd chronos-foundry/aws/cdk

# Destroy CDK stack
cdk destroy

# Delete S3 data (careful!)
aws s3 rm s3://$BUCKET/dev/ --recursive
aws s3 rm s3://$BUCKET/cached-datasets/ --recursive
```

**Warning**: This deletes all training results and cached datasets. Back up important data first.

## Questions?

- Check the [AWS README](../../aws/README.md) for overview
- Review other docs in this directory for detailed guides
- Open an issue on GitHub for bugs or feature requests


