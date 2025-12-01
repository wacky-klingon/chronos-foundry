# AWS EC2 Training Infrastructure for Chronos

This directory contains a **production-ready AWS reference implementation** for training Chronos models on EC2 GPU instances with zero idle cost.

## Overview

This infrastructure enables:
- **Ephemeral GPU Training**: Launch EC2 instances, train models, and auto-terminate
- **Zero Idle Cost**: No resources left running between training runs
- **State Management**: S3-based state tracking for monitoring and debugging
- **Cost Optimization**: Public subnet with S3 Gateway Endpoint (no NAT Gateway cost)
- **Security**: IAM-based access control, minimal egress rules

## Architecture

```
User Workstation                 AWS Cloud
┌──────────────┐                 ┌────────────────────────────────────┐
│              │                 │  VPC (Public Subnet)               │
│ launch_      │  AWS CLI        │  ┌──────────────────────────┐     │
│ training.sh ─┼────────────────▶│  │ EC2 GPU Instance         │     │
│              │                 │  │ (g4dn.2xlarge)           │     │
└──────────────┘                 │  │                          │     │
                                 │  │ 1. bootstrap.sh          │     │
                                 │  │ 2. preflight_check.py    │     │
                                 │  │ 3. training_wrapper.py   │     │
                                 │  │ 4. Self-terminates       │     │
                                 │  └────────────┬─────────────┘     │
                                 │               │                   │
                                 │               │ S3 Gateway        │
                                 │               │ Endpoint (Free)   │
                                 │               ▼                   │
                                 │  ┌────────────────────────┐       │
                                 │  │ S3 Bucket              │       │
                                 │  │                        │       │
                                 │  │ cached-datasets/       │       │
                                 │  │ {env}/                 │       │
                                 │  │   ├─ {run_id}/         │       │
                                 │  │   └─ system-state.json │       │
                                 │  │ logs/                  │       │
                                 │  └────────────────────────┘       │
                                 └────────────────────────────────────┘
```

## Directory Structure

```
aws/
├── cdk/                          # Infrastructure as Code
│   ├── lib/
│   │   └── chronos-training-stack.ts  # VPC, IAM, Security Group
│   ├── bin/
│   │   └── chronos-training.ts        # CDK app entry point
│   ├── package.json
│   └── .env.example
│
├── scripts/                      # EC2 orchestration and bootstrap
│   ├── lib/
│   │   └── state_helpers.sh          # Atomic S3 state management
│   ├── bootstrap.sh                  # EC2 user data script
│   ├── preflight_check.py            # GPU validation
│   ├── training_wrapper.py           # Training execution
│   ├── test_trainer.py               # Infrastructure testing
│   ├── launch_training.sh            # [TODO] Launch EC2 instance
│   ├── monitor_training.sh           # [TODO] Monitor state file
│   ├── kill_training.sh              # [TODO] Emergency stop
│   └── cleanup.sh                    # [TODO] Manual cleanup
│
└── README.md                     # This file

Note: Documentation has been moved to ../docs/aws/ directory
```

## Quick Start

### Prerequisites

1. **AWS CLI configured** with credentials and region
2. **Node.js 18+** and **npm** installed (for CDK)
3. **AWS CDK CLI** installed: `npm install -g aws-cdk`
4. **Bash** environment (Linux/macOS/WSL)

### 1. Deploy Infrastructure (One-Time)

```bash
cd aws/cdk

# Copy environment template
cp .env.example .env

# Edit .env with your bucket name
nano .env

# Install dependencies
npm install

# Bootstrap CDK (first time only)
cdk bootstrap

# Deploy the stack
cdk deploy
```

This creates:
- VPC with public subnet
- S3 Gateway Endpoint (free)
- IAM role for EC2 instances
- Security group (HTTPS egress only)

### 2. Prepare Training Data

Upload your training data to S3:

```bash
# Upload training dataset (Parquet files)
aws s3 sync ./data/ s3://YOUR-BUCKET/cached-datasets/training-data/

# Upload scripts and Python environment
aws s3 sync ./aws/scripts/ s3://YOUR-BUCKET/cached-datasets/scripts/
aws s3 cp ./venv.tar.gz s3://YOUR-BUCKET/cached-datasets/python-env/chronos-venv-3.11.13.tar.gz
```

### 2a. Upload EC2 Config Files (Admin Task)

**Note**: This requires admin credentials. The `trainer-runtime` user has read-only S3 access.

```bash
# Use admin credentials
export AWS_PROFILE=admin
export BUCKET_NAME=YOUR-BUCKET-NAME  # or get from CloudFormation exports

# Navigate to your project directory
cd your-project

# Upload EC2 config files (adjust file names to match your project)
aws s3 cp config/parquet_loader_config.ec2.yaml \
    s3://${BUCKET_NAME}/cached-datasets/configs/parquet_loader_config.yaml \
    --profile admin

aws s3 cp config/train.ec2.yaml \
    s3://${BUCKET_NAME}/cached-datasets/configs/train.yaml \
    --profile admin

# Upload other configs if they exist
aws s3 cp config/covariate_config.yaml \
    s3://${BUCKET_NAME}/cached-datasets/configs/covariate_config.yaml \
    --profile admin || true

aws s3 cp config/incremental_training_config.yaml \
    s3://${BUCKET_NAME}/cached-datasets/configs/incremental_training_config.yaml \
    --profile admin || true
```

### 3. Launch Training

```bash
cd aws/scripts

# Set environment
export BUCKET=YOUR-BUCKET-NAME
export ENVIRONMENT=dev

# Launch training
./launch_training.sh
```

### 4. Monitor Training

```bash
# Watch state file
./monitor_training.sh

# Or manually check
aws s3 cp s3://YOUR-BUCKET/dev/system-state.json -
```

### 5. Results

After training completes, find results at:
```
s3://YOUR-BUCKET/dev/{run_id}/
  ├── models/           # Trained model artifacts
  ├── training.json     # Metrics
  └── logs/             # Execution logs
```

## Cost Breakdown

### Infrastructure (Always Zero When Idle)
- VPC: $0
- S3 Gateway Endpoint: $0
- IAM Role: $0
- Security Group: $0

**Total Idle Cost: $0/month**

### Training Costs (Pay Per Use)
| Component | Cost | Notes |
|-----------|------|-------|
| g4dn.2xlarge (1 GPU) | ~$0.75/hr | NVIDIA T4 GPU |
| EBS 100GB | ~$0.01/hr | Ephemeral, deleted after run |
| S3 Storage | $0.023/GB/month | Models, logs, datasets |
| S3 Requests | Minimal | PUT/GET during training |

**Typical Training Run**: 1-2 hours = **~$1.50-$3.00**

## Configuration

### Environment Variables

Set these before running `launch_training.sh`:

```bash
export BUCKET=YOUR-BUCKET-NAME              # Required: S3 bucket name
export ENVIRONMENT=dev                       # Optional: dev/stage/prod (default: dev)
export INSTANCE_TYPE=g4dn.2xlarge           # Optional: EC2 instance type
export AWS_REGION=us-east-1                 # Optional: AWS region
export AWS_PROFILE=default                  # Optional: AWS CLI profile
```

### CDK Environment Variables

Set these in `cdk/.env`:

```bash
S3_BUCKET_NAME=your-bucket-name
CDK_ENVIRONMENT=dev
PROJECT_NAME=Chronos-Training
COST_CENTER=train
```

## State Machine

Training runs follow a defined state machine:

```
initialization → data_sync → training → results_sync → cleanup
         ↓           ↓           ↓            ↓           ↓
       failed      failed      failed       failed    (always succeeds)
                                 ↓
                               killed
```

State transitions are tracked in `s3://{bucket}/{env}/system-state.json`.

## Security

- **IAM Role**: EC2 instances use instance profile with minimal permissions
  - Read-only access to `cached-datasets/`
  - Read-write access to `{env}/` prefix only
  - Self-termination permission (tagged resources only)
- **Security Group**: HTTPS egress only (port 443)
- **No SSH**: Instance is not accessible via SSH
- **No Ingress**: No inbound traffic allowed

## Troubleshooting

### Training Stuck

```bash
# Check state file
aws s3 cp s3://YOUR-BUCKET/dev/system-state.json - | jq .

# Check instance logs (if instance still running)
INSTANCE_ID=$(jq -r .instance_id < state.json)
aws ec2 get-console-output --instance-id $INSTANCE_ID
```

### Emergency Stop

```bash
./kill_training.sh
```

### Manual Cleanup

```bash
./cleanup.sh
```

## Extending

This is a **reference implementation**. To adapt for your domain:

1. **Fork this directory** to your project
2. **Customize `bootstrap.sh`** with domain-specific setup
3. **Update `training_wrapper.py`** to call your training CLI
4. **Modify `chronos-training-stack.ts`** for custom IAM permissions
5. **Keep sensitive configs private** (see `PRIVATE_FILES.md` in your fork)

## Development

See `../docs/` for detailed guides:
- [AWS Documentation Index](../docs/aws/index.md) - Complete AWS reference
- [AWS Quickstart](../docs/getting-started/aws-quickstart.md) - Get started in 5 minutes
- [System Architecture](../docs/architecture/system-architecture.md) - Design principles
- [CDK Implementation](../docs/architecture/cdk-implementation.md) - Infrastructure as code

## License

This AWS reference implementation is part of the `chronos-foundry` library and is distributed under the same license. See the root `LICENSE` file for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Test with actual EC2 training runs
4. Submit a pull request with clear description

## Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Check the [AWS Documentation](../docs/aws/index.md) for detailed guides
- Review troubleshooting in [System Architecture](../docs/architecture/system-architecture.md#failure-modes)

## Changelog

### v1.0.0 (Initial Release)
- Core CDK infrastructure (VPC, IAM, Security Group)
- EC2 bootstrap and orchestration scripts
- State management with S3
- GPU preflight checks
- Training wrapper with metrics collection

