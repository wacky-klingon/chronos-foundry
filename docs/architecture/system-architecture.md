# AWS EC2 Training Architecture

This document describes the design principles, architecture, and implementation details of the Chronos EC2 training system.

## Design Principles

### 1. Zero Idle Cost

**Principle**: The system must incur **$0 in cost** when not actively training.

**Implementation**:
- No persistent compute (EC2 launches on-demand, auto-terminates)
- No NAT Gateway (uses public subnet + S3 Gateway Endpoint)
- No Load Balancer, no RDS, no persistent EFS
- All state stored in S3 (pay per GB, not per hour)

### 2. Ephemeral Compute

**Principle**: Every training run is self-contained, complete, and deleted afterward.

**Implementation**:
- EC2 instance launches, trains, uploads results, terminates
- EBS volume deleted on termination
- State tracked in S3, not on instance
- No SSH access required

### 3. Observable State

**Principle**: The system state must be queryable at any time without instance access.

**Implementation**:
- Atomic S3 state file: `s3://{bucket}/{env}/system-state.json`
- State transitions logged and validated
- Concurrent run prevention via state file existence
- Logs uploaded continuously to S3

### 4. Fail-Safe Cleanup

**Principle**: Resources must be cleaned up even on error.

**Implementation**:
- Bash `trap` handlers for ERR signal
- Best-effort cleanup (never fails)
- Instance self-terminates even on training failure
- State file deletion signals cleanup completion

## System Architecture

### High-Level Flow

```
┌─────────────────┐
│ User Workstation│
│                 │
│ launch_training │
│      .sh        │
└────────┬────────┘
         │ AWS CLI
         ▼
┌──────────────────────────────────────┐
│ AWS CloudFormation (CDK)             │
│                                      │
│ VPC + Subnet + IAM + Security Group  │
└──────────────────────────────────────┘
         │
         │ EC2 run-instances
         ▼
┌──────────────────────────────────────┐
│ EC2 GPU Instance (g4dn.2xlarge)      │
│                                      │
│ ┌────────────────────────────────┐  │
│ │ User Data (bootstrap.sh)       │  │
│ │                                │  │
│ │ 1. System setup                │  │
│ │ 2. Download venv from S3       │  │
│ │ 3. GPU preflight checks        │  │
│ │ 4. Sync training data          │  │
│ │ 5. Execute training            │  │
│ │ 6. Upload results              │  │
│ │ 7. Self-terminate              │  │
│ └────────────────────────────────┘  │
└──────────────┬───────────────────────┘
               │ S3 Gateway Endpoint (Free)
               ▼
┌──────────────────────────────────────┐
│ S3 Bucket                            │
│                                      │
│ cached-datasets/                     │
│   ├─ python-env/                     │
│   ├─ scripts/                        │
│   └─ training-data/                  │
│                                      │
│ {env}/                               │
│   ├─ system-state.json (atomic)      │
│   └─ {run_id}/                       │
│       ├─ models/                     │
│       ├─ training.json               │
│       └─ logs/                       │
└──────────────────────────────────────┘
```

### Components

#### 1. CDK Infrastructure (`cdk/`)

**Purpose**: Define AWS resources as code

**Resources**:
- **VPC**: Single AZ, public subnet only
- **S3 Gateway Endpoint**: Free connectivity to S3 from VPC
- **IAM Role**: Grants EC2 instances S3 and self-termination permissions
- **Security Group**: HTTPS egress only

**Lifecycle**: Persistent (deployed once, used many times)

**Cost**: $0/month

#### 2. Launch Script (`launch_training.sh`)

**Purpose**: Launch EC2 instance with proper configuration

**Responsibilities**:
- Pre-flight checks (AWS credentials, CloudFormation exports)
- Select latest Deep Learning AMI
- Generate user data (bootstrap.sh with environment variables)
- Launch EC2 instance with tags and IAM role
- Initialize S3 state file
- Prevent concurrent runs

**Lifecycle**: Runs on user workstation

#### 3. Bootstrap Script (`bootstrap.sh`)

**Purpose**: EC2 user data script that orchestrates training

**Execution Flow**:
```
initialization → system_setup → venv_download → gpu_preflight
                                                      ↓
cleanup ← results_sync ← training_execution ← data_sync
```

**State Transitions**:
- Each phase updates `system-state.json` atomically
- Errors trigger `status=failed` and cleanup
- Cleanup always runs (even on error)

**Lifecycle**: Runs once on EC2 instance launch

#### 4. Preflight Check (`preflight_check.py`)

**Purpose**: Validate GPU and environment before training

**Checks**:
- NVIDIA driver loaded (`nvidia-smi`)
- CUDA available (`torch.cuda.is_available()`)
- GPU count >= 1
- Python venv valid
- Disk space sufficient (>= 20GB)

**Exit**: 0 on success, 1 on failure (triggers cleanup)

#### 5. Training Wrapper (`training_wrapper.py`)

**Purpose**: Execute training with state management

**Responsibilities**:
- Update S3 state file during training
- Execute training CLI (or test trainer)
- Stream training output
- Collect metrics from logs
- Generate `training.json` artifact

**Lifecycle**: Runs on EC2 instance

#### 6. State Helpers (`state_helpers.sh`)

**Purpose**: Bash library for atomic S3 state operations

**Key Functions**:
- `atomic_write_state()`: Update state file with retries
- `check_concurrent_run()`: Prevent overlapping runs
- `transition_state()`: Validate state machine transitions
- `retry_with_backoff()`: Exponential backoff for AWS API calls

**Lifecycle**: Sourced by other scripts

## Data Flow

### Training Data Flow

```
Local → S3 (cached-datasets) → EC2 (/data) → Training → EC2 (/data/output) → S3 ({env}/{run_id})
```

**S3 Bucket Structure**:
```
s3://bucket/
├── cached-datasets/              # Immutable, long-lived
│   ├── python-env/
│   │   └── chronos-venv-3.11.13.tar.gz
│   ├── scripts/
│   │   ├── bootstrap.sh
│   │   ├── preflight_check.py
│   │   ├── training_wrapper.py
│   │   └── lib/
│   │       └── state_helpers.sh
│   └── training-data/            # Parquet files
│       ├── file1.parquet
│       └── file2.parquet
│
├── dev/                          # Ephemeral, per-run
│   ├── system-state.json         # Atomic state tracking
│   ├── 2025-01-15-143022/
│   │   ├── config.yaml
│   │   ├── models/
│   │   │   └── model.pkl
│   │   ├── training.json
│   │   └── logs/
│   │       └── bootstrap.log
│   └── 2025-01-16-091215/
│       └── ...
│
└── logs/                         # Global logs (optional)
    └── 2025-01-15-143022/
        └── phase_logs.json
```

### State File Format

**File**: `s3://{bucket}/{env}/system-state.json`

**Schema**:
```json
{
  "run_id": "2025-01-15-143022",
  "environment": "dev",
  "status": "training",
  "current_step": "epoch_5",
  "instance_id": "i-0123456789abcdef0",
  "instance_type": "g4dn.2xlarge",
  "s3_prefix": "s3://bucket/dev/2025-01-15-143022/",
  "timestamps": {
    "start": "2025-01-15T14:30:22Z",
    "last_update": "2025-01-15T15:42:10Z"
  },
  "training_config": {
    "prediction_length": 64,
    "context_length": 512
  },
  "error_message": null,
  "pid": null
}
```

**Status Values**: `initialization`, `data_sync`, `training`, `results_sync`, `cleanup`, `failed`, `killed`

**Atomicity**: Updates use S3 tmp file + copy (atomic on S3)

## State Machine

```
┌──────────────┐
│initialization│
└──────┬───────┘
       ├─────────────────┐
       ▼                 ▼
  ┌─────────┐       ┌────────┐
  │data_sync│       │ failed │
  └────┬────┘       └───┬────┘
       ├─────────────┐  │
       ▼             ▼  │
  ┌─────────┐  ┌────────┤
  │training │  │ failed │
  └────┬────┘  └───┬────┘
       ├──────────┐│
       ├─────────┐││
       ▼         ▼▼▼
  ┌────────────┐┌────────┐
  │results_sync││ killed │
  └──────┬─────┘└───┬────┘
         ├──────────┤
         ▼          ▼
     ┌─────────┐
     │ cleanup │ (always succeeds)
     └─────────┘
```

**Transition Rules**:
- `initialization` → `data_sync` | `failed`
- `data_sync` → `training` | `failed`
- `training` → `results_sync` | `failed` | `killed`
- `results_sync` → `cleanup` | `failed`
- `failed` | `killed` → `cleanup`
- `cleanup` → (terminal, state file deleted)

## Security Model

### IAM Permissions

**EC2 Instance Profile** (`ChronosTrainingInstanceProfile`):
```json
{
  "S3ReadAccess": {
    "Effect": "Allow",
    "Action": ["s3:GetObject", "s3:ListBucket"],
    "Resource": [
      "arn:aws:s3:::bucket",
      "arn:aws:s3:::bucket/cached-datasets/*"
    ]
  },
  "S3WriteAccess": {
    "Effect": "Allow",
    "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
    "Resource": [
      "arn:aws:s3:::bucket",
      "arn:aws:s3:::bucket/{env}/*"
    ]
  },
  "EC2SelfManagement": {
    "Effect": "Allow",
    "Action": ["ec2:TerminateInstances", "ec2:DescribeInstances"],
    "Resource": "*",
    "Condition": {
      "StringEquals": {
        "ec2:ResourceTag/Project": "Chronos-Training"
      }
    }
  }
}
```

**Key Principles**:
- Read-only access to shared datasets
- Write access limited to environment-specific prefix
- Cannot terminate instances from other projects

### Network Security

**Security Group** (`ChronosTrainingSG`):
```
Egress:
  - Port 443 (HTTPS) → 0.0.0.0/0

Ingress:
  - (none)
```

**Rationale**:
- HTTPS egress for AWS API calls, pip installs, apt updates
- No inbound traffic (no SSH, no remote access)
- S3 traffic routes through Gateway Endpoint (no NAT Gateway cost)

### Data Security

- **Encryption at Rest**: S3 default encryption (SSE-S3)
- **Encryption in Transit**: HTTPS for all AWS API calls
- **No Secrets on Instance**: No hardcoded credentials, uses IAM role
- **Ephemeral Storage**: EBS volume deleted on termination

## Cost Optimization

### Zero Idle Cost

| Component | Cost When Idle | Cost During Training |
|-----------|----------------|----------------------|
| VPC | $0 | $0 |
| Subnet | $0 | $0 |
| S3 Gateway Endpoint | $0 | $0 |
| IAM Role | $0 | $0 |
| Security Group | $0 | $0 |
| EC2 (terminated) | $0 | ~$0.75/hr (g4dn.2xlarge) |
| EBS (deleted) | $0 | ~$0.01/hr (100GB) |
| S3 Storage | ~$0.023/GB/mo | ~$0.023/GB/mo |

**Total Idle Cost**: **$0/month** (plus minimal S3 storage)

### Training Cost

**Example**: 2-hour training run on g4dn.2xlarge

| Resource | Cost |
|----------|------|
| g4dn.2xlarge (2 hr) | $1.50 |
| EBS 100GB (2 hr) | $0.02 |
| S3 PUT/GET (1000 requests) | $0.01 |
| S3 Storage (100MB model) | $0.002/mo |
| **Total** | **~$1.53** |

### Cost Reduction Strategies

1. **Use Spot Instances** (not implemented yet)
   - Save 60-70% on EC2 costs
   - Risk: May be interrupted mid-training
2. **Compress Venv** (implemented)
   - Reduces S3 transfer time
3. **Lifecycle Rules** (manual setup required)
   - Auto-delete logs after 14 days
   - Transition old models to Glacier
4. **Smaller Instance for Testing**
   - Use `t3.medium` (CPU-only) for testing scripts
   - Cost: ~$0.04/hr

## Failure Modes

### EC2 Launch Failure

**Symptom**: `launch_training.sh` exits with error

**Causes**:
- CloudFormation exports missing → Re-deploy CDK
- Invalid AMI ID → Check `get_latest_ami()` function
- Instance limit reached → Request limit increase
- IAM permissions missing → Verify IAM role

**Recovery**: Fix issue, re-run `launch_training.sh`

### GPU Preflight Failure

**Symptom**: State file shows `status=failed`, `current_step=gpu_preflight`

**Causes**:
- NVIDIA driver not loaded → Check AMI has GPU drivers
- Venv missing PyTorch → Rebuild venv with `pip install torch`
- Insufficient disk space → Increase EBS volume size

**Recovery**: Fix root cause, re-launch training

### Training Failure

**Symptom**: State file shows `status=failed`, `current_step=training`

**Causes**:
- Training script crashed → Check logs in S3
- Out of memory → Reduce batch size or use larger instance
- Data format mismatch → Validate Parquet schema

**Recovery**: Download logs, debug locally, re-launch

### Network Failure

**Symptom**: Instance stuck at `data_sync` for > 10 minutes

**Causes**:
- S3 Gateway Endpoint misconfigured → Check CDK deployment
- Security Group blocks HTTPS → Verify egress rule
- Large dataset timeout → Increase retry backoff

**Recovery**: Kill instance, fix issue, re-launch

### Zombie Instance

**Symptom**: Instance still running 6+ hours after launch

**Causes**:
- Cleanup script failed → Manual termination required
- State file corruption → Use `kill_training.sh`
- Bug in bootstrap script → Fix bug, redeploy

**Recovery**:
```bash
./kill_training.sh
# Or manual:
INSTANCE_ID=$(aws s3 cp s3://bucket/dev/system-state.json - | jq -r .instance_id)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
```

## Monitoring and Observability

### Real-Time Monitoring

**Watch state file**:
```bash
watch -n 10 "aws s3 cp s3://bucket/dev/system-state.json - | jq ."
```

**Watch instance console output**:
```bash
INSTANCE_ID=$(aws s3 cp s3://bucket/dev/system-state.json - | jq -r .instance_id)
aws ec2 get-console-output --instance-id $INSTANCE_ID
```

### Post-Training Analysis

**Download all logs**:
```bash
RUN_ID=2025-01-15-143022
aws s3 sync s3://bucket/dev/$RUN_ID/logs/ ./logs/
```

**Check training metrics**:
```bash
aws s3 cp s3://bucket/dev/$RUN_ID/training.json - | jq .
```

### CloudWatch Integration (Optional)

- EC2 metrics (CPU, GPU, network) available in CloudWatch
- Set up alarms for long-running instances (> 4 hours)
- Log groups for structured logging (requires CloudWatch agent)

## Next Steps

- Read [CDK Implementation Guide](cdk-implementation.md) for CDK deployment details
- Customize `bootstrap.sh` for your domain-specific needs
- Set up monitoring dashboards (CloudWatch, Grafana)


