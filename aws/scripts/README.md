# Training Orchestration Scripts

## Scripts

| Script | Purpose |
|--------|---------|
| `launch_training.sh` | Launch EC2 training run |
| `monitor_training.sh` | Monitor progress via S3 state file |
| `kill_training.sh` | Emergency termination |
| `bootstrap.sh` | EC2 initialization (user data) |
| `cleanup.sh` | Resource cleanup |
| `preflight_check.py` | GPU validation |
| `training_wrapper.py` | Training execution |
| `lib/state_helpers.sh` | Atomic S3 state management |

## Quick Usage

```bash
./launch_training.sh
./monitor_training.sh   # In another terminal
./kill_training.sh     # Emergency stop
```

## Python Dependencies

```bash
pip install -r requirements.txt   # boto3, torch, pytest
```

## Documentation

- [AWS Quickstart](../../docs/getting-started/aws-quickstart.md) - Deploy and run
- [Training Orchestration](../../docs/architecture/training-orchestration.md) - Workflow details
- [State Machine](../../docs/architecture/state-machine.md) - State management and error handling
