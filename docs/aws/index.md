# AWS Documentation Index

This directory contains all AWS-related documentation for the Chronos Training System.

## Overview

The Chronos Training System is an efficient, on-demand training solution for Chronos forecasting models on AWS EC2 instances with automatic resource cleanup and state file monitoring.

**MVP Scope**: This system excludes NAT Gateways, snapshot caches, CloudWatch dashboards, unified registries, and advanced lifecycle/telemetry. Anything more complex lives in [Future Enhancements](../architecture/future-enhancements.md). The MVP focuses on a lean, single-user, ephemeral trainer with $0 idle costs.

## Quick Navigation

### Getting Started
- **[AWS Quickstart](../getting-started/aws-quickstart.md)** - Get your first training running in 5 steps (~30 minutes)

### Architecture & Design
- **[System Overview](../architecture/overview.md)** - Unified system overview
- **[System Architecture](../architecture/system-architecture.md)** - System architecture, design principles, data flow
- **[Requirements](../architecture/requirements.md)** - Functional and non-functional requirements
- **[State Machine](../architecture/state-machine.md)** - State management and transitions

### Implementation
- **[CDK Implementation Guide](../architecture/cdk-implementation.md)** - CDK infrastructure implementation
- **[Training Orchestration](../architecture/training-orchestration.md)** - EC2 orchestration details

### Advanced & Future
- **[Future Enhancements](../architecture/future-enhancements.md)** - Planned features and enhancements
- **[Architecture Questions](architecture-questions.md)** - Architectural Q&A

## System Features

### State Management
- Single state file per environment (`s3://bucket/{env}/system-state.json`) with atomic writes
- Five execution steps: initialization → data_sync → training → results_sync → cleanup
- Three JSON artifacts per run: `system-state.json`, `training.json`, `cleanup_status.json`
- Real-time progress updates via S3-only monitoring
- Concurrent run prevention

### EC2 Orchestration
- Automatic instance launch and termination in public subnet (no NAT Gateway)
- GPU-optimized instances (g4dn.2xlarge) with ephemeral public IP
- Egress HTTPS-only security group (no inbound access)
- User data scripts for setup
- Comprehensive error handling

### S3 Data Management
- Read-only cached datasets accessed via S3 Gateway Endpoint (zero data transfer cost)
- Timestamped training outputs
- Simple lifecycle policy: IA at 7 days → Glacier at 30 → Delete at 60
- Data integrity verification

### Resource Management
- Automatic resource cleanup (EC2 + EBS volumes with DeleteOnTermination=true)
- Resource tagging for identification
- S3-based state file monitoring (no CloudWatch)
- Zero idle costs: $0/month between runs, ~$2-4 per run

## Quick Start

### For Users:
1. Follow the [AWS Quickstart Guide](../getting-started/aws-quickstart.md)
2. Use the three-command workflow: `launch_training.sh` → `monitor_training.sh` → `kill_training.sh`

### For Developers:
1. Read [Requirements](../architecture/requirements.md) to understand what to build
2. Follow the [CDK Implementation Guide](../architecture/cdk-implementation.md) for implementation
3. Review [Training Orchestration](../architecture/training-orchestration.md) for workflow details

### For Architects:
1. Read [System Architecture](../architecture/system-architecture.md) for design
2. Study [Training Orchestration](../architecture/training-orchestration.md) for implementation details
3. Review [State Machine](../architecture/state-machine.md) for state management

## Management Commands

The system provides three simple commands:

```bash
# Launch training
./aws/scripts/launch_training.sh

# Monitor progress
./aws/scripts/monitor_training.sh

# Kill training (if needed)
./aws/scripts/kill_training.sh
```

## Technical Stack

- **AWS Services**: EC2, S3, IAM
- **Infrastructure**: CDK (Infrastructure as Code) - VPC, subnet, S3 Gateway Endpoint, IAM roles only
- **State Management**: S3-based state file with atomic writes
- **Monitoring**: S3-only (no CloudWatch in MVP)
- **Security**: IAM roles, encryption, public subnet with egress-only security group
- **Networking**: Public subnet + Internet Gateway + S3 Gateway Endpoint (zero idle cost)

## Related Documentation

- [Complete Usage Guide](../user-guides/usage-guide.md) - Library usage
- [Architecture Documentation](../architecture/README.md) - System design
- [Main Documentation Index](../README.md) - All documentation

## System Overview

The Chronos Training System is a single-user, state-managed training orchestration system that:

- **Prevents concurrent runs** through state file locking
- **Manages EC2 instances** automatically with GPU optimization (launched in public subnet with ephemeral public IP)
- **Synchronizes data** between S3 and EC2 with simple lifecycle management
- **Provides monitoring** via S3-based state files (no CloudWatch in MVP)
- **Manages resources** through automatic cleanup (EC2 termination + DeleteOnTermination volumes)

## Key Features

### **State Management**
- Single state file per environment (`s3://bucket/{env}/system-state.json`) with atomic writes
- Five execution steps: initialization → data_sync → training → results_sync → cleanup
- Three JSON artifacts per run: `system-state.json`, `training.json`, `cleanup_status.json`
- Real-time progress updates via S3-only monitoring
- Concurrent run prevention

### **EC2 Orchestration**
- Automatic instance launch and termination in public subnet (no NAT Gateway)
- GPU-optimized instances (g4dn.2xlarge) with ephemeral public IP
- Egress HTTPS-only security group (no inbound access)
- User data scripts for setup
- Comprehensive error handling

### **S3 Data Management**
- Read-only cached datasets accessed via S3 Gateway Endpoint (zero data transfer cost)
- Timestamped training outputs
- Simple lifecycle policy: IA at 7 days → Glacier at 30 → Delete at 60
- Data integrity verification

### **Resource Management**
- Automatic resource cleanup (EC2 + EBS volumes with DeleteOnTermination=true)
- Resource tagging for identification
- S3-based state file monitoring (no CloudWatch)
- Zero idle costs: $0/month between runs, ~$2-4 per run

## Management Commands

The system provides three simple commands:

```bash
# Launch training
./aws/scripts/launch_training.sh

# Monitor progress
./aws/scripts/monitor_training.sh

# Kill training (if needed)
./aws/scripts/kill_training.sh
```

## Use Cases Covered

1. **Standard Training Run** - Launch, monitor, complete
2. **Concurrent Run Prevention** - Clear error messages
3. **Emergency Termination** - Kill script with cleanup
4. **Progress Monitoring** - Real-time status updates
5. **Resource Management** - Resource optimization and control

## Technical Stack

- **AWS Services**: EC2, S3, IAM
- **Infrastructure**: CDK (Infrastructure as Code) - VPC, subnet, S3 Gateway Endpoint, IAM roles only
- **State Management**: S3-based state file with atomic writes
- **Monitoring**: S3-only (no CloudWatch in MVP)
- **Security**: IAM roles, encryption, public subnet with egress-only security group
- **Networking**: Public subnet + Internet Gateway + S3 Gateway Endpoint (zero idle cost)

## Getting Started

1. **Read the Requirements** ([Requirements](../architecture/requirements.md) - includes user acceptance criteria)
2. **Learn to Use** ([Usage Guide](../user-guides/usage-guide.md))
3. **Understand Architecture** ([Training Orchestration](../architecture/training-orchestration.md) - includes S3 data flow)
4. **Implement** ([CDK Implementation](../architecture/cdk-implementation.md))
5. **Review Architecture** ([System Architecture](../architecture/system-architecture.md) - complete system design)

## Support

For questions or issues:
1. Check the [Usage Guide](../user-guides/usage-guide.md) for troubleshooting
2. Review the [Requirements](../architecture/requirements.md) for system capabilities
3. Consult the [CDK Implementation Guide](../architecture/cdk-implementation.md) for technical details

---

**Note**: This is a single-user system designed for efficient, on-demand training with automatic resource cleanup and state file monitoring.
