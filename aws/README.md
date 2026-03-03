# AWS EC2 Training Infrastructure for Chronos

Production-ready AWS reference implementation for training Chronos models on EC2 GPU instances with zero idle cost.

## Overview

- Ephemeral GPU training: launch, train, auto-terminate
- Zero idle cost: no resources between runs
- S3-based state tracking
- Public subnet + S3 Gateway Endpoint (no NAT cost)

## Directory Structure

```
aws/
├── cdk/                    # Infrastructure as Code (VPC, IAM, Security Group)
│   ├── lib/
│   ├── bin/
│   └── .env.example
├── scripts/                # EC2 orchestration (launch, monitor, kill, bootstrap, cleanup)
│   ├── lib/state_helpers.sh
│   ├── bootstrap.sh
│   ├── launch_training.sh
│   ├── monitor_training.sh
│   └── kill_training.sh
└── README.md
```

## Documentation

- [AWS Quickstart](../docs/getting-started/aws-quickstart.md) - Get started in 5 steps
- [AWS Documentation Index](../docs/aws/index.md) - Complete reference
- [System Architecture](../docs/architecture/system-architecture.md) - Design and troubleshooting
- [CDK Implementation](../docs/architecture/cdk-implementation.md) - Infrastructure as code

## License

Part of chronos-foundry. See root LICENSE file.
