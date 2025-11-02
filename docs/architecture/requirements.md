# Chronos Training System Requirements

## Overview

This document defines the functional and non-functional requirements for the Chronos training system. The system enables cost-effective, on-demand training of Chronos models on AWS EC2 instances with automatic resource cleanup and S3-based monitoring.

**MVP Scope**: This system excludes NAT Gateways, snapshot caches, CloudWatch dashboards, unified registries, and advanced lifecycle/telemetry. Anything more complex lives in [Future Enhancements](future-enhancements.md).

## User-Centric Requirements Summary

### Primary User: Data Scientist
**Goals**: Train Chronos models efficiently with minimal manual intervention
**Pain Points**: Manual resource management, lack of progress visibility, cost overruns

### MVP Must-Have Capabilities

| Epic | User Need | MVP Acceptance Criteria | Story Points |
|------|-----------|------------------------|--------------|
| **Training Execution** | Launch training with single command | ✓ `launch_training.sh` checks for existing sessions<br>✓ EC2 launches automatically<br>✓ Confirmation on start | 8 |
| | Monitor progress in real-time | ✓ `monitor_training.sh` shows current step<br>✓ Updates every 30 seconds<br>✓ Clear progress indicators | 5 |
| | Emergency termination | ✓ `kill_training.sh` stops training<br>✓ Resources cleaned up<br>✓ No orphaned resources | 5 |
| **State Management** | Prevent concurrent runs | ✓ Error if training already running<br>✓ Shows current session status | 3 |
| | Track training progress | ✓ See current step (init/sync/train/cleanup)<br>✓ Timestamps for each step | 5 |
| | Handle failures gracefully | ✓ Clear error messages<br>✓ Partial results preserved<br>✓ Resources cleaned up | 8 |
| **Resource Management** | Automatic cleanup | ✓ EC2 terminates after training<br>✓ EBS volumes deleted (DeleteOnTermination=true)<br>✓ Cleanup verification available | 5 |
| | S3 lifecycle management | ✓ Old runs deleted automatically<br>✓ Data transitions to cheaper storage<br>✓ Simple 2-rule lifecycle policy | 8 |
| | Cost prevention | ✓ Resource tagging for attribution<br>✓ `verify_cleanup.sh` available<br>✓ Manual cost review via AWS Cost Explorer | 3 |
| **Data Management** | Auto-sync training data | ✓ Cached datasets pulled from S3<br>✓ Data integrity verified<br>✓ Retry on failed sync | 8 |
| | Auto-store results | ✓ Outputs uploaded to timestamped S3 dirs<br>✓ Models, logs, checkpoints preserved | 5 |
| **Security** | Secure data access | ✓ IAM roles (no credentials)<br>✓ S3 encryption at rest/transit<br>✓ Least privilege access | 8 |
| **Usability** | Simple interface | ✓ Three commands (launch/monitor/kill)<br>✓ Clear feedback<br>✓ Helpful error messages | 5 |

**MVP Total Story Points**: 72

### Deferred to Phase 2

- **Advanced Monitoring**: CloudWatch metrics, dashboards, email/SMS alerts
- **EBS Snapshot Management**: Snapshot-cache for faster boot
- **NAT Gateway**: Private subnet option for compliance
- **Advanced Cost Tracking**: Real-time cost estimates, per-run reports
- **Training Reports**: Automated formatted summaries

See [Future Enhancements](future-enhancements.md) for deferred features.

## Functional Requirements

### FR-001: Single-User System

**Requirement**: The system must support only one active training session at a time.

**Acceptance Criteria**:

- System prevents concurrent training runs
- Clear error messages when attempting concurrent runs
- State file-based locking mechanism

**Priority**: High

### FR-002: State Management

**Requirement**: The system must track training progress through a state file.

**Acceptance Criteria**:

- State file stored in S3 (`s3://my-bucket/phase1/system-state.json`)
- Five major execution steps tracked
- Real-time progress updates
- State file removed upon completion

**Priority**: High

### FR-003: EC2 Instance Management

**Requirement**: The system must automatically manage EC2 instances for training.

**Acceptance Criteria**:

- Automatic instance launch with user data scripts
- GPU-optimized instance types (g4dn.2xlarge)
- Automatic instance termination after training
- Proper resource cleanup

**Priority**: High

### FR-004: Python Environment Management

**Requirement**: The system must use pre-built Python virtual environments for fast EC2 startup.

**Acceptance Criteria**:

- Pre-built .venv stored in S3 (`s3://bucket/cached-datasets/python-env/`)
- Python 3.11.13 environment with all dependencies
- Poetry used only on developer machine; EC2 never runs Poetry
- Bundle pyproject.toml, poetry.lock, and venv-metadata.json for provenance
- GPU preflight validation (nvidia-smi and torch.cuda.is_available())
- Run-scoped logging under phase1/logs/<run_id>/
- Auto-update support (upload new .venv, next run uses it)

**Priority**: High

### FR-005: Data Synchronization

**Requirement**: The system must synchronize data between S3 and EC2 instances.

**Acceptance Criteria**:

- Cached datasets pulled from read-only S3 store
- Training outputs uploaded to timestamped S3 directories
- Retry logic for S3 operations
- Data integrity verification

**Priority**: High

### FR-006: Training Execution

**Requirement**: The system must execute Chronos model training.

**Acceptance Criteria**:

- Support for incremental training
- Multiple model types (Chronos, AutoGluon)
- Comprehensive logging
- Error handling and recovery

**Priority**: High

### FR-006: Progress Monitoring

**Requirement**: The system must provide real-time progress monitoring.

**Acceptance Criteria**:

- Real-time status updates through state file
- Step-by-step progress tracking
- Clear status messages
- Monitoring script for dev machine

**Priority**: Medium

### FR-007: Configuration Override

**Requirement**: The system must support user configuration overrides.

**Acceptance Criteria**:

- User configs uploaded from dev machine to S3
- User configs override repository configs
- Support for multiple config file formats (YAML, JSON)
- Config validation before training execution
- Clear error messages for invalid configs

**Priority**: High

### FR-008: Model Metadata Generation

**Requirement**: The system must generate minimal model metadata for each training run.

**Acceptance Criteria**:

- Automatic metadata file generation (model_metadata.json)
- Metadata includes training timestamps, dataset information, Git commit hash
- Metadata includes software versions (Chronos, Python, AutoGluon)
- Metadata includes key performance metrics (MAE, Sharpe ratio, validation scores)
- Metadata stored alongside model artifacts in S3

**Note**: API endpoint access and report generation deferred to Phase 2 (see [Future Enhancements](future-enhancements.md)).

**Priority**: High

### FR-009: Unified Registry Management (DEFERRED TO PHASE 2)

**Requirement**: ~~The system must maintain a unified registry for all training runs.~~

**MVP Approach**: Use timestamped S3 prefixes and `aws s3 ls` to find runs. No central registry file needed for single-user operation.

**Rationale**: Single user can easily search S3 prefixes; central registry adds write-ordering complexity without clear benefit until multi-user or query needs emerge.

**Deferred to**: [Future Enhancements](future-enhancements.md)

**Priority**: ~~High~~ → Phase 2

### FR-010: Automatic Timeout Protection

**Requirement**: The system must automatically terminate training runs that exceed maximum duration.

**Acceptance Criteria**:

- Watcher script runs every 20 seconds as separate process
- Automatic termination after 12 hours from EC2 boot time
- Log timeout event and sync to S3 before cleanup
- Execute cleanup script for proper resource deletion
- Configurable timeout duration

**Priority**: High

### FR-011: Emergency Termination

**Requirement**: The system must support emergency termination of training.

**Acceptance Criteria**:

- Kill script for emergency termination
- Instance termination with cleanup
- State file removal
- Resource verification

**Priority**: Medium

### FR-012: Basic Resource Cleanup

**Requirement**: The system must clean up ephemeral AWS resources to prevent cost accumulation.

**Acceptance Criteria**:

- EC2 instances terminate automatically after training
- EBS volumes deleted via DeleteOnTermination=true
- S3 data lifecycle policy for automatic data management
- Resource cleanup verification script available

**Note**: NAT Gateway cleanup, EBS snapshot management, CloudWatch log retention, and advanced cost monitoring deferred to Phase 2 (see [Future Enhancements](future-enhancements.md)).

**Priority**: High

### FR-013: Basic Cost Prevention

**Requirement**: The system must prevent runaway costs through automatic cleanup.

**Acceptance Criteria**:

- Resource tagging for cost attribution
- S3 lifecycle policy for data transition and deletion
- Cleanup verification script available
- Manual cost review via AWS Cost Explorer

**Note**: Real-time cost monitoring, automated alerts, snapshot lifecycle management, and per-run cost reporting deferred to Phase 2 (see [Future Enhancements](future-enhancements.md)).

**Priority**: High

### FR-014: Resource Lifecycle Management

**Requirement**: The system must clearly categorize and manage resources by lifecycle type.

**Acceptance Criteria**:

- Manual actions/resources clearly identified and documented
- Long-term infrastructure resources documented with ownership
- Ephemeral training resources automatically managed
- Resource dependencies and prerequisites defined
- Implementation phases reflect resource lifecycle types
- Operational procedures defined for each resource type

**Priority**: High

### FR-015: Operational Procedures

**Requirement**: The system must provide clear operational procedures for different resource types.

**Acceptance Criteria**:

- Manual setup procedures documented with step-by-step instructions
- Long-term resource management procedures with maintenance schedules
- Ephemeral resource automation procedures with verification
- Resource verification and cleanup procedures
- Cost monitoring procedures for each resource type
- Troubleshooting procedures for resource failures

**Priority**: High

## Non-Functional Requirements

### NFR-001: Performance

**Requirement**: The system must complete training within acceptable timeframes.

**Acceptance Criteria**:

- Training completion within 4-6 hours for standard datasets
- GPU acceleration for Chronos models
- Efficient data synchronization
- Minimal overhead for state management
- Fast EC2 startup using pre-built Python environments (2-3 minutes vs 15-30 minutes)

**Priority**: High

### NFR-002: Reliability

**Requirement**: The system must be reliable and fault-tolerant.

**Acceptance Criteria**:

- Automatic error handling and recovery
- Retry logic for S3 operations
- Graceful failure handling
- Resource cleanup under all failure scenarios

**Priority**: High

### NFR-003: Security

**Requirement**: The system must implement appropriate security measures.

**Acceptance Criteria**:

- IAM roles with least privilege access
- S3 encryption at rest and in transit
- Private subnets for EC2 instances
- No stored credentials

**Priority**: High

### NFR-004: Cost Efficiency

**Requirement**: The system must optimize costs and prevent waste.

**Acceptance Criteria**:

- Automatic resource cleanup after each training run
- Zero idle costs ($0/month between training runs)
- S3 Gateway Endpoint for zero-cost S3 data transfer
- Simple S3 lifecycle policy for automated data management
- Manual cost review through AWS Cost Explorer
- Efficient resource utilization
- Training run cost target: ~$2-4 per run (EC2 + minimal networking)

**Note**: Automated cost monitoring and per-run cost reporting deferred to Phase 2 (see [Future Enhancements](future-enhancements.md)).

**Priority**: High

### NFR-005: Usability

**Requirement**: The system must be easy to use and understand.

**Acceptance Criteria**:

- Simple three-command interface
- Clear error messages
- Comprehensive documentation
- Intuitive progress monitoring

**Priority**: Medium

### NFR-006: Maintainability

**Requirement**: The system must be maintainable and extensible.

**Acceptance Criteria**:

- Modular architecture
- Clear separation of concerns
- Comprehensive documentation
- Easy to modify and extend

**Priority**: Medium

### NFR-007: Scalability

**Requirement**: The system must support future enhancements.

**Acceptance Criteria**:

- Support for multiple instance types
- Extensible architecture
- Support for additional data sources
- Future enhancement capabilities

**Priority**: Low

## Technical Requirements

### TR-001: AWS Infrastructure

**Requirement**: The system must use AWS services for infrastructure.

**Acceptance Criteria**:

- EC2 instances for compute
- S3 for data storage and monitoring
- IAM for access control

**Note**: CloudWatch monitoring deferred to Phase 2 - MVP uses S3-only monitoring (see [Future Enhancements](future-enhancements.md)).

**Priority**: High

### TR-002: CDK Implementation

**Requirement**: The system must use CDK for infrastructure as code.

**Acceptance Criteria**:

- CDK stack for all AWS resources
- Infrastructure as code
- Version control for infrastructure
- Reproducible deployments

**Priority**: High

### TR-003: Network Infrastructure (MVP: Public Subnet Only)

**Requirement**: The system must implement simple, cost-optimized networking with zero idle costs.

**Acceptance Criteria**:

- VPC with public subnet for EC2 instances (persistent, no cost)
- Internet Gateway attached to VPC (persistent, no cost)
- S3 Gateway Endpoint for free S3 access (persistent, no cost)
- Training instances launched in public subnet with ephemeral public IP
- No inbound security group rules (egress HTTPS only)
- All S3 traffic routes through Gateway Endpoint (zero data transfer cost)
- Internet access for package installation routes through Internet Gateway
- Security groups restrict outbound traffic to HTTPS only

**Cost Impact**: $0/month idle cost, ~$0.04/run for internet egress, $0 for S3 data transfer

**Security Rationale**: Public IP is acceptable for ephemeral (2-4 hour), single-user training instances with no inbound access and egress-only security group.

**Priority**: High

**Note**: Private subnet with ephemeral NAT Gateway is available as Phase 2 option for compliance scenarios (see [Future Enhancements](future-enhancements.md))

### TR-004: State File Format

**Requirement**: The state file must use JSON format.

**Acceptance Criteria**:

- JSON format for state file
- Required fields: status, instance_id, current_step, timestamps
- Optional fields: training_config, error_message
- Human-readable format

**Priority**: Medium

### TR-005: Management Scripts

**Requirement**: The system must provide management scripts.

**Acceptance Criteria**:

- Launch script (`launch_training.sh`)
- Monitor script (`monitor_training.sh`)
- Kill script (`kill_training.sh`)
- Executable permissions
- Error handling
- Cleanup verification script (`verify_cleanup.sh`)

**Priority**: High

### TR-006: Resource Cleanup Scripts

**Requirement**: The system must provide basic resource cleanup scripts.

**Acceptance Criteria**:

- Cleanup script runs automatically at end of training
- EC2 instance termination
- S3 lifecycle policy configured via CDK
- Resource verification script (`verify_cleanup.sh`)
- Cleanup status logging

**Note**: NAT Gateway deletion, snapshot pruning, and advanced cost monitoring scripts deferred to Phase 2 (see [Future Enhancements](future-enhancements.md)).

**Priority**: High

## Data Requirements

### DR-001: Cached Datasets

**Requirement**: The system must support cached datasets in S3.

**Acceptance Criteria**:

- Read-only access to cached datasets
- Parquet format support
- Organized by year/month
- Data integrity verification

**Priority**: High

### DR-002: Training Outputs

**Requirement**: The system must store training outputs in S3.

**Acceptance Criteria**:

- Timestamped directory structure
- Models, logs, and checkpoints
- Automatic lifecycle management
- Cost optimization

**Priority**: High

### DR-003: State File

**Requirement**: The system must maintain a state file for progress tracking.

**Acceptance Criteria**:

- Single state file per training session
- Real-time updates
- Automatic cleanup
- Error handling

**Priority**: High

### DR-004: Configuration Files

**Requirement**: The system must support user configuration files.

**Acceptance Criteria**:

- User configs stored in S3 phase1 folder
- Support for YAML and JSON formats
- Config precedence: user configs > repository configs
- Config validation and error reporting
- Automatic config application during training

**Priority**: High

### DR-005: Model Metadata and Provenance

**Requirement**: The system must track model provenance and metadata.

**Acceptance Criteria**:

- Model metadata file (model_metadata.json) generated for each training run
- Metadata includes training timestamps, dataset paths, Git commit hash, versions
- Metadata includes key performance metrics (MAE, Sharpe ratio, etc.)
- Metadata stored alongside model artifacts in S3
- Metadata accessible via inference service /info endpoint

**Priority**: High

### DR-006: Unified Registry System (DEFERRED TO PHASE 2)

**Requirement**: ~~The system must maintain a unified registry for tracking runs.~~

**MVP Approach**: Use timestamped S3 prefixes (`phase1/YYYY/MM/DD-HHMMSS/`) to track runs. Use `aws s3 ls` to search.

**Deferred to**: [Future Enhancements](future-enhancements.md)

**Priority**: ~~High~~ → Phase 2

### DR-007: Resource Cleanup Data

**Requirement**: The system must track resource cleanup status.

**Acceptance Criteria**:

- Cleanup status file (cleanup_status.json) for each training run
- Resource inventory before and after cleanup
- S3 lifecycle policy configured via CDK
- Cleanup verification script available

**Note**: Detailed cost tracking per resource type and snapshot metadata deferred to Phase 2 (see [Future Enhancements](future-enhancements.md)).

**Priority**: High

## Interface Requirements

### IR-001: Command Line Interface

**Requirement**: The system must provide a command line interface.

**Acceptance Criteria**:

- Three main commands (launch, monitor, kill)
- Clear error messages
- Progress feedback
- Help documentation

**Priority**: High

### IR-002: AWS CLI Integration

**Requirement**: The system must integrate with AWS CLI.

**Acceptance Criteria**:

- AWS CLI for S3 operations
- AWS CLI for EC2 management
- AWS CLI for monitoring
- Error handling

**Priority**: High

### IR-003: Monitoring Interface

**Requirement**: The system must provide S3-based monitoring capabilities.

**Acceptance Criteria**:

- Real-time progress monitoring via S3 state file
- Status file display via `monitor_training.sh`
- Error reporting in JSON artifacts
- Manual completion check via state file

**Note**: Automated completion notifications (email/SMS) deferred to Phase 2 (see [Future Enhancements](future-enhancements.md)).

**Priority**: Medium

## Quality Requirements

### QR-001: Error Handling

**Requirement**: The system must handle errors gracefully.

**Acceptance Criteria**:

- Clear error messages
- Automatic retry logic
- Graceful failure handling
- Resource cleanup

**Priority**: High

### QR-002: Logging

**Requirement**: The system must provide comprehensive logging.

**Acceptance Criteria**:

- Training process logs
- Error logs
- Progress logs
- Audit trail

**Priority**: Medium

### QR-003: Testing

**Requirement**: The system must be testable.

**Acceptance Criteria**:

- Unit tests for components
- Integration tests for workflows
- End-to-end tests for scenarios
- Test documentation

**Priority**: Medium

## Constraints

### C-001: Single User

**Constraint**: The system is designed for single-user operation.

**Impact**: No concurrent training sessions, simplified state management

### C-002: AWS Only

**Constraint**: The system must use AWS services exclusively.

**Impact**: Vendor lock-in, AWS-specific implementation

### C-003: Cost Optimization

**Constraint**: The system must optimize costs.

**Impact**: Automatic cleanup, lifecycle policies, cost monitoring

### C-004: KISS Principle

**Constraint**: The system must follow the KISS principle.

**Impact**: Simple design, minimal complexity, easy to understand

## Success Criteria

### SC-001: Training Completion

**Success**: Training completes successfully within expected timeframe.

**Metrics**: 95% success rate, <6 hours completion time

### SC-002: Cost Control

**Success**: Training costs are predictable and controlled.

**Metrics**: ~$2-4 per training run (EC2 + minimal networking), automatic cleanup, zero idle costs

### SC-003: User Experience

**Success**: Users can easily launch, monitor, and manage training.

**Metrics**: <5 minutes to launch, clear progress updates

### SC-004: Reliability

**Success**: System handles failures gracefully.

**Metrics**: 99% uptime, automatic recovery

## Dependencies

### D-001: AWS Account

**Dependency**: Valid AWS account with appropriate permissions.

**Impact**: Required for all operations

### D-002: AWS CLI

**Dependency**: AWS CLI configured with credentials.

**Impact**: Required for management scripts

### D-003: CDK

**Dependency**: CDK installed and configured.

**Impact**: Required for infrastructure deployment

### D-004: S3 Bucket

**Dependency**: S3 bucket with cached datasets.

**Impact**: Required for training data

## Assumptions

### A-001: Single User

**Assumption**: Only one user will use the system at a time.

**Impact**: Simplified state management, no concurrency issues

### A-002: AWS Environment

**Assumption**: System will run in AWS environment.

**Impact**: AWS-specific implementation, vendor lock-in

### A-003: Training Data

**Assumption**: Training data is available in S3.

**Impact**: Data dependency, S3 integration required

### A-004: Network Access

**Assumption**: EC2 instances have internet access for dependencies.

**Impact**: NAT gateway required, network configuration

This requirements document provides a comprehensive foundation for the Chronos training system implementation.
