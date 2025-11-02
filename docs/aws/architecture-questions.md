# S3 Sync Architecture Design Questions

**Note**: This document contains architectural Q&A from the design phase.

## Workflow Overview

### A. On Dev Machine
0. Initialize the EC2 instance, EBS storage, verify basic infrastructure
1. Upload executor scripts and config
   - Config: can be but not limited to GitHub URL, branch, config files to override what's in the repo

### B. On EC2 Instance - Transfer Following Instructions
0. Checkout latest code from GitHub (means setup of SSH key etc)
1. Run init script to pull in all data from S3 (cached dataset and trained models if any), run a basic verification
2. Run training script to train model, once complete irrespective of status of training, push to S3

### C. On Dev Machine Start Monitoring Scripts
- C.1 - Script to monitor resources and health of EC2 instance
- C.2 - Script to monitor progress of training on EC2 instance

### D. Once Training Process is Done
1. Check the S3 sync has happened
2. List running resources on AWS, drop all cost generating resources EC2 instance, EBS volumes etc
3. Print a report

## Data Flow Questions

### 1. S3 Bucket Structure
**Question**: What's the S3 bucket structure?

**Answer**:
```
s3://your-bucket/<other paths>/
├── cached-datasets/          # Read-only, source of truth
├── logs/                     # Bidirectional sync
├── models/                   # Bidirectional sync
└── checkpoints/              # Bidirectional sync
```
Open to design improvements

### 2. Local vs S3 Paths
**Question**: Are the local paths different from S3 paths?

**Answer**:
We need to map S3 path to local path, using the S3 sync or tools. Let's assume that they are separate and require one-to-one mapping.

For example:
- **Local**: `/home/user/data/cached_datasets/` → **S3**: `s3://bucket/cached-datasets/`
- **Local**: `/home/user/logs/` → **S3**: `s3://bucket/logs/`

### 3. Sync Direction
- **Cached datasets**: S3 → Local only (read-only)
- **Logs**: Bidirectional (Local ↔ S3)
- **Models**: Bidirectional (Local ↔ S3)
- **Checkpoints**: Bidirectional (Local ↔ S3)

### 4. Sync Triggers
**Question**: When does sync happen?

**Answer**:
- **Start of training**: Pull cached datasets from S3
- **End of training**: Push models/logs to S3
- **Periodic**: No periodic or automated pushes
- **Manual**: User-triggered sync only

### 5. Conflict Resolution
- If local and S3 files differ, files on EC2 instance take precedence
- For logs: Local overwrites S3
- For models: Local overwrites S3
- For checkpoints: Local most recent wins

### 6. File Types
- **Cached datasets**: Parquet files only
- **Logs**: Text files, JSON files
- **Models**: Pickle files, JSON metadata
- **Checkpoints**: JSON files, model files

### 7. Size Considerations
- **Cached datasets**: Assume 10GB
- **Models**: Assume 2GB
- **Sync type**: Full sync (not incremental)

### 8. Access Patterns
- **Read**: Training reads from local cached datasets
- **Write**: Training writes to local logs/models/checkpoints
- **Sync**: Push to S3 once process is over (nothing periodic)

### 9. Error Handling
**Requirements**:
- Cleanup of resources on EC2 and EBS is paramount - no ghost costs
- Otherwise follow best practices and the workflow described above

### 10. Security
- **IAM roles**: Keep it simple (KISS principle)
- **Encryption**: At rest using customer-provided keys
- **Versioning**: No S3 versioning (KISS principle)

## Proposed Architecture

Based on the design requirements:

```
Local EC2 Instance:
├── data/
│   ├── cached_datasets/     # S3 → Local (read-only)
│   ├── logs/                # Local ↔ S3 (bidirectional)
│   ├── models/              # Local ↔ S3 (bidirectional)
│   └── checkpoints/         # Local ↔ S3 (bidirectional)

S3 Bucket:
├── cached-datasets/         # Source of truth (read-only)
├── logs/                    # Bidirectional sync
├── models/                  # Bidirectional sync
└── checkpoints/             # Bidirectional sync
```

## Additional Questions

### 11. Training Workflow
**Question**: How does the training workflow integrate with S3 sync?

**Answer**: None - we sync the files manually via scripts pre and post training.
- Do you sync before training starts? Yes
- Do you sync after training completes? Yes
- Do you sync checkpoints during training? No

### 12. Multiple Instances
**Question**: Will multiple EC2 instances access the same S3 bucket?

**Answer**: Only one. Frugality is important, so is efficiency.
- How do you handle concurrent access to models/checkpoints? None - only one process at a time
- Do you need locking mechanisms? No

### 13. Data Lifecycle
**Question**: How long do you keep data in S3?

**Answer**: Use smart tier setup in S3, which will let S3 automatically manage data for use.
- Do you need data archival policies? Same as above
- Do you need cleanup of old models/logs? If we run the current cleanup script or its variations and run S3 sync, it should clean up locally and the S3 sync will clean up the bucket

### 14. Monitoring
- Do you need monitoring of sync operations? Yes (see monitoring scripts above)
- Do you need alerts for sync failures? Simple console log will suffice
- Do you need metrics on sync performance? For phase 2

### 15. Backup Strategy
- Is S3 the primary storage or backup? Both
- Do you need additional backup strategies? Phase 2
- Do you need cross-region replication? No

