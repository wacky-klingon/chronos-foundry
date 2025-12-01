# Usage Guide

Complete guide to implementing custom time series trainers using the Chronos Foundry framework.

## Table of Contents

- [Quick Start](#quick-start)
- [Creating Custom Trainers](#creating-custom-trainers)
- [Configuration](#configuration)
- [Data Loading](#data-loading)
- [Training Workflow](#training-workflow)
- [Model Management](#model-management)
- [AWS Deployment](#aws-deployment)
- [Best Practices](#best-practices)

---

## Quick Start

### 1. Installation

```bash
# Install library
poetry add chronos-foundry

# For development
poetry add chronos-foundry --dev
```

### 2. Create Configuration

```yaml
# config.yaml
data:
  root_dir: "/path/to/your/data"
  output_dir: "./output"

schema:
  datetime_column: "timestamp"
  target_columns: ["value"]
  covariate_columns:
    domain_features: []

model:
  name: "chronos-bolt-base"
  prediction_length: 64
  context_length: 512

training:
  preset: "high_quality"
  time_limit: 3600
```

### 3. Create Trainer

```python
from chronos_trainer import CovariateTrainer, ConfigProvider
import pandas as pd

class MyTrainer(CovariateTrainer):
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Your domain-specific feature engineering
        df['rolling_mean'] = df['value'].rolling(24).mean()
        return df

# Load config and train
config = ConfigProvider.load_config("config.yaml")
trainer = MyTrainer(config)
trainer.train(data)
```

---

## Creating Custom Trainers

### Base Trainer Pattern

All trainers inherit from base classes and override domain-specific methods:

```python
from chronos_trainer import CovariateTrainer
from typing import Dict, Any
import pandas as pd

class MyDomainTrainer(CovariateTrainer):
    """Custom trainer for your specific domain"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Your initialization
        self.domain_specific_params = {}

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Implement domain-specific feature engineering.

        This is called during data preprocessing.
        """
        # Add rolling statistics
        df['rolling_mean_24h'] = df['value'].rolling(24).mean()
        df['rolling_std_24h'] = df['value'].rolling(24).std()

        # Add temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        return df

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """
        Custom data validation logic.

        Return True if valid, False otherwise.
        """
        required_cols = ['timestamp', 'value']
        if not all(col in df.columns for col in required_cols):
            return False

        # Check for reasonable value ranges
        if df['value'].isna().sum() > len(df) * 0.1:  # >10% missing
            return False

        return True

    def _post_process(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Post-process predictions (optional).

        E.g., apply business rules, constraints, etc.
        """
        # Apply domain-specific constraints
        predictions['value'] = predictions['value'].clip(lower=0)
        return predictions
```

### Example: Retail Demand Forecasting

```python
from chronos_trainer import CovariateTrainer
import pandas as pd
import numpy as np

class RetailDemandTrainer(CovariateTrainer):
    """Retail demand forecasting with promotions and holidays"""

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Promotional impact
        df['promo_effect'] = df['is_promotion'] * df['demand'].shift(1)

        # Holiday indicators
        df['is_major_holiday'] = df['date'].isin(self.config['holidays'])
        df['days_until_holiday'] = self._days_until_next_holiday(df['date'])

        # Seasonality
        df['month'] = df['date'].dt.month
        df['week_of_year'] = df['date'].dt.isocalendar().week

        # Trend features
        df['demand_trend_7d'] = df['demand'].rolling(7).mean()
        df['demand_trend_30d'] = df['demand'].rolling(30).mean()

        # Day of week patterns
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        return df

    def _days_until_next_holiday(self, dates: pd.Series) -> pd.Series:
        """Calculate days until next major holiday"""
        holidays = pd.to_datetime(self.config['holidays'])
        return dates.apply(
            lambda d: min((h - d).days for h in holidays if h > d)
        )
```

### Example: Energy Load Prediction

```python
from chronos_trainer import CovariateTrainer
import pandas as pd

class EnergyLoadTrainer(CovariateTrainer):
    """Energy load prediction with weather and calendar features"""

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Temperature features
        df['temp_squared'] = df['temperature'] ** 2
        df['heating_degree_days'] = (65 - df['temperature']).clip(lower=0)
        df['cooling_degree_days'] = (df['temperature'] - 65).clip(lower=0)

        # Time-of-day patterns
        df['hour'] = df['timestamp'].dt.hour
        df['is_peak_hour'] = df['hour'].between(16, 20).astype(int)
        df['is_business_hours'] = df['hour'].between(8, 17).astype(int)

        # Calendar features
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df['timestamp'].dt.month

        # Lag features for load
        df['load_lag_1h'] = df['load'].shift(1)
        df['load_lag_24h'] = df['load'].shift(24)
        df['load_lag_168h'] = df['load'].shift(168)  # 1 week

        # Rolling statistics
        df['load_rolling_mean_24h'] = df['load'].rolling(24).mean()
        df['load_rolling_std_24h'] = df['load'].rolling(24).std()

        return df
```

---

## Configuration

### Complete Configuration Reference

```yaml
# Data paths and directories
data:
  root_dir: "/path/to/data"              # Root data directory
  output_dir: "./output"                 # Model and artifact output
  cache_dir: "./cache"                   # Temporary cache (optional)

# Data schema definition
schema:
  datetime_column: "timestamp"           # Name of datetime column
  target_columns:                        # Columns to forecast
    - "target_value"
  covariate_columns:                     # External variables
    weather: ["temperature", "humidity"]
    calendar: ["is_holiday", "day_of_week"]
    economic: []

# Model configuration
model:
  name: "chronos-bolt-base"              # Model variant
  # Options: chronos-bolt-tiny, chronos-bolt-mini, chronos-bolt-small, chronos-bolt-base
  prediction_length: 64                  # Forecast horizon
  context_length: 512                    # Historical context window
  num_samples: 20                        # Prediction samples (uncertainty)

# Training settings
training:
  preset: "high_quality"                 # Training quality preset
  # Options: best_quality, high_quality, good_quality, medium_quality, fast
  time_limit: 3600                       # Training time limit (seconds)

  # Incremental training
  incremental:
    enabled: true                        # Enable incremental updates
    performance_threshold: 0.05          # Max acceptable degradation (5%)
    rollback_enabled: true               # Auto-rollback on quality loss

# Covariate integration
covariates:
  enabled: true                          # Enable covariate models
  regressors:                            # Ensemble regressors
    - catboost
    - xgboost
    - lightgbm
  validation_split: 0.2                  # Validation set size

# Data processing
processing:
  null_handling: "forward_fill"          # How to handle missing values
  # Options: forward_fill, backward_fill, interpolate, drop
  outlier_detection: true                # Enable outlier detection
  outlier_method: "iqr"                  # Outlier detection method
  outlier_threshold: 3.0                 # IQR multiplier
  chunk_size: 10000                      # Processing chunk size

# Checkpoint settings
checkpointing:
  enabled: true                          # Enable checkpoint saves
  interval: 1000                         # Save every N batches
  keep_last_n: 3                         # Number of checkpoints to keep

# Logging
logging:
  level: "INFO"                          # Log level
  # Options: DEBUG, INFO, WARNING, ERROR
  format: "structured"                   # Log format
  output: "both"                         # Console and file
```

---

## Data Loading

### Basic Data Loading

```python
from chronos_trainer import ResumableDataLoader, ConfigProvider

# Load configuration
config = ConfigProvider.load_config("config.yaml")

# Initialize loader
loader = ResumableDataLoader(config)

# Load data from parquet files
data = loader.load_from_directory(
    directory="/path/to/data",
    pattern="*.parquet",
    start_date="2020-01-01",
    end_date="2023-12-31"
)

print(f"Loaded {len(data)} rows")
```

### Resumable Loading (Large Datasets)

For large date ranges, use checkpoint-based loading:

```python
from chronos_trainer import ResumableDataLoader

loader = ResumableDataLoader(config)

# Load with checkpointing (automatically resumes if interrupted)
data = loader.load_incrementally(
    directory="/path/to/data/YYYY/MM/",  # Temporal structure
    start_date="2010-01-01",
    end_date="2023-12-31",
    checkpoint_file="./checkpoints/load_state.json"
)

# Check loading progress
progress = loader.get_progress()
print(f"Loaded: {progress['months_loaded']}/{progress['total_months']} months")
```

### Custom Data Converter

Create a converter for your data format:

```python
from chronos_trainer.data import BaseDataConverter
import pandas as pd

class MyDataConverter(BaseDataConverter):
    """Convert your data format to Chronos format"""

    def convert(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert to required format:
        - datetime column
        - target columns
        - covariate columns
        """
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['date_column'])

        # Rename target
        df['target'] = df['your_value_column']

        # Select covariates
        covariates = ['feature1', 'feature2', 'feature3']

        # Return clean dataframe
        return df[['timestamp', 'target'] + covariates]
```

---

## Training Workflow

### Complete Training Pipeline

```python
from chronos_trainer import (
    CovariateTrainer,
    ConfigProvider,
    CheckpointManager,
    ModelVersioning
)
import pandas as pd

# 1. Load configuration
config = ConfigProvider.load_config("config.yaml")

# 2. Initialize trainer
class MyTrainer(CovariateTrainer):
    def _engineer_features(self, df):
        df['rolling_mean'] = df['value'].rolling(24).mean()
        return df

trainer = MyTrainer(config)

# 3. Load data
data = pd.read_parquet("/path/to/data.parquet")

# 4. Train model
results = trainer.train(
    data=data,
    validation_split=0.2,
    save_path="./models/my_model"
)

# 5. Evaluate
print(f"Training MAE: {results['mae']:.4f}")
print(f"Training MAPE: {results['mape']:.2f}%")

# 6. Save model with versioning
versioning = ModelVersioning(config)
version_info = versioning.save_version(
    model=trainer.model,
    metrics=results,
    metadata={
        "training_date": pd.Timestamp.now(),
        "data_range": "2020-2023"
    }
)

print(f"Model saved: v{version_info['version']}")
```

### Incremental Training

Update existing model with new data:

```python
from chronos_trainer import IncrementalTrainer

# Load existing model
trainer = IncrementalTrainer.from_checkpoint("./models/my_model")

# Train on new data
new_data = pd.read_parquet("/path/to/new_data.parquet")

results = trainer.train_incremental(
    new_data=new_data,
    performance_threshold=0.05,  # Max 5% degradation
    rollback_on_failure=True
)

if results['rollback_performed']:
    print("⚠️ Model degraded, rolled back to previous version")
else:
    print(f"✓ Model improved: {results['improvement']:.2%}")
```

### Checkpoint Management

Save and resume training:

```python
from chronos_trainer import CheckpointManager

manager = CheckpointManager(config)

# During training - save checkpoint
manager.save_checkpoint(
    trainer=trainer,
    epoch=10,
    metrics={"loss": 0.123},
    path="./checkpoints/epoch_10.ckpt"
)

# Resume from checkpoint
trainer, state = manager.load_checkpoint("./checkpoints/epoch_10.ckpt")
print(f"Resumed from epoch {state['epoch']}")
```

---

## Model Management

### Model Versioning

```python
from chronos_trainer import ModelVersioning

versioning = ModelVersioning(config)

# Save new version
version_info = versioning.save_version(
    model=trainer.model,
    metrics={"mae": 0.05, "mape": 2.1},
    metadata={
        "description": "Added weather features",
        "data_version": "v2.3"
    }
)

# List all versions
versions = versioning.list_versions()
for v in versions:
    print(f"v{v['version']}: MAE={v['metrics']['mae']:.4f}")

# Load specific version
model = versioning.load_version(version=3)

# Promote to production
versioning.promote_to_production(version=5)
```

### Model Evaluation

```python
from chronos_trainer import CovariateTrainer
import pandas as pd

# Load model
trainer = CovariateTrainer.from_checkpoint("./models/my_model")

# Evaluate on test data
test_data = pd.read_parquet("/path/to/test.parquet")

metrics = trainer.evaluate(
    data=test_data,
    metrics=["mae", "mape", "rmse", "mse"]
)

print(f"Test MAE: {metrics['mae']:.4f}")
print(f"Test MAPE: {metrics['mape']:.2f}%")
print(f"Test RMSE: {metrics['rmse']:.4f}")
```

---

## AWS Deployment

### Setup AWS Infrastructure

```bash
# 1. Clone repository for AWS reference
git clone https://github.com/yourusername/chronos-foundry.git
cd chronos-foundry

# 2. Configure environment (in aws/cdk directory)
cd aws/cdk
cp .env.example .env

# Edit .env with your settings:
# S3_BUCKET_NAME=your-bucket-name
# CDK_ENVIRONMENT=dev
# PROJECT_NAME=Chronos-Training
# COST_CENTER=train
```

### Deploy Infrastructure

```bash
cd aws/cdk

# Install dependencies
npm install

# Bootstrap CDK (first time only)
cdk bootstrap

# Deploy stack
cdk deploy

# Outputs:
# - S3 bucket name
# - IAM role ARN
# - Security group ID
```

### Launch Training Job

```bash
cd aws/scripts

# Launch training on EC2
./launch_training.sh

# Output:
# Instance ID: i-1234567890abcdef0
# State file: s3://your-bucket/state/training_TIMESTAMP.json
```

### Monitor Training

```bash
# Watch training progress
./monitor_training.sh

# Shows:
# - Instance status
# - Training progress
# - Logs and metrics
# - Estimated completion
```

### Emergency Stop

```bash
# Kill running training job
./kill_training.sh

# Confirms termination and cleans up resources
```

---

## Best Practices

### 1. Configuration Management

```python
# Use environment-specific configs
config_dev = ConfigProvider.load_config("config_dev.yaml")
config_prod = ConfigProvider.load_config("config_prod.yaml")

# Override with environment variables
import os
config['data']['root_dir'] = os.getenv('DATA_ROOT', config['data']['root_dir'])
```

### 2. Error Handling

```python
from chronos_trainer import CovariateTrainer, ConfigProvider
import logging

logger = logging.getLogger(__name__)

try:
    trainer = MyTrainer(config)
    results = trainer.train(data)
except Exception as e:
    logger.error(f"Training failed: {e}")
    # Send alert, rollback, etc.
    raise
```

### 3. Data Validation

```python
def validate_data(df: pd.DataFrame) -> bool:
    """Comprehensive data validation"""

    # Check required columns
    required = ['timestamp', 'value']
    if not all(col in df.columns for col in required):
        return False

    # Check data types
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        return False

    # Check for duplicates
    if df.duplicated(subset=['timestamp']).any():
        logger.warning("Duplicate timestamps found")
        return False

    # Check value ranges
    if df['value'].isna().sum() > len(df) * 0.1:
        logger.error("Too many missing values")
        return False

    return True
```

### 4. Feature Engineering Testing

```python
def test_feature_engineering():
    """Unit test feature engineering logic"""

    # Create sample data
    df = pd.DataFrame({
        'timestamp': pd.date_range('2020-01-01', periods=100, freq='1H'),
        'value': np.random.randn(100)
    })

    # Apply feature engineering
    trainer = MyTrainer(config)
    df_features = trainer._engineer_features(df)

    # Validate features
    assert 'rolling_mean' in df_features.columns
    assert not df_features['rolling_mean'].isna().all()
```

### 5. Model Registry Pattern

```python
# Maintain model registry
MODEL_REGISTRY = {
    'dev': './models/dev/latest',
    'staging': './models/staging/v1.2.3',
    'production': './models/prod/v1.2.0'
}

def load_model(environment: str):
    """Load model for specific environment"""
    model_path = MODEL_REGISTRY.get(environment)
    return CovariateTrainer.from_checkpoint(model_path)

# Usage
prod_model = load_model('production')
```

### 6. Idempotent Pipelines

```python
import hashlib
import json

def get_data_hash(df: pd.DataFrame) -> str:
    """Generate hash for data version tracking"""
    return hashlib.sha256(
        pd.util.hash_pandas_object(df).values
    ).hexdigest()[:16]

# Check if retraining needed
data_hash = get_data_hash(data)
if data_hash != last_training_hash:
    trainer.train(data)
    save_metadata({'data_hash': data_hash})
```

---

## Updating S3 Data (Admin Task)

When you generate new configs or cached datasets, upload them to S3. Simple process: remove old files and upload new ones.

```bash
# Use admin credentials (trainer-runtime has read-only access)
export AWS_PROFILE=admin
export BUCKET_NAME=YOUR-BUCKET-NAME  # or get from CloudFormation exports

# Upload new cached datasets (replace LOCAL_DATA_PATH with your local path)
aws s3 rm s3://${BUCKET_NAME}/cached-datasets/training-data/ --recursive
aws s3 sync LOCAL_DATA_PATH \
    s3://${BUCKET_NAME}/cached-datasets/training-data/ \
    --exclude "*.tmp" \
    --exclude "*.log" \
    --exclude "__pycache__/*" \
    --exclude "*.pyc" \
    --profile admin

# Upload new config files (navigate to your project directory first)
cd your-project
aws s3 rm s3://${BUCKET_NAME}/cached-datasets/configs/ --recursive
aws s3 cp config/parquet_loader_config.ec2.yaml \
    s3://${BUCKET_NAME}/cached-datasets/configs/parquet_loader_config.yaml \
    --profile admin
aws s3 cp config/train.ec2.yaml \
    s3://${BUCKET_NAME}/cached-datasets/configs/train.yaml \
    --profile admin

# Verify uploads
aws s3 ls s3://${BUCKET_NAME}/cached-datasets/training-data/ --recursive --profile admin | head -20
aws s3 ls s3://${BUCKET_NAME}/cached-datasets/configs/ --profile admin
```

**Note**: This is an **admin-only task**. The `trainer-runtime` user has read-only S3 access.

## Troubleshooting

### Common Issues

**Import Errors:**
```python
# If imports fail, ensure package installed
poetry show chronos-foundry

# Reinstall if needed
poetry install --no-cache
```

**Memory Issues with Large Datasets:**
```python
# Use chunked loading
loader = ResumableDataLoader(config)
for chunk in loader.load_chunks(chunk_size=10000):
    trainer.train_batch(chunk)
```

**AWS Permission Errors:**
```bash
# Verify IAM role has required permissions
aws sts get-caller-identity
aws s3 ls s3://your-bucket/

# If you get AccessDenied, switch to admin credentials:
export AWS_PROFILE=admin
aws sts get-caller-identity  # Should show admin user, not trainer-runtime
```

---

## Additional Resources

- [AWS Quickstart](../getting-started/aws-quickstart.md) - Get started with AWS EC2 training in 5 steps
- [System Architecture](../architecture/system-architecture.md) - Detailed AWS infrastructure documentation
- [AWS Documentation Index](../aws/index.md) - Complete AWS reference
- [Project README](../README.md) - Project overview and setup

---

For questions and issues, please consult the documentation or open an issue on your project repository.

