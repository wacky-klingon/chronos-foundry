# Quick Example

Get started with Chronos Foundry in 5 minutes. This example shows how to create a custom trainer and run your first training.

## Prerequisites

- Chronos Foundry installed (see [Installation Guide](installation.md))
- Python 3.10+

## Step 1: Create Configuration

Create a `config.yaml` file:

```yaml
# Data paths
data:
  root_dir: "/path/to/your/data"
  output_dir: "./output"

# Define your schema
schema:
  datetime_column: "timestamp"
  target_columns:
    - "value"
  covariate_columns:
    external_factors: []  # Weather, economic indicators, etc.
    domain_features: []    # Your custom features

# Model configuration
model:
  name: "chronos-bolt-base"
  prediction_length: 64
  context_length: 512

# Training settings
training:
  preset: "high_quality"
  time_limit: 3600
```

## Step 2: Create Custom Trainer

Create `trainer.py`:

```python
from chronos_trainer import CovariateTrainer, ConfigProvider
import pandas as pd

# Load configuration
config = ConfigProvider.load_config("config.yaml")

# Create custom trainer for your domain
class MyDomainTrainer(CovariateTrainer):
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Implement your domain-specific feature engineering"""

        # Example: Add rolling statistics
        df['rolling_mean_24h'] = df['value'].rolling(24).mean()
        df['rolling_std_24h'] = df['value'].rolling(24).std()

        # Example: Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek

        # Your domain-specific logic here
        return df

# Initialize trainer
trainer = MyDomainTrainer(config)

# Load your data (CSV, Parquet, etc.)
data = pd.read_csv("your_data.csv")
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Train the model
trainer.train(data)

# Save the model
trainer.save_model("./models/my_model")
```

## Step 3: Run Training

```bash
python trainer.py
```

That's it! Your model is now trained and saved.

## Next Steps

- [Complete Usage Guide](../user-guides/usage-guide.md) - Learn advanced features
- [AWS Quickstart](aws-quickstart.md) - Deploy training on AWS EC2
- [Testing Guide](../user-guides/testing.md) - Write tests for your trainer

## Common Patterns

### Adding Covariates

```python
# Add external factors
df['external_factor'] = fetch_external_data(df['timestamp'])
```

### Incremental Training

Enable in `config.yaml`:

```yaml
training:
  incremental:
    enabled: true
    performance_threshold: 0.05
    rollback_enabled: true
```

### Checkpoint-Based Training

The trainer automatically handles checkpoints for large datasets. Configure in your config:

```yaml
training:
  checkpoint_dir: "./checkpoints"
```

## Troubleshooting

**Import Error**: Make sure Chronos Foundry is installed
```bash
poetry install  # or pip install chronos-foundry
```

**Data Format Error**: Ensure your data has a datetime column matching `schema.datetime_column`

**GPU Issues**: The library automatically detects GPUs. For CPU-only training, see the usage guide.

