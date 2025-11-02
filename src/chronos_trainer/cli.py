"""
Command Line Interface for chronos-foundry

Provides CLI commands for training and managing Chronos models.
"""

import click
import sys
import json
from pathlib import Path
from typing import Optional

from .core.config_provider import CentralConfigProvider, ConfigValidationError
from .core.logging_manager import LoggingManager
from .training import ChronosTrainer, CovariateTrainer, IncrementalTrainer
from .training.checkpoint_manager import CheckpointManager
from .data import ResumableDataLoader
from .core.config_helpers import ConfigHelpers


@click.group()
@click.option(
    "--config",
    "-c",
    multiple=True,
    help="Configuration file path (can be specified multiple times)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx, config, verbose):
    """Chronos Trainer - Production time series forecasting framework"""

    # Initialize logging first (before any other operations)
    logging_config = {
        "level": "DEBUG" if verbose else "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file_logging": False,  # Disable file logging for CLI simplicity
    }

    logging_manager = LoggingManager(logging_config)
    logging_manager.initialize_logging()

    logger = logging_manager.get_logger("chronos_trainer.cli")
    logger.info("Chronos Trainer CLI starting")

    # Initialize configuration provider
    try:
        # If no config provided, use defaults or fail
        config_files = list(config) if config else []

        if not config_files:
            # Try to find default config files
            default_configs = [
                "config/training_config.yaml",
                "src/chronos_trainer/config/templates/training_config.yaml",
            ]
            for default_path in default_configs:
                if Path(default_path).exists():
                    config_files.append(default_path)
                    logger.info(f"Using default config: {default_path}")
                    break

        if not config_files:
            logger.warning(
                "No configuration files specified. Some commands may require explicit --config"
            )

        if config_files:
            config_provider = CentralConfigProvider(config_files)
            config_provider.initialize()
        else:
            config_provider = None

        # Store in context for subcommands
        ctx.ensure_object(dict)
        ctx.obj["config_provider"] = config_provider
        ctx.obj["logger"] = logger
        ctx.obj["logging_manager"] = logging_manager

    except ConfigValidationError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize CLI: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    help="Training configuration YAML file",
)
@click.option(
    "--start-date",
    help="Start date for training data (YYYY-MM-DD). Can also be specified in config.",
)
@click.option(
    "--end-date",
    help="End date for training data (YYYY-MM-DD). Can also be specified in config.",
)
@click.option(
    "--model-path",
    help="Path to save trained model (defaults to config value)",
)
@click.option(
    "--use-covariates",
    is_flag=True,
    help="Enable covariate integration",
)
@click.pass_context
def train(ctx, config_file, start_date, end_date, model_path, use_covariates):
    """Train a Chronos model on the provided data"""

    logger = ctx.obj["logger"]

    try:
        logger.info("Starting model training")

        # Get configuration
        config_provider = ctx.obj["config_provider"]
        if config_provider is None:
            if not config_file:
                raise ConfigValidationError(
                    "Configuration required. Specify --config-file or use --config in CLI base command"
                )
            config_provider = CentralConfigProvider([config_file])
            config_provider.initialize()

        # Get merged config
        full_config = config_provider.get_merged_config(
            include_covariates=use_covariates
        )

        # Override model path if provided
        if model_path:
            full_config["model_path"] = model_path

        # Get date range from arguments or config
        if not start_date:
            start_date = full_config.get("start_date")
        if not end_date:
            end_date = full_config.get("end_date")

        if not start_date or not end_date:
            raise ValueError(
                "Start date and end date must be provided either via --start-date/--end-date "
                "or in the configuration file"
            )

        # Initialize data loader
        base_data_path = ConfigHelpers.get_parquet_root_dir(full_config)
        data_loader = ResumableDataLoader(base_data_path, checkpoint_manager=None)

        # Load data for the date range
        logger.info(f"Loading data from {start_date} to {end_date}")

        parquet_files = data_loader.get_parquet_files(start_date, end_date)
        if not parquet_files:
            raise ValueError(f"No data found for {start_date} to {end_date}")

        # Load and combine all files
        import pandas as pd

        dataframes = []
        for file_path, year, month in parquet_files:
            df = data_loader.load_parquet_file(file_path, year, month)
            if df is not None:
                dataframes.append(df)

        if not dataframes:
            raise ValueError(
                f"No valid data files found for {start_date} to {end_date}"
            )

        # Combine all data
        df = pd.concat(dataframes, ignore_index=True)

        # Convert to TimeSeriesDataFrame
        ts_df = data_loader.convert_to_timeseries_dataframe(df, full_config)
        if ts_df is None:
            raise ValueError("Failed to convert data to TimeSeriesDataFrame")

        # Get data summary
        summary = data_loader.get_data_stats(df)
        logger.info(f"Data summary: {json.dumps(summary, indent=2, default=str)}")

        # Initialize trainer
        if use_covariates:
            trainer = CovariateTrainer(full_config)
            logger.info("Training Chronos model with covariate integration...")
        else:
            trainer = ChronosTrainer(full_config)
            logger.info("Training Chronos model...")

        # Train model
        model_save_path = trainer.train_model(ts_df, model_path)

        logger.info(f"Training completed successfully! Model saved to: {model_save_path}")

    except (ValueError, ConfigValidationError) as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    help="Training configuration YAML file",
)
@click.option(
    "--start-date",
    help="Start date for training data (YYYY-MM-DD). Can also be specified in config.",
)
@click.option(
    "--end-date",
    help="End date for training data (YYYY-MM-DD). Can also be specified in config.",
)
@click.option(
    "--checkpoint-dir",
    help="Directory to store checkpoints. Can also be specified in config.",
)
@click.option(
    "--validation-start-date",
    help="Start date for validation data (YYYY-MM-DD)",
)
@click.option(
    "--validation-end-date",
    help="End date for validation data (YYYY-MM-DD)",
)
@click.option(
    "--model-path",
    help="Path to save trained model (defaults to config value)",
)
@click.pass_context
def train_incremental(
    ctx,
    config_file,
    start_date,
    end_date,
    checkpoint_dir,
    validation_start_date,
    validation_end_date,
    model_path,
):
    """Train a Chronos model incrementally with checkpoint support"""

    logger = ctx.obj["logger"]

    try:
        logger.info("Starting incremental model training")

        # Get configuration
        config_provider = ctx.obj["config_provider"]
        if config_provider is None:
            if not config_file:
                raise ConfigValidationError(
                    "Configuration required. Specify --config-file or use --config in CLI base command"
                )
            config_provider = CentralConfigProvider([config_file])
            config_provider.initialize()

        # Get merged config with incremental training
        full_config = config_provider.get_merged_config(
            include_covariates=True, include_incremental=True
        )

        # Override model path if provided
        if model_path:
            full_config["model_path"] = model_path

        # Get date ranges from arguments or config
        if not start_date:
            start_date = full_config.get("start_date")
        if not end_date:
            end_date = full_config.get("end_date")
        if not checkpoint_dir:
            checkpoint_dir = full_config.get("checkpoint_dir")
            if not checkpoint_dir:
                # Get from incremental config
                inc_config = full_config.get("incremental_training", {})
                checkpoint_dir = inc_config.get("checkpoint_dir")

        # Validate required parameters
        if not start_date or not end_date:
            raise ValueError(
                "Start date and end date must be provided either via --start-date/--end-date "
                "or in the configuration file"
            )
        if not checkpoint_dir:
            raise ValueError(
                "Checkpoint directory must be provided either via --checkpoint-dir "
                "or in the configuration file"
            )

        # Set default validation dates if not provided
        if not validation_start_date:
            validation_start_date = start_date
        if not validation_end_date:
            validation_end_date = end_date

        # Initialize incremental trainer
        trainer = IncrementalTrainer(full_config)

        # Train with checkpoints
        result = trainer.train_with_checkpoints(
            start_date=start_date,
            end_date=end_date,
            validation_start_date=validation_start_date,
            validation_end_date=validation_end_date,
            checkpoint_dir=checkpoint_dir,
            previous_model_path=None,
        )

        if result.get("status") == "completed":
            logger.info("Incremental training completed successfully!")
            logger.info(f"Checkpoint directory: {result.get('checkpoint_dir')}")
        else:
            logger.warning(f"Training completed with status: {result.get('status')}")

    except (ValueError, ConfigValidationError) as e:
        logger.error(f"Incremental training failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during incremental training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    cli(obj={})

