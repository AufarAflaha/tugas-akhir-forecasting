"""Configuration loader and validation.

Loads parameters from configs/config.yaml using Pydantic for validation.
No magic numbers should exist in code — everything comes from config.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic models for config sections
# ---------------------------------------------------------------------------

class ProjectConfig(BaseModel):
    """Top-level project settings."""

    name: str = "ta-forecasting"
    seed: int = 42
    last_actual_quarter: str = "2026Q1"


class DataConfig(BaseModel):
    """Paths to data directories and the raw file."""

    raw_dir: str = "data/raw"
    raw_file: str = "Market_Segment_Cost_Models_-_Q2_2026.xlsx"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    db_dir: str = "data/db"

    @property
    def raw_path(self) -> Path:
        return Path(self.raw_dir) / self.raw_file


class ValidationConfig(BaseModel):
    """Train/val/test split strategy."""

    strategy: str = "temporal_split"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


class MetricsConfig(BaseModel):
    """Evaluation metrics."""

    primary: str = "mape"
    secondary: list[str] = Field(default_factory=lambda: ["rmse", "mae", "smape"])


class ETSConfig(BaseModel):
    """ETS model hyperparameters."""

    seasonal_periods: int = 4
    trend: str = "add"
    seasonal: str = "add"
    damped_trend: bool = True


class ARIMAConfig(BaseModel):
    """ARIMA model hyperparameters."""

    max_p: int = 5
    max_d: int = 2
    max_q: int = 5
    seasonal_order_P: int = 2
    seasonal_order_D: int = 1
    seasonal_order_Q: int = 2
    seasonal_period: int = 4


class TFTConfig(BaseModel):
    """TFT model hyperparameters."""

    hidden_size: int = 64
    attention_head_size: int = 4
    dropout: float = 0.1
    hidden_continuous_size: int = 16
    max_epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 64
    gradient_clip_val: float = 0.1
    early_stop_patience: int = 10
    reduce_lr_patience: int = 5


class OutputConfig(BaseModel):
    """Paths to output directories."""

    models_dir: str = "models"
    outputs_dir: str = "outputs"
    reports_dir: str = "reports"
    eda_dir: str = "outputs/eda"


class Config(BaseModel):
    """Root configuration container."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    ets: ETSConfig = Field(default_factory=ETSConfig)
    arima: ARIMAConfig = Field(default_factory=ARIMAConfig)
    tft: TFTConfig = Field(default_factory=TFTConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path = "configs/config.yaml") -> Config:
    """Load and validate configuration from a YAML file.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    Config
        Validated configuration object.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    return Config(**raw)


def set_global_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries.

    Parameters
    ----------
    seed : int
        The random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch not installed yet — fine during early phases
