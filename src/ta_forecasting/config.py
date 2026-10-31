"""Configuration loader and validation.

Loads parameters from configs/config.yaml using Pydantic for validation.
No magic numbers should exist in code — everything comes from config.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field, model_validator

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


class PrepColumnsConfig(BaseModel):
    """Column names of the interim datasets used by the preparation step."""

    date: str = "Date"
    series_id: str = "ID"
    value: str = "Index value ($)"
    series_type: str = "Series type"
    lookup_id: str = "Series ID"
    concept: str = "Concept"
    region: str = "Region"


class PrepConfig(BaseModel):
    """Phase 3 data preparation settings."""

    source_dataset: str = "Data"
    lookup_dataset: str = "ID_lookup"
    target_series_type: str = "Cost Index"
    columns: PrepColumnsConfig = Field(default_factory=PrepColumnsConfig)
    target_column: str = "target"
    quarter_period: int = 4
    max_encoder_length: int = 16
    min_encoder_length: int = 16
    max_prediction_length: int = 4
    normalizer_method: str = "standard"
    panel_file: str = "panel.parquet"
    scaler_file: str = "scaler.json"
    summary_file: str = "split_summary.json"


class MetricsConfig(BaseModel):
    """Evaluation metrics."""

    primary: str = "mape"
    secondary: list[str] = Field(default_factory=lambda: ["rmse", "mae", "smape"])


class EvaluationConfig(BaseModel):
    """Phase 5 rolling-origin evaluation settings."""

    splits: list[str] = Field(default_factory=lambda: ["val", "test"])
    main_models: list[str] = Field(default_factory=lambda: ["ets", "arima", "tft"])
    baselines: list[Literal["naive", "seasonal_naive"]] = Field(
        default_factory=lambda: ["naive", "seasonal_naive"]
    )
    sensitivity_models: list[Literal["ets", "arima"]] = Field(
        default_factory=lambda: ["ets", "arima"]
    )
    run_sensitivity: bool = True
    n_jobs: int = 1
    percentiles: list[float] = Field(default_factory=lambda: [10, 25, 50, 75, 90])
    dm_split: str = "test"
    dm_loss: Literal["ape", "ae", "se"] = "ape"
    dm_alpha: float = Field(0.05, gt=0, lt=1)
    dm_fdr_method: str = "fdr_bh"
    output_dir: str = "outputs/eval"
    predictions_file: str = "predictions.parquet"
    per_series_file: str = "per_series_metrics.csv"
    report_prefix: str = "phase5"


class RollingConfig(BaseModel):
    """Rolling-origin forecasting protocol shared by all models."""

    refit_statistical: bool = False


class ETSConfig(BaseModel):
    """ETS model hyperparameters."""

    seasonal_periods: int = 4
    trend: str | None = "add"
    seasonal: str | None = "add"
    damped_trend: bool = True
    initialization_method: str = "estimated"
    fallback_seasonal: str | None = None


class ARIMAConfig(BaseModel):
    """ARIMA order-search bounds and fitting options."""

    max_p: int = Field(3, ge=0)
    max_d: int = Field(2, ge=0)
    max_q: int = Field(3, ge=0)
    max_P: int = Field(1, ge=0)
    max_D: int = Field(1, ge=0)
    max_Q: int = Field(1, ge=0)
    seasonal_period: int = Field(4, ge=2)
    kpss_alpha: float = Field(0.05, gt=0, lt=1)
    kpss_regression: Literal["c", "ct"] = "c"
    kpss_nlags: str | int = "auto"
    seasonal_strength_threshold: float = Field(0.64, ge=0, le=1)
    information_criterion: Literal["aic", "aicc", "bic"] = "aic"
    constant_max_total_diff: int = Field(1, ge=0)
    fit_method: str = "lbfgs"
    maxiter: int = Field(200, ge=1)
    enforce_stationarity: bool = True
    enforce_invertibility: bool = True
    n_jobs: int = 1


class TFTConfig(BaseModel):
    """TFT model hyperparameters and trainer settings."""

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
    quantiles: list[float] = Field(
        default_factory=lambda: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    )
    point_quantile: float = 0.5
    monitor: str = "val_loss"
    accelerator: Literal["auto", "cpu", "gpu", "cuda", "mps"] = "auto"
    devices: int = 1
    num_threads: int = Field(4, ge=1)
    num_workers: int = Field(0, ge=0)
    deterministic: bool = True
    log_every_n_steps: int = Field(10, ge=1)

    @model_validator(mode="after")
    def _point_quantile_is_listed(self) -> TFTConfig:
        if self.point_quantile not in self.quantiles:
            raise ValueError("tft.point_quantile must be one of tft.quantiles.")
        return self


class ReportFigureConfig(BaseModel):
    """Figure settings of the Phase 6 report."""

    dpi: int = Field(300, ge=72)
    size: tuple[float, float] = (8.0, 4.5)
    font_size: float = Field(10, gt=0)
    heatmap_cmap: str = "RdYlGn"
    heatmap_bounds: list[float] = Field(
        default_factory=lambda: [0.0, 0.001, 0.01, 0.05, 0.1, 1.0]
    )

    @model_validator(mode="after")
    def _bounds_increase(self) -> ReportFigureConfig:
        b = self.heatmap_bounds
        if len(b) < 2 or any(lo >= hi for lo, hi in zip(b, b[1:], strict=False)):
            raise ValueError("report.figure.heatmap_bounds must be strictly increasing.")
        return self


class ReportFigureFiles(BaseModel):
    """File names of the Phase 6 figures."""

    overall: str = "fig_mape_per_model.png"
    horizon: str = "fig_mape_per_horizon.png"
    boxplot: str = "fig_mape_per_series_boxplot.png"
    dm_heatmap: str = "fig_dm_pvalue_heatmap.png"


class ReportConfig(BaseModel):
    """Phase 6 final report settings."""

    output_file: str = "FINAL_REPORT.md"
    figures_dir: str = "outputs/report"
    split: str = "test"
    main_mode: str = "main"
    sensitivity_mode: str = "refit"
    model_order: list[str] = Field(
        default_factory=lambda: ["ets", "arima", "tft", "naive", "seasonal_naive"]
    )
    model_labels: dict[str, str] = Field(default_factory=dict)
    split_labels: dict[str, str] = Field(default_factory=dict)
    metric_labels: dict[str, str] = Field(default_factory=dict)
    decimals: int = Field(2, ge=0)
    p_value_decimals: int = Field(4, ge=1)
    decimal_separator: str = ","
    thousands_separator: str = "."
    forbidden_columns: list[str] = Field(
        default_factory=lambda: ["ID", "actual", "prediction", "time_idx", "origin"]
    )
    figure: ReportFigureConfig = Field(default_factory=ReportFigureConfig)
    figure_files: ReportFigureFiles = Field(default_factory=ReportFigureFiles)


class ForecastConfig(BaseModel):
    """Phase 7a production forecasts and long-horizon backtest."""

    models: list[Literal["ets", "arima", "tft", "naive", "seasonal_naive"]] = Field(
        default_factory=lambda: ["ets", "arima", "tft", "naive"]
    )
    max_horizon_models: list[str] = Field(default_factory=lambda: ["tft"])
    end_quarter: str = "2031Q4"
    backtest_first_split: str = "val"
    calibration_split: str = "val"
    coverage_split: str = "test"
    interval_level: float = Field(0.8, gt=0, lt=1)
    flat_tolerance: float = Field(1e-9, ge=0)
    composition_note_drop: float = Field(0.1, ge=0, lt=1)
    n_jobs: int = 1
    forecast_file: str = "forecast.parquet"
    summary_file: str = "forecast_summary.parquet"
    report_prefix: str = "phase7"


class OutputConfig(BaseModel):
    """Paths to output directories and model artefact file names."""

    models_dir: str = "models"
    outputs_dir: str = "outputs"
    reports_dir: str = "reports"
    eda_dir: str = "outputs/eda"
    logs_dir: str = "outputs/logs"
    ets_file: str = "ets.pkl"
    arima_file: str = "arima.pkl"
    arima_orders_file: str = "arima_orders.json"
    tft_file: str = "tft.ckpt"
    training_summary_file: str = "training_summary.json"


class Config(BaseModel):
    """Root configuration container."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    prep: PrepConfig = Field(default_factory=PrepConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    rolling: RollingConfig = Field(default_factory=RollingConfig)
    ets: ETSConfig = Field(default_factory=ETSConfig)
    arima: ARIMAConfig = Field(default_factory=ARIMAConfig)
    tft: TFTConfig = Field(default_factory=TFTConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
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
