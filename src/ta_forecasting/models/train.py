"""Phase 4 — train ETS, ARIMA and TFT on the processed panel and write the artefacts.

* ETS and ARIMA are fitted on train rows only.
* TFT is fitted on train rows; validation rows drive early stopping and checkpointing.
* Test rows are never read here. Quarters after ``last_actual_quarter`` are rejected.

``models/training_summary.json`` only holds aggregate information (times, counts, order
distribution, epochs, best validation loss) and is updated per model, so training one
model keeps the entries of the others.
"""

from __future__ import annotations

import json
import logging
import platform
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

from ta_forecasting.config import Config
from ta_forecasting.data.prepare import SPLIT_COLUMN, TIME_IDX, TRAIN, VAL
from ta_forecasting.models.arima import ARIMAForecaster
from ta_forecasting.models.base import Forecaster, check_no_projection
from ta_forecasting.models.ets import ETSForecaster
from ta_forecasting.models.tft import TFTForecaster

logger = logging.getLogger(__name__)

MODELS: dict[str, type[Forecaster]] = {
    ETSForecaster.name: ETSForecaster,
    ARIMAForecaster.name: ARIMAForecaster,
    TFTForecaster.name: TFTForecaster,
}
ALL = "all"
PACKAGES = ("numpy", "pandas", "statsmodels", "torch", "lightning", "pytorch-forecasting")


def artefact_path(name: str, cfg: Config) -> Path:
    """Path of the saved model ``name`` inside ``output.models_dir``."""
    o = cfg.output
    files = {"ets": o.ets_file, "arima": o.arima_file, "tft": o.tft_file}
    return Path(o.models_dir) / files[name]


def resolve_names(model: str) -> list[str]:
    """Expand ``all`` and validate a model name."""
    if model == ALL:
        return list(MODELS)
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; choose from {[*MODELS, ALL]}.")
    return [model]


def load_train_val(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the processed panel and return its train and validation rows only."""
    panel = pd.read_parquet(Path(cfg.data.processed_dir) / cfg.prep.panel_file)
    check_no_projection(panel, cfg)
    train = panel[panel[SPLIT_COLUMN] == TRAIN].reset_index(drop=True)
    val = panel[panel[SPLIT_COLUMN] == VAL].reset_index(drop=True)
    if train.empty or val.empty:
        raise ValueError("Processed panel has no train or validation rows; run `prep` first.")
    if train[TIME_IDX].max() >= val[TIME_IDX].min():
        raise ValueError("Train rows must precede validation rows.")
    return train, val


def _environment() -> dict[str, str]:
    env = {"python": platform.python_version(), "machine": platform.machine()}
    for pkg in PACKAGES:
        try:
            env[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            env[pkg] = "not installed"
    return env


def _data_scope(train: pd.DataFrame, val: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    sid = cfg.prep.columns.series_id
    return {
        "n_series": int(train[sid].nunique()),
        "train_time_idx": [int(train[TIME_IDX].min()), int(train[TIME_IDX].max())],
        "val_time_idx": [int(val[TIME_IDX].min()), int(val[TIME_IDX].max())],
        "last_actual_quarter": cfg.project.last_actual_quarter,
        "test_rows_used": False,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def train_models(cfg: Config, names: list[str]) -> dict[str, Any]:
    """Fit, save and summarise the requested models. Returns the updated summary."""
    train, val = load_train_val(cfg)
    models_dir = Path(cfg.output.models_dir)
    summary_path = models_dir / cfg.output.training_summary_file
    summary: dict[str, Any] = (
        json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    )
    summary.setdefault("models", {})

    for name in names:
        logger.info("Training %s", name)
        model = MODELS[name](cfg)
        if name == TFTForecaster.name:
            model.fit(train, val)
        else:
            model.fit(train)
        path = artefact_path(name, cfg)
        model.save(path)
        artefacts = [str(path)]
        if isinstance(model, ARIMAForecaster):
            orders_path = models_dir / cfg.output.arima_orders_file
            _write_json(orders_path, model.order_table())
            artefacts.append(str(orders_path))
        summary["models"][name] = {
            **model.fit_info,
            "artefacts": artefacts,
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        logger.info("Saved %s to %s", name, ", ".join(artefacts))

    summary["seed"] = cfg.project.seed
    summary["rolling_refit_statistical"] = cfg.rolling.refit_statistical
    summary["data_scope"] = _data_scope(train, val, cfg)
    summary["environment"] = _environment()
    _write_json(summary_path, summary)
    return summary


def setup_logging(cfg: Config, filename: str) -> Path:
    """Log INFO to the console and to ``output.logs_dir/filename`` (no data values)."""
    logs_dir = Path(cfg.output.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / filename
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for old in [h for h in root.handlers if getattr(h, "_ta_forecasting", False)]:
        root.removeHandler(old)
        old.close()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for handler in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(fmt)
        handler._ta_forecasting = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    return log_path
