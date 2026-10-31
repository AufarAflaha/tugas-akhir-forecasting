"""TFT — one global Temporal Fusion Transformer (pytorch-forecasting + Lightning).

* Datasets come from :mod:`ta_forecasting.data.prepare` (encoder 16, horizon 4, normalizers
  fitted on train rows only). Validation windows drive early stopping, checkpointing and
  ``ReduceLROnPlateau``.
* Loss: ``QuantileLoss``; the ``point_quantile`` (median) is the point forecast.
* Seed from ``project.seed`` (``seed_everything(workers=True)``), ``deterministic=True``,
  ``num_workers`` from config (0).
* The accelerator is resolved from ``tft.accelerator`` with a safe fallback to CPU.

``forecast(history, horizon)`` uses the last ``max_encoder_length`` quarters of each series
as encoder and appends ``max_prediction_length`` calendar rows (known covariates only). The
target in those rows is a placeholder: the TFT never reads decoder targets as input.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd

from ta_forecasting.config import Config
from ta_forecasting.data.prepare import (
    QUARTER,
    QUARTER_COS,
    QUARTER_SIN,
    TIME_IDX,
    as_tft_frame,
    build_evaluation_dataset,
    build_training_dataset,
    quarter_encoding,
    static_categoricals,
)
from ta_forecasting.models.base import (
    Forecaster,
    check_no_projection,
    forecast_frame,
    split_series,
    validate_horizon,
)

logger = logging.getLogger(__name__)

CHECKPOINT_EXTRA_KEY = "ta_forecasting"


def resolve_accelerator(requested: str) -> str:
    """Map the configured accelerator to one that is available, falling back to CPU."""
    import torch

    cuda = torch.cuda.is_available()
    mps_backend = getattr(torch.backends, "mps", None)
    mps = bool(mps_backend is not None and mps_backend.is_available())
    if requested == "cpu":
        return "cpu"
    if requested == "auto":
        return "gpu" if cuda else ("mps" if mps else "cpu")
    if requested in ("gpu", "cuda") and cuda:
        return "gpu"
    if requested == "mps" and mps:
        return "mps"
    logger.warning("Accelerator %r not available; falling back to CPU.", requested)
    return "cpu"


def build_predict_frame(history: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Encoder tail of every series plus future rows holding only known covariates.

    The placeholder target of the future rows is the last observed value. It never reaches
    the network as input; it only satisfies the dataset's non-null requirement.
    """
    p = cfg.prep
    sid = p.columns.series_id
    encoder_len = p.max_encoder_length
    pred_len = p.max_prediction_length
    statics = static_categoricals(cfg)

    ordered = history.sort_values([sid, TIME_IDX])
    tail = ordered.groupby(sid, sort=True).tail(encoder_len)
    short = tail.groupby(sid).size()
    if (short < encoder_len).any():
        raise ValueError(f"Every series needs at least {encoder_len} quarters of history.")

    last = tail.groupby(sid, sort=True).tail(1).reset_index(drop=True)
    future_parts = []
    for step in range(1, pred_len + 1):
        part = last[[sid, *[c for c in statics if c != sid]]].copy()
        part[TIME_IDX] = last[TIME_IDX] + step
        part[QUARTER] = (last[QUARTER] - 1 + step) % p.quarter_period + 1
        part[QUARTER_SIN], part[QUARTER_COS] = quarter_encoding(part[QUARTER], p.quarter_period)
        part[p.target_column] = last[p.target_column]
        future_parts.append(part)
    columns = [*statics, TIME_IDX, QUARTER, QUARTER_SIN, QUARTER_COS, p.target_column]
    frame = pd.concat([tail[columns], *future_parts], ignore_index=True)
    return as_tft_frame(frame.sort_values([sid, TIME_IDX]).reset_index(drop=True), cfg)


class TFTForecaster(Forecaster):
    """Global TFT behind the common forecaster interface."""

    name: ClassVar[str] = "tft"

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.model: Any = None
        self._checkpoint: dict[str, Any] | None = None
        self._train_tail: pd.DataFrame | None = None

    # -- training ----------------------------------------------------------------------

    def _build_model(self, training: Any) -> Any:
        from pytorch_forecasting import TemporalFusionTransformer
        from pytorch_forecasting.metrics import QuantileLoss

        t = self.cfg.tft
        return TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=t.learning_rate,
            hidden_size=t.hidden_size,
            attention_head_size=t.attention_head_size,
            dropout=t.dropout,
            hidden_continuous_size=t.hidden_continuous_size,
            loss=QuantileLoss(quantiles=list(t.quantiles)),
            reduce_on_plateau_patience=t.reduce_lr_patience,
            log_interval=-1,
        )

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> Self:
        """Train on ``train``; ``val`` (required) drives early stopping and checkpointing."""
        import lightning.pytorch as pl
        import torch
        from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
        from lightning.pytorch.loggers import CSVLogger
        from pytorch_forecasting import TemporalFusionTransformer

        if val is None or val.empty:
            raise ValueError("TFT needs validation rows for early stopping.")
        check_no_projection(train, self.cfg)
        check_no_projection(val, self.cfg)
        val_first, val_last = int(val[TIME_IDX].min()), int(val[TIME_IDX].max())
        if int(train[TIME_IDX].max()) >= val_first:
            raise ValueError("Validation rows must come strictly after the training rows.")

        t = self.cfg.tft
        pl.seed_everything(self.cfg.project.seed, workers=True)
        torch.set_num_threads(t.num_threads)
        accelerator = resolve_accelerator(t.accelerator)

        training = build_training_dataset(train, self.cfg)
        validation = build_evaluation_dataset(
            training, pd.concat([train, val], ignore_index=True), val_first, val_last, self.cfg
        )
        train_loader = training.to_dataloader(
            train=True, batch_size=t.batch_size, num_workers=t.num_workers
        )
        val_loader = validation.to_dataloader(
            train=False, batch_size=t.batch_size, num_workers=t.num_workers
        )
        model = self._build_model(training)

        logs_dir = Path(self.cfg.output.logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="tft_ckpt_") as ckpt_dir:
            early_stop = EarlyStopping(
                monitor=t.monitor, patience=t.early_stop_patience, mode="min"
            )
            checkpoint = ModelCheckpoint(
                dirpath=ckpt_dir, filename="best", monitor=t.monitor, mode="min", save_top_k=1
            )
            trainer = pl.Trainer(
                max_epochs=t.max_epochs,
                accelerator=accelerator,
                devices=t.devices,
                gradient_clip_val=t.gradient_clip_val,
                deterministic=t.deterministic,
                callbacks=[early_stop, LearningRateMonitor(logging_interval="epoch"), checkpoint],
                logger=CSVLogger(save_dir=str(logs_dir), name="tft"),
                log_every_n_steps=t.log_every_n_steps,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            start = time.perf_counter()
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            seconds = time.perf_counter() - start

            best_path = checkpoint.best_model_path
            if not best_path:
                raise RuntimeError("TFT training produced no checkpoint.")
            self._checkpoint = torch.load(best_path, map_location="cpu")
            self.model = TemporalFusionTransformer.load_from_checkpoint(
                best_path, map_location="cpu"
            )
        self.model.eval()

        epochs_run = int(trainer.current_epoch)
        self.fit_info = {
            "fit_seconds": round(seconds, 3),
            "epochs_run": epochs_run,
            "best_epoch": int(self._checkpoint["epoch"]) + 1,
            "best_val_loss": float(checkpoint.best_model_score),
            "stopped_early": bool(early_stop.stopped_epoch > 0),
            "max_epochs": t.max_epochs,
            "final_learning_rate": float(trainer.optimizers[0].param_groups[0]["lr"]),
            "accelerator": accelerator,
            "num_threads": t.num_threads,
            "n_parameters": int(sum(p.numel() for p in self.model.parameters())),
            "n_train_windows": len(training),
            "n_val_windows": len(validation),
            "n_series": int(train[self.cfg.prep.columns.series_id].nunique()),
            "loss": "QuantileLoss",
            "quantiles": list(t.quantiles),
            "point_quantile": t.point_quantile,
        }
        self._train_tail = self._encoder_tail(train)
        logger.info(
            "TFT trained %d epochs in %.1f s (best epoch %d)",
            epochs_run, seconds, self.fit_info["best_epoch"],
        )  # fmt: skip
        return self

    def _encoder_tail(self, df: pd.DataFrame) -> pd.DataFrame:
        sid = self.cfg.prep.columns.series_id
        return (
            df.sort_values([sid, TIME_IDX])
            .groupby(sid, sort=True)
            .tail(self.cfg.prep.max_encoder_length)
            .reset_index(drop=True)
        )

    # -- inference ---------------------------------------------------------------------

    def predict(self, horizon: int) -> pd.DataFrame:
        """Forecast ``horizon`` quarters after the end of the training data."""
        self._require_fitted()
        assert self._train_tail is not None
        return self.forecast(self._train_tail, horizon)

    def forecast(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Median forecast of ``horizon`` quarters after the end of ``history``."""
        from pytorch_forecasting import TimeSeriesDataSet

        self._require_fitted()
        horizon = validate_horizon(horizon)
        if horizon > self.cfg.prep.max_prediction_length:
            raise ValueError(
                f"TFT was trained for horizon {self.cfg.prep.max_prediction_length}; "
                f"got {horizon}."
            )
        series = split_series(history, self.cfg)
        ends = {h.end for h in series.values()}
        if len(ends) != 1:
            raise ValueError("All series in the history must end at the same origin.")

        frame = build_predict_frame(history, self.cfg)
        dataset = TimeSeriesDataSet.from_parameters(
            self.model.dataset_parameters, frame, predict=True, stop_randomization=True
        )
        values, index = self.predict_windows(dataset)
        sid = self.cfg.prep.columns.series_id
        forecasts = {
            str(key): values[i, :horizon]
            for i, key in enumerate(index[sid].astype(str).to_numpy())
        }
        origins = {key: hist.end + 1 for key, hist in series.items()}
        if set(forecasts) != set(origins):
            raise RuntimeError("TFT prediction index does not match the history series.")
        return forecast_frame(forecasts, origins, self.cfg)

    def predict_windows(self, dataset: Any) -> tuple[np.ndarray, pd.DataFrame]:
        """Point (``point_quantile``) forecast for every window of ``dataset``.

        Returns an array of shape (n_windows, max_prediction_length) and the window index
        (series id and first predicted ``time_idx``) in the same row order.
        """
        self._require_fitted()
        t = self.cfg.tft
        out = self.model.predict(
            dataset,
            mode="quantiles",
            return_index=True,
            batch_size=t.batch_size,
            num_workers=t.num_workers,
            trainer_kwargs={
                "accelerator": resolve_accelerator(t.accelerator),
                "devices": t.devices,
                "logger": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
            },
        )
        q_idx = list(t.quantiles).index(t.point_quantile)
        values = out.output[..., q_idx].detach().cpu().numpy().astype(float)
        return values, out.index.reset_index(drop=True)

    # -- persistence -------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Write the best Lightning checkpoint plus config, fit summary and encoder tail."""
        import torch

        self._require_fitted()
        assert self._checkpoint is not None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self._checkpoint)
        payload[CHECKPOINT_EXTRA_KEY] = {
            "config": self.cfg.model_dump(),
            "fit_info": self.fit_info,
            "train_tail": self._train_tail,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Restore a model written by :meth:`save` (trusted local artefact)."""
        import torch
        from pytorch_forecasting import TemporalFusionTransformer

        checkpoint = torch.load(Path(path), map_location="cpu")
        extra = checkpoint.get(CHECKPOINT_EXTRA_KEY)
        if extra is None:
            raise ValueError(f"{path} is not a checkpoint written by TFTForecaster.save().")
        obj = cls(Config(**extra["config"]))
        obj.model = TemporalFusionTransformer.load_from_checkpoint(Path(path), map_location="cpu")
        obj.model.eval()
        obj._checkpoint = {k: v for k, v in checkpoint.items() if k != CHECKPOINT_EXTRA_KEY}
        obj.fit_info = extra["fit_info"]
        obj._train_tail = extra["train_tail"]
        return obj
