"""Thesis-ready figures drawn from the Phase 5 aggregate tables only.

The per-series boxplot is built from the stored percentiles (``ax.bxp``), so no per-series
value is needed: box = p25–p75, line = median, whiskers = lowest/highest stored
percentile (p10/p90 by default), marker = mean.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from ta_forecasting.config import Config  # noqa: E402
from ta_forecasting.data.prepare import SPLIT_COLUMN  # noqa: E402
from ta_forecasting.evaluation.backtest import MODE, MODEL  # noqa: E402
from ta_forecasting.models.base import STEP  # noqa: E402
from ta_forecasting.reporting.formatting import Formatter  # noqa: E402
from ta_forecasting.reporting.inputs import (  # noqa: E402
    PAIR_A,
    PAIR_B,
    ReportInputs,
    all_models,
    model_pairs,
)

HORIZON_LABEL = "Horizon (kuartal ke depan)"


def make_figures(inputs: ReportInputs, cfg: Config) -> dict[str, Path]:
    """Write all figures; returns their paths by key (see ``report.figure_files``)."""
    out = Path(cfg.report.figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = cfg.report.figure_files
    fmt = Formatter(cfg)
    plt.rcParams.update({"font.size": cfg.report.figure.font_size})
    return {
        "overall": plot_overall(inputs, cfg, fmt, out / files.overall),
        "horizon": plot_horizon(inputs, cfg, fmt, out / files.horizon),
        "boxplot": plot_boxplot(inputs, cfg, fmt, out / files.boxplot),
        "dm_heatmap": plot_dm_heatmap(inputs, cfg, fmt, out / files.dm_heatmap),
    }


def plot_overall(inputs: ReportInputs, cfg: Config, fmt: Formatter, path: Path) -> Path:
    """Grouped bars of the primary metric per model, one bar per split."""
    primary, r = cfg.metrics.primary, cfg.report
    models = all_models(cfg)
    overall = inputs.tables["overall"]
    data = overall[overall[MODE] == r.main_mode].pivot(index=MODEL, columns=SPLIT_COLUMN,
                                                       values=primary)
    data = data.loc[models, cfg.evaluation.splits]
    fig, ax = plt.subplots(figsize=r.figure.size)
    x = np.arange(len(models))
    width = 0.8 / len(data.columns)
    for i, split in enumerate(data.columns):
        bars = ax.bar(x + (i - (len(data.columns) - 1) / 2) * width, data[split], width,
                      label=fmt.split(split))
        ax.bar_label(bars, labels=[fmt.num(v) for v in data[split]], fontsize="small",
                     padding=2)
    ax.set_xticks(x, [fmt.model(m) for m in models])
    ax.set_xlabel("Model")
    ax.set_ylabel(fmt.metric(primary))
    ax.set_title(f"{fmt.metric_short(primary)} per model (rolling origin)")
    ax.legend(title="Split")
    ax.margins(y=0.15)
    _decimal_axis(ax, cfg)
    return _save(fig, path, cfg)


def plot_horizon(inputs: ReportInputs, cfg: Config, fmt: Formatter, path: Path) -> Path:
    """Primary metric against the forecast horizon on the headline split."""
    primary, r = cfg.metrics.primary, cfg.report
    step = inputs.tables["per_horizon"]
    part = step[(step[MODE] == r.main_mode) & (step[SPLIT_COLUMN] == r.split)]
    data = part.pivot(index=STEP, columns=MODEL, values=primary)
    fig, ax = plt.subplots(figsize=r.figure.size)
    for model in all_models(cfg):
        ax.plot(data.index, data[model], marker="o", label=fmt.model(model))
    ax.set_xticks(data.index)
    ax.set_xlabel(HORIZON_LABEL)
    ax.set_ylabel(fmt.metric(primary))
    ax.set_title(f"{fmt.metric_short(primary)} per horizon, data {fmt.split(r.split).lower()}")
    ax.legend()
    ax.grid(alpha=0.3)
    _decimal_axis(ax, cfg)
    return _save(fig, path, cfg)


def plot_boxplot(inputs: ReportInputs, cfg: Config, fmt: Formatter, path: Path) -> Path:
    """Per-series distribution of the primary metric, drawn from stored percentiles."""
    primary, r = cfg.metrics.primary, cfg.report
    pct = sorted(cfg.evaluation.percentiles)
    dist = inputs.tables["per_series_distribution"]
    part = dist[(dist[MODE] == r.main_mode) & (dist[SPLIT_COLUMN] == r.split)
                & (dist["metric"] == primary)].set_index(MODEL)
    models = all_models(cfg)
    stats = [
        {
            "label": fmt.model(m),
            "whislo": part.at[m, f"p{pct[0]:g}"],
            "q1": part.at[m, f"p{_nearest(pct, 25):g}"],
            "med": part.at[m, f"p{_nearest(pct, 50):g}"],
            "q3": part.at[m, f"p{_nearest(pct, 75):g}"],
            "whishi": part.at[m, f"p{pct[-1]:g}"],
            "mean": part.at[m, "mean"],
            "fliers": [],
        }
        for m in models
    ]
    fig, ax = plt.subplots(figsize=r.figure.size)
    ax.bxp(stats, showmeans=True, showfliers=False)
    ax.set_xlabel("Model")
    ax.set_ylabel(f"{fmt.metric(primary)} per seri")
    ax.set_title(
        f"Sebaran {fmt.metric_short(primary)} per seri, data {fmt.split(r.split).lower()} "
        f"(kotak p{_nearest(pct, 25):g}–p{_nearest(pct, 75):g}, "
        f"whisker p{pct[0]:g}–p{pct[-1]:g}, ▲ rata-rata)"
    )
    ax.grid(alpha=0.3, axis="y")
    _decimal_axis(ax, cfg)
    return _save(fig, path, cfg)


def plot_dm_heatmap(inputs: ReportInputs, cfg: Config, fmt: Formatter, path: Path) -> Path:
    """p-values of the panel Diebold-Mariano test, model pair × horizon."""
    f = cfg.report.figure
    dm = inputs.tables["dm_panel"].set_index([PAIR_A, PAIR_B, STEP])
    pairs = model_pairs(cfg)
    steps = list(range(1, cfg.prep.max_prediction_length + 1))
    values = np.array([[dm.at[(a, b, h), "p_value"] for h in steps] for a, b in pairs])
    norm = BoundaryNorm(f.heatmap_bounds, ncolors=256)
    fig, ax = plt.subplots(figsize=f.size)
    image = ax.imshow(values, cmap=f.heatmap_cmap, norm=norm, aspect="auto")
    for i, (a, b) in enumerate(pairs):
        for j, h in enumerate(steps):
            row = dm.loc[(a, b, h)]
            better = fmt.model(row["better"]) if bool(row["significant"]) else "n.s."
            ax.text(j, i, f"{fmt.pval(row['p_value'])}\n{better}", ha="center", va="center",
                    fontsize="small")
    ax.set_xticks(range(len(steps)), [str(h) for h in steps])
    ax.set_yticks(range(len(pairs)), [f"{fmt.model(a)} vs {fmt.model(b)}" for a, b in pairs])
    ax.set_xlabel(HORIZON_LABEL)
    ax.set_ylabel("Pasangan model")
    ax.set_title(
        f"p-value uji Diebold-Mariano (HLN), data {fmt.split(cfg.evaluation.dm_split).lower()}, "
        f"α = {fmt.num(cfg.evaluation.dm_alpha)}"
    )
    bar = fig.colorbar(image, ax=ax, ticks=f.heatmap_bounds)
    bar.ax.set_yticklabels([fmt.num(b, cfg.report.p_value_decimals) for b in f.heatmap_bounds])
    bar.set_label("p-value")
    return _save(fig, path, cfg)


def _nearest(values: list[float], target: float) -> float:
    return min(values, key=lambda v: abs(v - target))


def _decimal_axis(ax: plt.Axes, cfg: Config) -> None:
    """Y tick labels with the configured decimal separator."""
    sep = cfg.report.decimal_separator
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}".replace(".", sep)))


def _save(fig: plt.Figure, path: Path, cfg: Config) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=cfg.report.figure.dpi)
    plt.close(fig)
    return path
