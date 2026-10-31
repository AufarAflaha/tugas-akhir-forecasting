"""Aggregate tables and plots for Phase 5. Nothing here contains raw series values."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ta_forecasting.data.prepare import SPLIT_COLUMN  # noqa: E402
from ta_forecasting.evaluation.backtest import MAIN, MODE, MODEL  # noqa: E402
from ta_forecasting.models.base import STEP  # noqa: E402

TIE = "tie"
FLOAT_DIGITS = 4
FIGSIZE = (8, 4.5)
DPI = 150


def distribution(per_series: pd.DataFrame, names: list[str], pct: list[float]) -> pd.DataFrame:
    """Mean and percentiles of each per-series metric, per (mode, model, split)."""
    rows = []
    for (mode, model, split), group in per_series.groupby([MODE, MODEL, SPLIT_COLUMN]):
        for name in names:
            values = group[name].to_numpy()
            row = {MODE: mode, MODEL: model, SPLIT_COLUMN: split, "metric": name}
            row["mean"] = float(values.mean())
            row.update({f"p{q:g}": float(np.percentile(values, q)) for q in pct})
            row["n_series"] = len(values)
            rows.append(row)
    return pd.DataFrame(rows)


def wins(per_series: pd.DataFrame, models: list[str], metric: str, sid: str) -> pd.DataFrame:
    """Number of series on which each model has the lowest ``metric`` (ties counted apart)."""
    rows = []
    main = per_series[(per_series[MODE] == MAIN) & per_series[MODEL].isin(models)]
    for split, group in main.groupby(SPLIT_COLUMN):
        wide = group.pivot(index=sid, columns=MODEL, values=metric)[models]
        best = wide.min(axis=1)
        is_best = wide.eq(best, axis=0)
        winner = np.where(is_best.sum(axis=1) > 1, TIE, is_best.idxmax(axis=1))
        counts = pd.Series(winner).value_counts()
        row = {SPLIT_COLUMN: split, "metric": metric, "n_series": len(wide)}
        row.update({m: int(counts.get(m, 0)) for m in [*models, TIE]})
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(frame: pd.DataFrame, digits: int = FLOAT_DIGITS) -> str:
    """Render a DataFrame as a GitHub Markdown table (no extra dependency)."""

    def fmt(value: object) -> str:
        if isinstance(value, float | np.floating):
            return "nan" if np.isnan(value) else f"{value:.{digits}f}"
        return str(value)

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def plot_overall(overall: pd.DataFrame, metric: str, path: Path) -> Path:
    """Bar chart of ``metric`` per model and split (main mode)."""
    data = overall[overall[MODE] == MAIN].pivot(index=MODEL, columns=SPLIT_COLUMN, values=metric)
    ax = data.plot(kind="bar", figsize=FIGSIZE, rot=0)
    ax.set_ylabel(metric.upper())
    ax.set_xlabel("")
    ax.set_title(f"{metric.upper()} per model (rolling origin, pooled)")
    return _save(ax, path)


def plot_per_horizon(per_step: pd.DataFrame, metric: str, split: str, path: Path) -> Path:
    """Line chart of ``metric`` against forecast step for one split (main mode)."""
    part = per_step[(per_step[MODE] == MAIN) & (per_step[SPLIT_COLUMN] == split)]
    data = part.pivot(index=STEP, columns=MODEL, values=metric)
    ax = data.plot(marker="o", figsize=FIGSIZE)
    ax.set_xticks(data.index)
    ax.set_xlabel("Horizon (quarters ahead)")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} per horizon — {split}")
    return _save(ax, path)


def plot_series_boxplot(
    per_series: pd.DataFrame, metric: str, split: str, models: list[str], path: Path
) -> Path:
    """Boxplot of the per-series ``metric`` distribution (no series values shown)."""
    part = per_series[(per_series[MODE] == MAIN) & (per_series[SPLIT_COLUMN] == split)]
    present = [m for m in models if m in set(part[MODEL])]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.boxplot([part.loc[part[MODEL] == m, metric].to_numpy() for m in present], showfliers=True)
    ax.set_xticks(range(1, len(present) + 1), present)
    ax.set_ylabel(f"{metric.upper()} per series")
    ax.set_title(f"Per-series {metric.upper()} — {split}")
    return _save(ax, path)


def _save(ax: plt.Axes, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path
