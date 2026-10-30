"""EDA and reporting plots.

All plots saved to disk — never displays raw data values in labels.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

logger = logging.getLogger(__name__)

# Consistent style
sns.set_theme(style="whitegrid", palette="muted")


def _plot_time_series_trends(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Plot quarterly aggregate cost index trend across key series types."""
    req_cols = ("Date", "Index value ($)", "Series type")
    if not all(c in df.columns for c in req_cols):
        return None

    target_types = [
        "Cost Index",
        "Cost Index with Operating Margins (Price Index)",
        "Equipment",
        "Raw Materials",
        "Bulks",
    ]
    df_filtered = df[df["Series type"].isin(target_types)].copy()
    if df_filtered.empty:
        return None

    df_filtered["Date"] = pd.to_datetime(df_filtered["Date"])
    grouped = (
        df_filtered.groupby(["Date", "Series type"])["Index value ($)"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    for st in target_types:
        sub = grouped[grouped["Series type"] == st]
        if not sub.empty:
            ax.plot(sub["Date"], sub["Index value ($)"], label=st, linewidth=2)

    ax.set_title(
        "Quarterly Cost Index Aggregate Trends by Series Type (Actual Period)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Quarter", fontsize=12)
    ax.set_ylabel("Index Value (Normalized)", fontsize=12)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    out_file = output_dir / "eda_time_series_trends.png"

    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_file)
    return out_file


def _plot_cost_distribution(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Plot distribution of index values across key series types."""
    if "Index value ($)" not in df.columns or "Series type" not in df.columns:
        return None

    target_types = [
        "Cost Index",
        "Cost Index with Operating Margins (Price Index)",
        "Equipment",
        "Raw Materials",
        "Bulks",
    ]
    sub = df[df["Series type"].isin(target_types)].copy()
    if sub.empty:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Boxplot
    sns.boxplot(data=sub, x="Series type", y="Index value ($)", ax=ax1, palette="Set2")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha="right")
    ax1.set_title("Distribution of Index Values (Boxplot)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Index Value")

    # KDE / Histogram for Cost Index
    ci = sub[sub["Series type"] == "Cost Index"]["Index value ($)"].dropna()
    sns.histplot(ci, kde=True, ax=ax2, color="#2b5c8f", bins=30)
    ax2.set_title("Distribution of Primary 'Cost Index' Series", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Index Value")
    ax2.set_ylabel("Frequency")

    plt.tight_layout()
    out_file = output_dir / "eda_cost_index_distribution.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_file)
    return out_file


def _plot_seasonal_decomposition(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Plot 4-component seasonal decomposition for quarterly Cost Index."""
    if "Date" not in df.columns or "Index value ($)" not in df.columns:
        return None

    from statsmodels.tsa.seasonal import seasonal_decompose

    ci = df[df.get("Series type") == "Cost Index"] if "Series type" in df.columns else df
    if ci.empty:
        ci = df

    ts = (
        ci.groupby(pd.to_datetime(ci["Date"]))["Index value ($)"]
        .mean()
        .asfreq("QE")
        .interpolate()
    )

    if len(ts) < 16:  # need at least 4 full seasonal cycles
        return None

    decomp = seasonal_decompose(ts, model="additive", period=4)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(decomp.observed, color="#1f77b4", linewidth=1.8)
    axes[0].set_ylabel("Observed")
    axes[0].set_title(
        "Quarterly Cost Index Seasonal Decomposition (Additive, Period=4)",
        fontsize=13,
        fontweight="bold",
    )

    axes[1].plot(decomp.trend, color="#ff7f0e", linewidth=1.8)
    axes[1].set_ylabel("Trend")

    axes[2].plot(decomp.seasonal, color="#2ca02c", linewidth=1.8)
    axes[2].set_ylabel("Seasonal")

    axes[3].scatter(decomp.resid.index, decomp.resid, color="#d62728", s=15, alpha=0.7)
    axes[3].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Quarter")

    plt.tight_layout()
    out_file = output_dir / "eda_seasonality_decomposition.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_file)
    return out_file


def _plot_series_type_breakdown(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Plot series count breakdown by series type."""
    col = "Series type" if "Series type" in df.columns else None
    if not col:
        return None

    counts = df[col].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot.barh(ax=ax, color="#3470a3", edgecolor="black", alpha=0.85)
    ax.invert_yaxis()
    ax.set_title(
        "Observation Count by Series Type (Actual Period)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Number of Quarterly Observations")
    plt.tight_layout()

    out_file = output_dir / "eda_series_type_breakdown.png"

    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_file)
    return out_file


def generate_eda_plots(
    datasets: dict[str, pd.DataFrame],
    output_dir: Path | str = Path("outputs/eda"),
) -> list[Path]:
    """Generate initial EDA visualisations for all datasets.

    Saves plots to *output_dir*. Returns list of saved file paths.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Mapping of dataset names to DataFrames.
    output_dir : Path | str
        Directory to save plots to.

    Returns
    -------
    list[Path]
        Paths to saved plot files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []

    # Check for primary time series dataset (Data or data)
    primary_data = datasets.get("Data") if "Data" in datasets else datasets.get("data")
    if primary_data is not None:

        p1 = _plot_time_series_trends(primary_data, output_dir)
        if p1:
            saved.append(p1)

        p2 = _plot_cost_distribution(primary_data, output_dir)
        if p2:
            saved.append(p2)

        p3 = _plot_seasonal_decomposition(primary_data, output_dir)
        if p3:
            saved.append(p3)

        p4 = _plot_series_type_breakdown(primary_data, output_dir)
        if p4:
            saved.append(p4)

    # General dataset plots (missing values, boxplot)
    for name, df in datasets.items():
        safe = name.replace("/", "_").replace(" ", "_").lower()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        # Missing values plot
        missing = df.isnull().sum()
        if missing.sum() > 0:
            try:
                fig, ax = plt.subplots(figsize=(8, 4))
                missing[missing > 0].plot.bar(ax=ax, color="#c0392b")
                ax.set_title(f"Missing Values Count — {name}", fontsize=11, fontweight="bold")
                ax.set_ylabel("Count")
                plt.tight_layout()
                p_miss = output_dir / f"{safe}_missing.png"
                fig.savefig(p_miss, dpi=150)
                plt.close(fig)
                saved.append(p_miss)
                logger.info("Saved %s", p_miss)
            except Exception:
                logger.warning("Could not create missing-value plot for '%s'", name, exc_info=True)

        if not numeric_cols:
            continue

        # Correlation heatmap (if >= 2 numeric cols)
        if len(numeric_cols) >= 2:
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                corr = df[numeric_cols].corr()
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                ax.set_title(f"Correlation Matrix — {name}", fontsize=11, fontweight="bold")
                plt.tight_layout()
                p_corr = output_dir / f"{safe}_correlation.png"
                fig.savefig(p_corr, dpi=150)
                plt.close(fig)
                saved.append(p_corr)
            except Exception:
                logger.warning(
                    "Could not create correlation heatmap for '%s'", name, exc_info=True
                )

    return saved

