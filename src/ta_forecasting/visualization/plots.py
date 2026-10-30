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

    for name, df in datasets.items():
        safe = name.replace("/", "_").replace(" ", "_")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if not numeric_cols:
            logger.warning("Sheet '%s' has no numeric columns — skipping EDA plots", name)
            continue

        # 1. Distribution boxplot
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            df[numeric_cols].boxplot(ax=ax, rot=45)
            ax.set_title(f"Distribution — {name}")
            ax.set_ylabel("Value")
            plt.tight_layout()
            p = output_dir / f"{safe}_boxplot.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            saved.append(p)
            logger.info("Saved %s", p)
        except Exception:
            logger.warning("Could not create boxplot for '%s'", name, exc_info=True)

        # 2. Correlation heatmap (if ≥2 numeric cols)
        if len(numeric_cols) >= 2:
            try:
                fig, ax = plt.subplots(figsize=(10, 8))
                corr = df[numeric_cols].corr()
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                ax.set_title(f"Correlation — {name}")
                plt.tight_layout()
                p = output_dir / f"{safe}_correlation.png"
                fig.savefig(p, dpi=150)
                plt.close(fig)
                saved.append(p)
                logger.info("Saved %s", p)
            except Exception:
                logger.warning(
                    "Could not create correlation heatmap for '%s'",
                    name,
                    exc_info=True,
                )

        # 3. Missing-value summary
        try:
            missing = df.isnull().sum()
            if missing.sum() > 0:
                fig, ax = plt.subplots(figsize=(10, 5))
                missing[missing > 0].plot.bar(ax=ax)
                ax.set_title(f"Missing Values — {name}")
                ax.set_ylabel("Count")
                plt.tight_layout()
                p = output_dir / f"{safe}_missing.png"
                fig.savefig(p, dpi=150)
                plt.close(fig)
                saved.append(p)
                logger.info("Saved %s", p)
        except Exception:
            logger.warning("Could not create missing-value plot for '%s'", name, exc_info=True)

    return saved
