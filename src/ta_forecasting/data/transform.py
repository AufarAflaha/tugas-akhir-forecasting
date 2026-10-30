"""Transform extracted data.

Key responsibilities:
1. Filter out S&P projection data after LAST_ACTUAL_QUARTER.
2. Standardise temporal columns to pd.Period (quarterly).
3. Basic data cleaning and type coercion.
"""

from __future__ import annotations

import logging

import pandas as pd

from ta_forecasting.config import Config

logger = logging.getLogger(__name__)


def _quarter_to_period(q_str: str) -> pd.Period:
    """Convert a quarter string like '2026Q1' to a pd.Period."""
    return pd.Period(q_str, freq="Q")


def filter_after_last_actual(
    df: pd.DataFrame,
    time_col: str,
    last_actual_quarter: str,
) -> pd.DataFrame:
    """Remove rows whose temporal column is after LAST_ACTUAL_QUARTER.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a temporal column.
    time_col : str
        Name of the column containing quarter/date information.
    last_actual_quarter : str
        Cutoff quarter string, e.g. "2026Q1".

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only actual data.
    """
    cutoff = _quarter_to_period(last_actual_quarter)

    # Try to convert the time column to Period
    periods = pd.PeriodIndex(df[time_col].astype(str), freq="Q")

    mask = periods <= cutoff
    n_before = len(df)
    filtered = df.loc[mask].copy()
    n_removed = n_before - len(filtered)

    logger.info(
        "Temporal filter on '%s': kept %d rows, removed %d projection rows (after %s)",
        time_col,
        len(filtered),
        n_removed,
        last_actual_quarter,
    )

    return filtered


def transform_sheets(
    sheets: dict[str, pd.DataFrame],
    config: Config,
) -> dict[str, pd.DataFrame]:
    """Apply transformations to all extracted sheets.

    This is a first-pass transform for data understanding.
    Full feature engineering happens in Phase 3.

    Parameters
    ----------
    sheets : dict[str, pd.DataFrame]
        Raw extracted sheets.
    config : Config
        Project configuration.

    Returns
    -------
    dict[str, pd.DataFrame]
        Transformed sheets.
    """
    transformed: dict[str, pd.DataFrame] = {}

    for name, df in sheets.items():
        logger.info("Transforming sheet '%s' (%d rows × %d cols)", name, len(df), len(df.columns))

        # Identify potential time columns (heuristic — will refine in Phase 2 execution)
        time_cols = [
            c for c in df.columns
            if any(kw in str(c).lower() for kw in ("quarter", "date", "period", "year", "q"))
        ]

        if time_cols:
            # Use the first matching column as temporal key
            time_col = time_cols[0]
            logger.info("Using '%s' as temporal column for sheet '%s'", time_col, name)

            try:
                df = filter_after_last_actual(
                    df,
                    time_col=time_col,
                    last_actual_quarter=config.project.last_actual_quarter,
                )
            except Exception:
                logger.warning(
                    "Could not apply temporal filter on '%s' in sheet '%s' — "
                    "column may need manual inspection",
                    time_col,
                    name,
                )
        else:
            logger.warning(
                "No temporal column detected in sheet '%s' — skipping temporal filter",
                name,
            )

        transformed[name] = df

    return transformed
