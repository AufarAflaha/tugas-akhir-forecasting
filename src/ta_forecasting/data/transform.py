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

    # Convert the time column safely to quarterly PeriodIndex
    if pd.api.types.is_datetime64_any_dtype(df[time_col]):
        periods = df[time_col].dt.to_period("Q")
    elif isinstance(df[time_col].dtype, pd.PeriodDtype):
        periods = df[time_col]
    else:
        # String dates (e.g. '2026-03-31') or quarter strings (e.g. '2026Q1')
        try:
            periods = pd.PeriodIndex(df[time_col].astype(str), freq="Q")
        except Exception:
            dt_series = pd.to_datetime(df[time_col], errors="coerce")
            periods = dt_series.dt.to_period("Q")

    if isinstance(periods, pd.Series):
        mask = (periods <= cutoff).fillna(False).to_numpy()
    else:
        mask = (periods <= cutoff) & ~periods.isna()
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
        logger.info(
            "Transforming sheet '%s' (%d rows × %d cols)",
            name,
            len(df),
            len(df.columns),
        )

        df_out = df.copy()

        # Handle sheet 'Data' specifically (primary time series dataset)
        if name == "Data" and "Date" in df_out.columns:
            # Ensure Date is datetime
            df_out["Date"] = pd.to_datetime(df_out["Date"], errors="coerce")
            df_out = df_out.dropna(subset=["Date"]).copy()

            # Filter projections after LAST_ACTUAL_QUARTER
            df_out = filter_after_last_actual(
                df_out,
                time_col="Date",
                last_actual_quarter=config.project.last_actual_quarter,
            )

            # Standardize Quarter as string column (e.g. '2000Q1') for easy querying
            df_out["Quarter"] = df_out["Date"].dt.to_period("Q").astype(str)

            # Ensure numeric index value
            if "Index value ($)" in df_out.columns:
                df_out["Index value ($)"] = pd.to_numeric(
                    df_out["Index value ($)"], errors="coerce"
                )

            # Ensure string types for identifiers
            for str_col in ["ID", "Lookup", "Label", "Series type"]:
                if str_col in df_out.columns:
                    df_out[str_col] = df_out[str_col].astype(str)

        else:
            # For other sheets: check if an actual temporal column exists
            # Only treat as temporal if values can truly be parsed as dates/quarters
            temporal_col = None
            for col in df_out.columns:
                col_name_lower = str(col).lower()
                is_q = any(kw in col_name_lower for kw in ("quarter", "period"))
                if is_q and not col_name_lower.startswith("unnamed"):
                    temporal_col = col
                    break

                elif "date" in col_name_lower and name != "IO Weight":
                    # In IO Weight, 'Date' column is actually a model identifier
                    parsed = pd.to_datetime(df_out[col].head(10), errors="coerce")
                    if parsed.notna().mean() > 0.5:
                        temporal_col = col
                        break

            if temporal_col:
                logger.info(
                    "Applying temporal filter on '%s' for sheet '%s'",
                    temporal_col,
                    name,
                )
                try:
                    df_out = filter_after_last_actual(
                        df_out,
                        time_col=temporal_col,
                        last_actual_quarter=config.project.last_actual_quarter,
                    )
                except Exception:
                    logger.warning(
                        "Could not apply temporal filter on '%s' in sheet '%s'",
                        temporal_col,
                        name,
                    )

        transformed[name] = df_out

    return transformed

