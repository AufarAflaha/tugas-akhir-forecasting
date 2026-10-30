"""Extract data from the raw S&P Global .xlsx workbook.

IMPORTANT: This module only reads the raw file via pandas/openpyxl.
It NEVER displays, logs, or returns raw data rows — only metadata
(sheet names, row counts, column names, dtypes).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Known header offsets and key columns for key sheets in the S&P workbook
KNOWN_SHEET_CONFIGS: dict[str, dict[str, Any]] = {
    "Data": {
        "header": 4,
        "valid_key": "Date",
        "expected_cols": ["Date", "ID", "Index value ($)", "Lookup", "Label", "Series type"],
        "is_temporal": True,
    },
    "ID lookup": {
        "header": 4,
        "valid_key": "Series ID",
        "expected_cols": [
            "Series ID",
            "Label",
            "Concept",
            "Region",
            "Local currency",
            "Series type",
            "Source",
            "Definition",
        ],
        "is_temporal": False,
    },
    "Cost Model Desc": {
        "header": 3,
        "valid_key": "Cost Model",
        "expected_cols": ["Cost Model", "ID", "Description", "Cost Split", "Warning"],
        "is_temporal": False,
    },
    "IO Weight": {
        "header": 5,
        "valid_key": "Value",
        "expected_cols": ["Date", "ID", "Value"],
        "is_temporal": False,
    },
}


def _clean_sheet_df(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Clean extracted DataFrame based on sheet configuration."""
    cfg = KNOWN_SHEET_CONFIGS.get(name)

    if cfg:
        # Keep only expected columns that exist
        expected = cfg.get("expected_cols", [])
        existing = [c for c in expected if c in df.columns]
        if existing:
            df = df[existing].copy()

        # Drop rows where valid_key is missing
        valid_key = cfg.get("valid_key")
        if valid_key and valid_key in df.columns:
            if cfg.get("is_temporal"):
                # Drop footer/comment rows that cannot be converted to valid dates
                parsed_dates = pd.to_datetime(df[valid_key], errors="coerce")
                df = df.loc[parsed_dates.notna()].copy()
                df[valid_key] = pd.to_datetime(df[valid_key])
            else:
                df = df.dropna(subset=[valid_key]).copy()
    else:
        # Drop columns and rows that are completely empty
        df = df.dropna(how="all", axis=1)
        df = df.dropna(how="all", axis=0)
        # Drop unnamed empty index columns if present
        cols_to_keep = [
            c for c in df.columns
            if not (str(c).startswith("Unnamed:") and df[c].isna().mean() > 0.9)
        ]
        if cols_to_keep:
            df = df[cols_to_keep].copy()

    return df


def extract_workbook(
    path: Path | str,
    sheet_name: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Read sheets from the raw .xlsx workbook.

    Parameters
    ----------
    path : Path | str
        Absolute or relative path to the .xlsx file.
    sheet_name : str | None
        If provided, read only this sheet. Otherwise read all sheets.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of sheet names to DataFrames.

    Raises
    ------
    FileNotFoundError
        If the workbook does not exist at *path*.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Workbook not found: {path}")

    xls = pd.ExcelFile(path, engine="openpyxl")
    names = [sheet_name] if sheet_name else xls.sheet_names

    sheets: dict[str, pd.DataFrame] = {}
    for name in names:
        header = KNOWN_SHEET_CONFIGS.get(name, {}).get("header", 0)
        df = pd.read_excel(xls, sheet_name=name, header=header)
        df = _clean_sheet_df(name, df)

        sheets[name] = df
        # Log only metadata — never raw rows
        logger.info(
            "Sheet '%s': %d rows × %d cols",
            name,
            len(df),
            len(df.columns),
        )

    return sheets

