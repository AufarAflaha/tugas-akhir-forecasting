"""Extract data from the raw S&P Global .xlsx workbook.

IMPORTANT: This module only reads the raw file via pandas/openpyxl.
It NEVER displays, logs, or returns raw data rows — only metadata
(sheet names, row counts, column names, dtypes).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


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
        If provided, read only this sheet.  Otherwise read all sheets.

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
        df = pd.read_excel(xls, sheet_name=name)
        sheets[name] = df
        # Log only metadata — never raw rows
        logger.info(
            "Sheet '%s': %d rows × %d cols",
            name,
            len(df),
            len(df.columns),
        )

    return sheets
