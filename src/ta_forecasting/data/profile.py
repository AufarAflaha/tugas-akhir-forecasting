"""Profile extracted data — produce schema and aggregate statistics.

Outputs go to docs/DATA_DICTIONARY.md.
NEVER writes raw data rows — only column names, dtypes, missing %,
and aggregate statistics (min, max, mean, median, std).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _profile_sheet(name: str, df: pd.DataFrame) -> str:
    """Build a markdown section profiling a single sheet.

    Returns only schema and aggregate statistics — no raw data.
    """
    lines: list[str] = []
    lines.append(f"## Sheet: `{name}`")
    lines.append("")
    lines.append(f"- **Rows**: {len(df):,}")
    lines.append(f"- **Columns**: {len(df.columns)}")
    lines.append("")

    # Column schema table
    lines.append("| # | Column | Dtype | Non-Null | Missing % |")
    lines.append("|---|--------|-------|----------|-----------|")
    for i, col in enumerate(df.columns, 1):
        non_null = df[col].notna().sum()
        missing_pct = df[col].isna().mean() * 100
        lines.append(
            f"| {i} | `{col}` | {df[col].dtype} | {non_null:,} | {missing_pct:.1f}% |"
        )
    lines.append("")

    # Aggregate stats for numeric columns only
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        lines.append("### Aggregate Statistics (numeric columns)")
        lines.append("")
        lines.append("| Column | Min | Max | Mean | Median | Std |")
        lines.append("|--------|-----|-----|------|--------|-----|")
        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) == 0:
                lines.append(f"| `{col}` | — | — | — | — | — |")
            else:
                lines.append(
                    f"| `{col}` | {s.min():.4g} | {s.max():.4g} "
                    f"| {s.mean():.4g} | {s.median():.4g} | {s.std():.4g} |"
                )
        lines.append("")

    return "\n".join(lines)


def profile_workbook(
    sheets: dict[str, pd.DataFrame],
    output_dir: Path | str = Path("docs"),
) -> Path:
    """Profile all sheets and write docs/DATA_DICTIONARY.md.

    Parameters
    ----------
    sheets : dict[str, pd.DataFrame]
        Mapping of sheet names to DataFrames (from extract).
    output_dir : Path | str
        Directory where DATA_DICTIONARY.md will be written.

    Returns
    -------
    Path
        Path to the generated DATA_DICTIONARY.md.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sections: list[str] = [
        "# Data Dictionary",
        "",
        "Auto-generated data profile. Contains **only** schema and aggregate statistics.",
        "Raw data rows are never included.",
        "",
        f"**Total sheets**: {len(sheets)}",
        "",
        "---",
        "",
    ]

    for name, df in sheets.items():
        sections.append(_profile_sheet(name, df))
        sections.append("---")
        sections.append("")

    out_path = output_dir / "DATA_DICTIONARY.md"
    out_path.write_text("\n".join(sections), encoding="utf-8")
    logger.info("Data dictionary written to %s", out_path)
    return out_path
