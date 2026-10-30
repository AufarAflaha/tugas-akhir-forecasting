"""Load (persist) transformed data to interim/processed storage.

Supports Parquet as the primary format.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def save_to_parquet(
    datasets: dict[str, pd.DataFrame],
    output_dir: Path | str,
) -> list[Path]:
    """Save each DataFrame as a Parquet file.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Mapping of dataset names to DataFrames.
    output_dir : Path | str
        Directory to write Parquet files to.

    Returns
    -------
    list[Path]
        Paths to the written Parquet files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for name, df in datasets.items():
        # Sanitise sheet name for filesystem
        safe_name = name.replace("/", "_").replace(" ", "_").replace("\\", "_")
        out_path = output_dir / f"{safe_name}.parquet"
        df.to_parquet(out_path, engine="pyarrow", index=False)
        paths.append(out_path)
        logger.info("Saved '%s' → %s (%d rows)", name, out_path, len(df))

    return paths


def load_from_parquet(input_dir: Path | str) -> dict[str, pd.DataFrame]:
    """Load all Parquet files from a directory.

    Parameters
    ----------
    input_dir : Path | str
        Directory containing .parquet files.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of file stems to DataFrames.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    datasets: dict[str, pd.DataFrame] = {}
    for f in sorted(input_dir.glob("*.parquet")):
        df = pd.read_parquet(f, engine="pyarrow")
        datasets[f.stem] = df
        logger.info("Loaded '%s' from %s (%d rows)", f.stem, f, len(df))

    return datasets
