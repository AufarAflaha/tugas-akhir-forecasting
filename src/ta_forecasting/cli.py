"""CLI entry point for the ta-forecasting pipeline.

Each CRISP-DM phase is exposed as a CLI command.
Usage: ta-forecasting <command> --config configs/config.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ta_forecasting.config import Config, load_config, set_global_seed

app = typer.Typer(
    name="ta-forecasting",
    help="Forecasting cost indices: ETS vs ARIMA vs TFT",
    add_completion=False,
)

# Common option
ConfigOption = Annotated[
    Path,
    typer.Option("--config", "-c", help="Path to config.yaml"),
]


def _init(config_path: Path) -> Config:
    """Load config and set global seed."""
    cfg = load_config(config_path)
    set_global_seed(cfg.project.seed)
    return cfg


# ---------------------------------------------------------------------------
# Phase 2 commands
# ---------------------------------------------------------------------------


@app.command()
def extract(
    config: ConfigOption = Path("configs/config.yaml"),
    sheet: str | None = typer.Option(None, help="Extract a single sheet by name"),
) -> None:
    """Extract data from the raw .xlsx workbook."""
    cfg = _init(config)

    from ta_forecasting.data.extract import extract_workbook

    result = extract_workbook(cfg.data.raw_path, sheet_name=sheet)
    typer.echo(f"Extracted {len(result)} sheet(s).")


@app.command()
def profile(
    config: ConfigOption = Path("configs/config.yaml"),
) -> None:
    """Profile extracted data — outputs schema & aggregate stats only."""
    cfg = _init(config)

    from ta_forecasting.data.extract import extract_workbook
    from ta_forecasting.data.profile import profile_workbook

    sheets = extract_workbook(cfg.data.raw_path)
    profile_workbook(sheets, output_dir=Path("docs"))
    typer.echo("Profiling complete → docs/DATA_DICTIONARY.md")


@app.command()
def transform(
    config: ConfigOption = Path("configs/config.yaml"),
) -> None:
    """Transform & load data into interim storage (Parquet)."""
    cfg = _init(config)

    from ta_forecasting.data.extract import extract_workbook
    from ta_forecasting.data.load import save_to_parquet
    from ta_forecasting.data.transform import transform_sheets

    sheets = extract_workbook(cfg.data.raw_path)
    transformed = transform_sheets(sheets, cfg)
    save_to_parquet(transformed, output_dir=Path(cfg.data.interim_dir))
    typer.echo(f"Transformed and saved {len(transformed)} dataset(s) to {cfg.data.interim_dir}/")


@app.command()
def eda(
    config: ConfigOption = Path("configs/config.yaml"),
) -> None:
    """Generate EDA visualizations from interim data."""
    cfg = _init(config)

    from ta_forecasting.data.load import load_from_parquet
    from ta_forecasting.visualization.plots import generate_eda_plots

    datasets = load_from_parquet(Path(cfg.data.interim_dir))
    generate_eda_plots(datasets, output_dir=Path(cfg.output.eda_dir))
    typer.echo(f"EDA plots saved to {cfg.output.eda_dir}/")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
