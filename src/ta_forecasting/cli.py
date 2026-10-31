"""CLI entry point for the ta-forecasting pipeline.

Each CRISP-DM phase is exposed as a CLI command.
Usage: ta-forecasting <command> --config configs/config.yaml
"""

from __future__ import annotations

from enum import StrEnum
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
# Phase 3 commands
# ---------------------------------------------------------------------------


@app.command()
def prep(
    config: ConfigOption = Path("configs/config.yaml"),
) -> None:
    """Prepare the model-ready panel, temporal split, scaler and TFT datasets."""
    cfg = _init(config)

    import pandas as pd

    from ta_forecasting.data.prepare import build_timeseries_datasets, run_prep

    result = run_prep(cfg)
    summary = result.summary
    typer.echo(
        f"Panel: {summary['n_rows']} rows, {summary['n_series']} series, "
        f"{summary['n_quarters']} quarters"
    )
    for name, info in summary["splits"].items():
        typer.echo(
            f"  {name:<5} {info['start_quarter']} - {info['end_quarter']} "
            f"({info['n_quarters']} quarters, {info['n_rows']} rows)"
        )

    panel = pd.read_parquet(result.panel_path)
    datasets = build_timeseries_datasets(panel, cfg)
    names = ("train", "val", "test")
    sizes = ", ".join(f"{n}={len(ds)}" for n, ds in zip(names, datasets, strict=True))
    typer.echo(f"TimeSeriesDataSet windows: {sizes}")
    typer.echo(f"Saved artefacts to {cfg.data.processed_dir}/")


# ---------------------------------------------------------------------------
# Phase 4 commands
# ---------------------------------------------------------------------------


class ModelChoice(StrEnum):
    """Models accepted by ``train --model``."""

    ets = "ets"
    arima = "arima"
    tft = "tft"
    all = "all"


@app.command()
def train(
    model: Annotated[
        ModelChoice, typer.Option("--model", "-m", help="Model to train")
    ] = ModelChoice.all,
    config: ConfigOption = Path("configs/config.yaml"),
) -> None:
    """Train ETS / ARIMA / TFT on train (and val for TFT) and save artefacts to models/."""
    cfg = _init(config)

    from ta_forecasting.models.train import resolve_names, setup_logging, train_models

    log_path = setup_logging(cfg, f"train_{model.value}.log")
    names = resolve_names(model.value)
    summary = train_models(cfg, names)
    for name in names:
        info = summary["models"][name]
        typer.echo(f"{name}: {info['fit_seconds']:.1f} s -> {', '.join(info['artefacts'])}")
    summary_file = Path(cfg.output.models_dir) / cfg.output.training_summary_file
    typer.echo(f"Summary: {summary_file} | log: {log_path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
