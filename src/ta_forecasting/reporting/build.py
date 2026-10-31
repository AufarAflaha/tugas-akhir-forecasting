"""Phase 6 entry point: validated inputs -> figures -> ``FINAL_REPORT.md``.

Nothing is trained or evaluated here; see :mod:`ta_forecasting.reporting.inputs` for the
files that are read.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ta_forecasting.config import Config
from ta_forecasting.reporting.document import Table, build_report
from ta_forecasting.reporting.figures import make_figures
from ta_forecasting.reporting.inputs import ReportInputs, check_no_series_ids, load_inputs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReportResult:
    """Paths written by the report step, plus the rendered tables (for tests)."""

    report_path: Path
    figures: dict[str, Path]
    tables: list[Table]
    inputs: ReportInputs


def report_path(cfg: Config) -> Path:
    return Path(cfg.output.reports_dir) / cfg.report.output_file


def run_report(cfg: Config) -> ReportResult:
    """Build the final report and its figures."""
    inputs = load_inputs(cfg)
    path = report_path(cfg)
    figures = make_figures(inputs, cfg)
    text, tables = build_report(inputs, cfg, figures, path)
    check_no_series_ids(text, inputs, path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("Report written to %s (%d tables, %d figures)", path, len(tables), len(figures))
    return ReportResult(path, figures, tables, inputs)
