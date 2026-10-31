"""Number formatting and Markdown tables for the Indonesian report."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd

from ta_forecasting.config import Config


class Formatter:
    """Formats numbers and labels with the settings in ``report``."""

    def __init__(self, cfg: Config) -> None:
        self.r = cfg.report

    def num(self, value: float, decimals: int | None = None) -> str:
        """Fixed-point number with the configured decimal separator."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "–"
        digits = self.r.decimals if decimals is None else decimals
        return f"{float(value):.{digits}f}".replace(".", self.r.decimal_separator)

    def signed(self, value: float, decimals: int | None = None) -> str:
        """Number with an explicit sign (for deltas)."""
        text = self.num(abs(value), decimals)
        return ("+" if value >= 0 else "-") + text

    def count(self, value: float) -> str:
        """Integer with the configured thousands separator."""
        return f"{int(value):,}".replace(",", self.r.thousands_separator)

    def mval(self, metric: str, value: float) -> str:
        """Metric value with ``%`` when the metric label is a percentage."""
        return self.num(value) + ("%" if "(%)" in self.metric(metric) else "")

    def pval(self, value: float) -> str:
        """p-value; values that round to zero are shown as ``< 0,000…1``."""
        digits = self.r.p_value_decimals
        if round(float(value), digits) == 0:
            return "< " + self.num(10.0**-digits, digits)
        return self.num(value, digits)

    def pct(self, part: float, whole: float) -> str:
        """``part`` as a percentage of ``whole``."""
        return self.num(100.0 * part / whole) + "%"

    def model(self, name: str) -> str:
        return self.r.model_labels.get(name, name)

    def split(self, name: str) -> str:
        return self.r.split_labels.get(name, name)

    def metric(self, name: str) -> str:
        return self.r.metric_labels.get(name, name.upper())

    def metric_short(self, name: str) -> str:
        """Metric name without the unit, e.g. ``MAPE``."""
        return self.metric(name).split(" (")[0]

    def join(self, items: Sequence[str]) -> str:
        """Indonesian list: ``a``, ``a dan b``, ``a, b, dan c``."""
        items = list(items)
        if len(items) <= 1:
            return "".join(items)
        if len(items) == 2:
            return f"{items[0]} dan {items[1]}"
        return ", ".join(items[:-1]) + ", dan " + items[-1]


def markdown_table(frame: pd.DataFrame) -> str:
    """GitHub Markdown table; cells must already be formatted strings."""
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])
