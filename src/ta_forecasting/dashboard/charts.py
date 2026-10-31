"""Plotly figures for the dashboard (pure functions, no Streamlit)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from ta_forecasting.config import Config
from ta_forecasting.dashboard.data import QOQ, QUARTER_END, YEAR
from ta_forecasting.data.prepare import QUARTER_LABEL
from ta_forecasting.evaluation.backtest import MODEL
from ta_forecasting.forecasting.horizon import validated_steps
from ta_forecasting.forecasting.intervals import LOWER, UPPER
from ta_forecasting.models.base import PREDICTION, STEP

DEFAULT_COLOR = "#7f7f7f"


def _hex_to_rgba(color: str, alpha: float) -> str:
    color = color.lstrip("#")
    r, g, b = (int(color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _quarter_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Period(ts, freq="Q").to_timestamp(how="start")


def main_chart(actual: pd.DataFrame, forecast: pd.DataFrame, start: int, end: int,
               base_year: int, labels: dict[str, str], cfg: Config) -> go.Figure:
    """Actual index, forecasts with intervals, horizon shading and QoQ bars (right axis)."""
    d = cfg.dashboard
    colors = d.colors
    act = actual[(actual[YEAR] >= start) & (actual[YEAR] <= end)]
    fc = forecast[(forecast[YEAR] >= start) & (forecast[YEAR] <= end)]
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=act[QUARTER_END], y=act[QOQ], name="Perubahan QoQ aktual (%)", yaxis="y2",
        marker_color=colors.get("qoq", DEFAULT_COLOR), opacity=0.6,
        hovertemplate="%{customdata}<br>QoQ: %{y:.1f}%<extra></extra>",
        customdata=act[QUARTER_LABEL]))
    fig.add_trace(go.Scatter(
        x=act[QUARTER_END], y=act[PREDICTION], name="Aktual", mode="lines",
        line={"color": colors.get("actual", "#000000"), "width": 2},
        customdata=act[QUARTER_LABEL],
        hovertemplate="%{customdata}<br>Aktual: %{y:.1f}<extra></extra>"))

    last = actual.iloc[-1] if len(actual) else None
    for model, g in fc.groupby(MODEL, sort=False):
        color = colors.get(model, DEFAULT_COLOR)
        name = labels.get(model, model)
        band = g.dropna(subset=[LOWER, UPPER])
        if len(band):
            fig.add_trace(go.Scatter(
                x=pd.concat([band[QUARTER_END], band[QUARTER_END][::-1]]),
                y=pd.concat([band[UPPER], band[LOWER][::-1]]),
                fill="toself", fillcolor=_hex_to_rgba(color, d.interval_opacity),
                line={"width": 0}, hoverinfo="skip", showlegend=True,
                name=f"Interval {cfg.forecast.interval_level:.0%} {name}",
                legendgroup=model))
        line = g
        if last is not None and last[YEAR] >= start:
            line = pd.concat([pd.DataFrame({QUARTER_END: [last[QUARTER_END]],
                                            PREDICTION: [last[PREDICTION]],
                                            QUARTER_LABEL: [last[QUARTER_LABEL]]}), g])
        fig.add_trace(go.Scatter(
            x=line[QUARTER_END], y=line[PREDICTION], name=f"Ramalan {name}", mode="lines",
            line={"color": color, "width": 2, "dash": "dash"}, legendgroup=model,
            customdata=line[QUARTER_LABEL],
            hovertemplate="%{customdata}<br>" + name + ": %{y:.1f}<extra></extra>"))

    _horizon_shading(fig, forecast, start, end, cfg)
    fig.update_layout(
        height=d.chart_height, hovermode="x unified", barmode="relative",
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        yaxis={"title": f"{d.value_label} ({base_year} = 100)", "rangemode": "tozero"},
        yaxis2={"title": "Perubahan QoQ (%)", "overlaying": "y", "side": "right",
                "showgrid": False, "zeroline": True},
        xaxis={"title": "Kuartal"},
        separators=",.",
    )
    return fig


def _horizon_shading(fig: go.Figure, forecast: pd.DataFrame, start: int, end: int,
                     cfg: Config) -> None:
    """Boundary line plus shaded validated (h <= H) and long-horizon areas."""
    if forecast.empty:
        return
    d = cfg.dashboard
    v = validated_steps(cfg)
    steps = forecast.drop_duplicates(STEP).sort_values(STEP)
    first = _quarter_start(steps[QUARTER_END].iloc[0])
    if first.year > end:
        return
    val_end = steps.loc[steps[STEP] == min(v, steps[STEP].max()), QUARTER_END].iloc[0]
    last_end = steps[QUARTER_END].iloc[-1]
    fig.add_vline(x=first, line={"color": "#555555", "dash": "dot", "width": 1})
    fig.add_vrect(x0=first, x1=val_end, fillcolor=d.colors.get("validated_band", "rgba(0,0,0,0)"),
                  line_width=0, layer="below", annotation_text=f"tervalidasi utama (h 1–{v})",
                  annotation_position="top left", annotation_font_size=10)
    if steps[STEP].max() > v:
        fig.add_vrect(x0=val_end, x1=last_end,
                      fillcolor=d.colors.get("long_band", "rgba(0,0,0,0)"), line_width=0,
                      layer="below", annotation_text=f"jangka panjang (h > {v})",
                      annotation_position="top left", annotation_font_size=10)


def error_chart(errors: pd.DataFrame, models: list[str], labels: dict[str, str],
                cfg: Config) -> go.Figure:
    """Long-horizon MAPE per step and model (aggregate)."""
    primary = cfg.metrics.primary
    fig = go.Figure()
    for m in models:
        g = errors[errors[MODEL] == m].sort_values(STEP)
        fig.add_trace(go.Scatter(x=g[STEP], y=g[primary], name=labels.get(m, m),
                                 mode="lines+markers",
                                 line={"color": cfg.dashboard.colors.get(m, DEFAULT_COLOR)}))
    v = validated_steps(cfg)
    fig.add_vrect(x0=0.5, x1=v + 0.5, fillcolor=cfg.dashboard.colors.get("validated_band"),
                  line_width=0, layer="below")
    fig.update_layout(height=380, xaxis={"title": "Horizon (kuartal ke depan)", "dtick": 1},
                      yaxis={"title": f"{primary.upper()} backtest (%)"}, separators=",.",
                      margin={"l": 10, "r": 10, "t": 30, "b": 10}, hovermode="x unified")
    return fig
