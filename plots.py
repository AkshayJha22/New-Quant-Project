"""
plots.py — every chart in the app, built on one shared dark theme.

Colour language (kept consistent across all figures so the dashboard reads
as one system, not thirteen unrelated charts):

    TEAL   premium collected / strategy / wins
    ROSE   losses, drawdowns, tail risk
    SAND   implied volatility (what you sell)
    BLUE   realised volatility (what is delivered)
    SLATE  benchmark & neutral reference lines
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TEAL = "#3FD0B6"
ROSE = "#F06A8A"
SAND = "#E5B96B"
BLUE = "#5AA7F0"
SLATE = "#7C8DB0"
PURPLE = "#A78BFA"
BG = "rgba(0,0,0,0)"
PLOT_BG = "#101720"
GRID = "#1D2735"
TEXT = "#D6DEE8"

COMPARE_COLORS = [TEAL, SAND, BLUE, PURPLE, ROSE]


def _theme(fig: go.Figure, height: int = 420, legend_top: bool = True) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT, family="Inter, 'Segoe UI', sans-serif", size=13),
        height=height, margin=dict(l=10, r=10, t=48, b=10),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#182230", bordercolor=GRID),
        legend=(dict(orientation="h", yanchor="bottom", y=1.02, x=0)
                if legend_top else dict()),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

def fig_equity(daily: pd.DataFrame, log: bool = False) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.index, y=daily["bench"], name="Buy & hold",
                             line=dict(color=SLATE, width=1.4, dash="dot")))
    fig.add_trace(go.Scatter(x=daily.index, y=daily["equity"], name="Strategy",
                             line=dict(color=TEAL, width=2.2),
                             fill="tozeroy", fillcolor="rgba(63,208,182,0.06)"))
    fig.update_layout(title="Equity curve — strategy vs buy & hold",
                      yaxis_title="Account value ($)")
    if log:
        fig.update_yaxes(type="log")
    return _theme(fig, 440)


def fig_drawdown(dd: pd.Series, dd_bench: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd_bench.index, y=100 * dd_bench, name="Buy & hold",
                             line=dict(color=SLATE, width=1.2, dash="dot")))
    fig.add_trace(go.Scatter(x=dd.index, y=100 * dd, name="Strategy",
                             line=dict(color=ROSE, width=1.8),
                             fill="tozeroy", fillcolor="rgba(240,106,138,0.12)"))
    fig.update_layout(title="Drawdown from peak", yaxis_title="Drawdown (%)")
    return _theme(fig, 300)


def fig_monthly_heatmap(pivot: pd.DataFrame) -> go.Figure:
    z = 100 * pivot.values
    txt = np.where(np.isnan(z), "", np.vectorize(lambda v: f"{v:.1f}")(np.nan_to_num(z)))
    lim = max(np.nanmax(np.abs(z)), 1e-9)
    fig = go.Figure(go.Heatmap(
        z=z, x=list(pivot.columns), y=[str(y) for y in pivot.index],
        colorscale=[[0.0, ROSE], [0.5, PLOT_BG], [1.0, TEAL]],
        zmin=-lim, zmax=lim, text=txt, texttemplate="%{text}",
        textfont=dict(size=10), hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>",
        colorbar=dict(title="%", outlinewidth=0),
    ))
    fig.update_layout(title="Monthly returns (%)")
    fig.update_yaxes(autorange="reversed")
    f = _theme(fig, max(300, 26 * len(pivot.index) + 120))
    f.update_layout(hovermode="closest")
    return f


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

def fig_trade_bars(trades: pd.DataFrame) -> go.Figure:
    colors = np.where(trades["pnl_$"] > 0, TEAL, ROSE)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=trades["exit_date"], y=trades["pnl_$"], name="Trade PnL",
                         marker_color=colors, marker_line_width=0,
                         customdata=trades["legs"],
                         hovertemplate="%{x|%d %b %Y}<br>%{customdata}<br>PnL $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=trades["exit_date"], y=trades["pnl_$"].cumsum(),
                             name="Cumulative", line=dict(color=SAND, width=1.8)),
                  secondary_y=True)
    fig.update_layout(title="PnL per trade over time")
    fig.update_yaxes(title_text="Trade PnL ($)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative ($)", secondary_y=True,
                     gridcolor="rgba(0,0,0,0)")
    return _theme(fig, 400)


def fig_pnl_hist(trades: pd.DataFrame) -> go.Figure:
    x = 100 * trades["pnl_pct_credit"].replace([np.inf, -np.inf], np.nan).dropna()
    skew = x.skew() if len(x) > 2 else float("nan")
    fig = go.Figure(go.Histogram(
        x=x, nbinsx=40, marker_color=TEAL, opacity=0.85,
        marker_line=dict(color=PLOT_BG, width=1)))
    fig.add_vline(x=0, line_color=TEXT, line_width=1, line_dash="dash")
    if len(x):
        fig.add_vline(x=float(x.mean()), line_color=SAND, line_width=1.6,
                      annotation_text=f"mean {x.mean():.0f}%",
                      annotation_font_color=SAND)
    fig.update_layout(
        title=f"Trade PnL distribution (% of credit) — skew {skew:.2f}",
        xaxis_title="PnL as % of credit received", yaxis_title="Trades",
        hovermode="closest")
    return _theme(fig, 380)


def fig_exit_donut(trades: pd.DataFrame) -> go.Figure:
    counts = trades["exit_reason"].value_counts()
    cmap = {"Profit target": TEAL, "Expiry": SLATE, "Stop loss": ROSE}
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.62,
        marker=dict(colors=[cmap.get(k, PURPLE) for k in counts.index],
                    line=dict(color=PLOT_BG, width=2)),
        textinfo="label+percent"))
    fig.update_layout(title="How trades ended", showlegend=False, hovermode="closest")
    return _theme(fig, 380, legend_top=False)


def fig_iv_pnl_scatter(trades: pd.DataFrame) -> go.Figure:
    t = trades.dropna(subset=["entry_iv", "pnl_pct_credit"])
    x, y = 100 * t["entry_iv"], 100 * t["pnl_pct_credit"]
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="markers", name="Trades",
        marker=dict(color=np.where(y > 0, TEAL, ROSE), size=7, opacity=0.8),
        customdata=t["entry_date"].dt.strftime("%d %b %Y"),
        hovertemplate="Entry %{customdata}<br>IV %{x:.1f}% → PnL %{y:.0f}% of credit<extra></extra>"))
    if len(x) > 2:
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(float(x.min()), float(x.max()), 20)
        fig.add_trace(go.Scatter(x=xs, y=a + b * xs, name=f"fit: {b:.1f}%/vol pt",
                                 line=dict(color=SAND, width=1.6, dash="dash")))
    fig.update_layout(title="Does selling richer vol pay? Entry IV vs trade outcome",
                      xaxis_title="Implied vol at entry (%)",
                      yaxis_title="Trade PnL (% of credit)", hovermode="closest")
    return _theme(fig, 400)


# ---------------------------------------------------------------------------
# Volatility risk premium
# ---------------------------------------------------------------------------

def fig_vrp_timeseries(trades: pd.DataFrame) -> go.Figure:
    t = trades.dropna(subset=["entry_iv", "rv_trade"]).sort_values("entry_date")
    spread = 100 * (t["entry_iv"] - t["rv_trade"])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38],
                        vertical_spacing=0.07)
    fig.add_trace(go.Scatter(x=t["entry_date"], y=100 * t["entry_iv"],
                             name="IV sold at entry", line=dict(color=SAND, width=1.8)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t["entry_date"], y=100 * t["rv_trade"],
                             name="RV delivered over trade",
                             line=dict(color=BLUE, width=1.6)), row=1, col=1)
    fig.add_trace(go.Bar(x=t["entry_date"], y=spread, name="Premium (IV − RV)",
                         marker_color=np.where(spread > 0, TEAL, ROSE),
                         marker_line_width=0), row=2, col=1)
    fig.update_layout(title="The volatility risk premium, trade by trade")
    fig.update_yaxes(title_text="Vol (%)", row=1, col=1)
    fig.update_yaxes(title_text="IV − RV (pts)", row=2, col=1)
    return _theme(fig, 520)


def fig_vrp_scatter(trades: pd.DataFrame) -> go.Figure:
    t = trades.dropna(subset=["entry_iv", "rv_trade"])
    x, y = 100 * t["entry_iv"], 100 * t["rv_trade"]
    hi = float(max(x.max(), y.max())) * 1.05 if len(x) else 50.0
    won = t["pnl_$"] > 0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", name="IV = RV",
                             line=dict(color=TEXT, width=1, dash="dash")))
    for mask, name, color in [(won, "Winning trades", TEAL),
                              (~won, "Losing trades", ROSE)]:
        fig.add_trace(go.Scatter(
            x=x[mask], y=y[mask], mode="markers", name=name,
            marker=dict(color=color, size=7, opacity=0.8)))
    share = float((x > y).mean()) if len(x) else float("nan")
    fig.update_layout(
        title=f"Implied vs delivered vol — {share:.0%} of trades below the line "
              f"(premium was real)",
        xaxis_title="IV at entry (%)", yaxis_title="Realised vol over trade (%)",
        hovermode="closest")
    return _theme(fig, 440)


def fig_rolling_vol(daily: pd.DataFrame, trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.index, y=100 * daily["iv"], name="Implied vol",
                             line=dict(color=SAND, width=1.6)))
    fig.add_trace(go.Scatter(x=daily.index, y=100 * daily["rv"],
                             name="Realised vol (trailing)",
                             line=dict(color=BLUE, width=1.3)))
    if len(trades):
        ent = trades.set_index("entry_date")["entry_iv"]
        fig.add_trace(go.Scatter(x=ent.index, y=100 * ent, mode="markers",
                                 name="Entries",
                                 marker=dict(symbol="triangle-down", size=8,
                                             color=TEAL,
                                             line=dict(color=PLOT_BG, width=1))))
    fig.update_layout(title="Volatility regime — implied vs realised",
                      yaxis_title="Annualised vol (%)")
    return _theme(fig, 400)


# ---------------------------------------------------------------------------
# Strategy & stress
# ---------------------------------------------------------------------------

def fig_payoff(curves: dict) -> go.Figure:
    g, yE, y0 = curves["grid"], curves["expiry"], curves["t0"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g, y=np.maximum(yE, 0), mode="lines", showlegend=False,
                             line=dict(width=0), fill="tozeroy",
                             fillcolor="rgba(63,208,182,0.14)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=g, y=np.minimum(yE, 0), mode="lines", showlegend=False,
                             line=dict(width=0), fill="tozeroy",
                             fillcolor="rgba(240,106,138,0.14)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=g, y=yE, name="At expiry",
                             line=dict(color=TEAL, width=2.4)))
    fig.add_trace(go.Scatter(x=g, y=y0, name="At entry (T-0)",
                             line=dict(color=SAND, width=1.5, dash="dash")))
    fig.add_hline(y=0, line_color=TEXT, line_width=1)

    s0 = float(g[len(g) // 2])
    fig.add_vline(x=s0, line_color=SLATE, line_width=1, line_dash="dot",
                  annotation_text="spot", annotation_font_color=SLATE)
    # Breakevens: zero crossings of the expiry payoff.
    sign = np.sign(yE)
    for i in np.where(np.diff(sign) != 0)[0]:
        x0, x1, f0, f1 = g[i], g[i + 1], yE[i], yE[i + 1]
        bx = x0 if f1 == f0 else x0 - f0 * (x1 - x0) / (f1 - f0)
        fig.add_trace(go.Scatter(x=[bx], y=[0], mode="markers", showlegend=False,
                                 marker=dict(color=TEXT, size=8, symbol="diamond"),
                                 hovertemplate=f"breakeven {bx:,.0f}<extra></extra>"))
    fig.update_layout(
        title=f"Payoff per contract — {curves['label']}  "
              f"(net credit ${curves['credit_$']:,.0f})",
        xaxis_title="Underlying at expiry", yaxis_title="PnL per contract ($)",
        hovermode="x")
    return _theme(fig, 460)


def fig_cost_sensitivity(sens: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=sens["slippage_pct"] * 100, y=sens["sharpe"],
                             name="Sharpe", line=dict(color=TEAL, width=2.2),
                             mode="lines+markers"), secondary_y=False)
    fig.add_trace(go.Scatter(x=sens["slippage_pct"] * 100, y=100 * sens["cagr"],
                             name="CAGR", line=dict(color=SAND, width=1.8, dash="dash"),
                             mode="lines+markers"), secondary_y=True)
    fig.add_hline(y=0, line_color=TEXT, line_width=1)
    fig.update_layout(title="How fast do costs eat the edge? Slippage sensitivity")
    fig.update_xaxes(title_text="Slippage per side (% of premium)")
    fig.update_yaxes(title_text="Sharpe", secondary_y=False)
    fig.update_yaxes(title_text="CAGR (%)", secondary_y=True,
                     gridcolor="rgba(0,0,0,0)")
    return _theme(fig, 400)


def fig_compare(curves: dict[str, pd.Series], bench: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bench.index, y=100 * bench / bench.iloc[0],
                             name="Buy & hold",
                             line=dict(color=SLATE, width=1.2, dash="dot")))
    for (name, eq), color in zip(curves.items(), COMPARE_COLORS):
        fig.add_trace(go.Scatter(x=eq.index, y=100 * eq / eq.iloc[0], name=name,
                                 line=dict(color=color, width=1.8)))
    fig.update_layout(title="All strategies, same settings (indexed to 100)",
                      yaxis_title="Growth of 100")
    return _theme(fig, 460)
