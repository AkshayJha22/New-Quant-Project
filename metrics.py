"""
metrics.py — performance statistics on equity curves and trade logs.

All functions are pure (DataFrame/Series in, numbers out) so they can be
tested without Streamlit and reused for the benchmark and the strategy-
comparison view.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fraction below the running peak (0 at highs, negative in drawdown)."""
    return equity / equity.cummax() - 1.0


def perf_stats(equity: pd.Series, rf: float = 0.0) -> dict:
    """Annualised performance statistics for an equity curve."""
    rets = equity.pct_change().dropna()
    n_years = max(len(rets) / TRADING_DAYS, 1e-9)

    total = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / n_years) - 1.0
    ann_vol = rets.std() * np.sqrt(TRADING_DAYS)

    excess = rets - rf / TRADING_DAYS
    sharpe = excess.mean() / rets.std() * np.sqrt(TRADING_DAYS) if rets.std() > 0 else np.nan
    downside = rets[rets < 0].std()
    sortino = (excess.mean() * TRADING_DAYS / (downside * np.sqrt(TRADING_DAYS))
               if downside and downside > 0 else np.nan)

    dd = drawdown_series(equity)
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    return {
        "Total return": total, "CAGR": cagr, "Ann. vol": ann_vol,
        "Sharpe": sharpe, "Sortino": sortino,
        "Max drawdown": max_dd, "Calmar": calmar,
        "Skew (daily)": rets.skew(), "Kurtosis (daily)": rets.kurt(),
    }


def trade_stats(trades: pd.DataFrame) -> dict:
    """Win rate, expectancy and tail shape of the per-trade PnL distribution."""
    if trades is None or len(trades) == 0:
        return {}
    pnl = trades["pnl_$"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    return {
        "Trades": int(len(pnl)),
        "Win rate": len(wins) / len(pnl),
        "Avg win $": wins.mean() if len(wins) else np.nan,
        "Avg loss $": losses.mean() if len(losses) else np.nan,
        "Profit factor": (wins.sum() / abs(losses.sum())
                          if len(losses) and losses.sum() != 0 else np.inf),
        "Best trade $": pnl.max(), "Worst trade $": pnl.min(),
        "Trade PnL skew": pnl.skew() if len(pnl) > 2 else np.nan,
        "Avg days held": trades["days_held"].mean(),
    }


def monthly_return_table(equity: pd.Series) -> pd.DataFrame:
    """Year x month table of compounded monthly returns (for the heatmap)."""
    rets = equity.pct_change().dropna()
    monthly = (1.0 + rets).resample("ME").prod() - 1.0
    tbl = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "ret": monthly.values,
    })
    pivot = tbl.pivot(index="year", columns="month", values="ret")
    pivot.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in pivot.columns]
    return pivot


def time_in_market(daily: pd.DataFrame) -> float:
    return float(daily["in_market"].mean())
