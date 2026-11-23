"""
validation.py: compare the backtester with Cboe's PUT and BXM indices.

Cboe's S&P 500 PutWrite index (PUT) and BuyWrite index (BXM) are built from
REAL S&P 500 option prices:
  PUT  sells a one-month at-the-money SPX put every month, fully
       collateralised by Treasury bills.
  BXM  holds the S&P 500 (dividends reinvested) and sells a one-month
       at-the-money SPX call every month.

Running the same two strategies through this backtester, whose premiums are
synthesised from the VIX, measures how far synthetic premiums are from real
ones. The replication uses the S&P 500 index (^GSPC raw closes):
  * short put: notional = 100% of equity (fully cash-secured), strike at spot,
    held to expiry, no costs, cash earning the 13-week T-bill;
  * covered call: the same, holding the index and earning a dividend yield
    estimated from SPY's adjusted and raw closes.
Each runs twice: with a flat smile (the VIX prices every strike, the original
engine) and with the volatility smile (ATM volatility backed out of the VIX).

Usage:
  python validation.py              # uses the committed snapshot under data/
Writes results/validation.json and results/validation.png.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import data_loader as dl
from backtest import Config, run_backtest

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
START = "2007-01-03"     # PUT's continuous daily history starts here


def dividend_yield(close: pd.Series, adj: pd.Series) -> float:
    """Average annual dividend yield implied by total-return minus price-return growth."""
    years = (close.index[-1] - close.index[0]).days / 365.25
    return float(np.log((adj.iloc[-1] / adj.iloc[0]) / (close.iloc[-1] / close.iloc[0])) / years)


def monthly_returns(s: pd.Series) -> pd.Series:
    return s.resample("ME").last().pct_change().dropna()


def compare(model: pd.Series, index: pd.Series) -> dict:
    """Return, risk and tracking statistics of a model equity curve against an index."""
    common = model.index.intersection(index.index)
    m = model.loc[common]
    x = index.loc[common]
    m, x = m / m.iloc[0], x / x.iloc[0]
    years = (common[-1] - common[0]).days / 365.25
    cagr_m = float(m.iloc[-1] ** (1 / years) - 1)
    cagr_x = float(x.iloc[-1] ** (1 / years) - 1)
    rm, rx = monthly_returns(m), monthly_returns(x)
    both = pd.concat([rm, rx], axis=1).dropna()
    diff = both.iloc[:, 0] - both.iloc[:, 1]

    def max_dd(s):
        return float((s / s.cummax() - 1).min())

    return {
        "model_cagr": cagr_m, "index_cagr": cagr_x, "cagr_gap": cagr_m - cagr_x,
        "monthly_corr": float(both.corr().iloc[0, 1]),
        "tracking_error": float(diff.std() * np.sqrt(12)),
        "mean_monthly_gap": float(diff.mean()),
        "model_max_dd": max_dd(m), "index_max_dd": max_dd(x),
        "model_vol": float(both.iloc[:, 0].std() * np.sqrt(12)),
        "index_vol": float(both.iloc[:, 1].std() * np.sqrt(12)),
        "months": int(len(both)),
    }


def replicate(close, vix, rates, q, strategy, skew) -> pd.Series:
    cfg = Config(strategy=strategy, dte=30, alloc_pct=1.0, strike_method="% OTM", otm_pct=0.0,
                 pt_enabled=False, sl_enabled=False, commission=0.0, slippage_pct=0.0,
                 stock_cost_bps=0.0, q=q, skew=skew, start_capital=1.0)
    res = run_backtest(close, vix, cfg, rates=rates)
    return res["daily"]["equity"]


def run(data_dir: Path | None = None, start: str = START) -> tuple[dict, dict]:
    data_dir = data_dir or dl.DATA_DIR
    read = lambda g, n: pd.read_csv(data_dir / g / f"{n}.csv", parse_dates=["date"],
                                    index_col="date")
    spx = read("prices", "GSPC")["close"].loc[start:]
    spy = read("prices", "SPY").loc[start:]
    vix = (read("vol", "VIX")["close"] / 100.0)
    vix = vix.reindex(spx.index.union(vix.index)).sort_index().ffill().reindex(spx.index)
    irx = read("rates", "IRX")["close"] / 100.0
    rates = irx.reindex(spx.index.union(irx.index)).sort_index().ffill().reindex(spx.index)
    put_idx = read("cboe", "PUT")["close"].loc[start:]
    bxm_idx = read("cboe", "BXM")["close"].loc[start:]
    q = dividend_yield(spy["close"], spy["adj_close"])

    curves, stats = {}, {}
    for strategy, index, key in (("Short Put", put_idx, "PUT"), ("Covered Call", bxm_idx, "BXM")):
        for skew in (False, True):
            label = f"{key} model, {'smile' if skew else 'flat'}"
            eq = replicate(spx, vix, rates, q, strategy, skew)
            curves[label] = eq
            stats[label] = compare(eq, index)
        curves[f"{key} index"] = index

    end = min(c.index[-1] for c in curves.values())
    summary = {
        "period": f"{start} to {end:%Y-%m-%d}",
        "dividend_yield_used": q,
        "assumptions": "Monthly (30-day) at-the-money options on ^GSPC, held to expiry, no costs, "
                       "100% notional, cash at the 13-week T-bill. Flat: VIX prices every strike. "
                       "Smile: ATM vol backed out of the VIX through pricing.EQUITY_INDEX_SKEW.",
        "results": stats,
    }
    return summary, curves


def plot(curves: dict, summary: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    colors = {"index": "#111827", "flat": "#d97706", "smile": "#2563eb"}
    for ax, key, title in ((axes[0], "PUT", "Short put vs Cboe PUT index"),
                           (axes[1], "BXM", "Covered call vs Cboe BXM index")):
        for label, s in curves.items():
            if not label.startswith(key):
                continue
            kind = "index" if label.endswith("index") else label.split(", ")[1]
            s = s.loc[s.index >= summary["period"][:10]]
            s = s / s.iloc[0]
            stat = summary["results"].get(label)
            lab = label if stat is None else f"{label} ({stat['model_cagr']:.1%} a year)"
            if stat is None:
                first = next(v for k, v in summary["results"].items() if k.startswith(key))
                lab = f"{label} ({first['index_cagr']:.1%} a year)"
            ax.plot(s.index, s.values, color=colors[kind], lw=1.6 if kind == "index" else 1.2,
                    label=lab)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    fig.suptitle(f"Synthetic premiums vs real option prices, {summary['period']}", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    summary, curves = run()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(curves, summary, RESULTS / "validation.png")
    for label, s in summary["results"].items():
        print(f"{label:22s} model {s['model_cagr']:6.2%}  index {s['index_cagr']:6.2%}  "
              f"gap {s['cagr_gap']:+6.2%}  corr {s['monthly_corr']:.3f}  TE {s['tracking_error']:.2%}  "
              f"maxDD {s['model_max_dd']:.1%} vs {s['index_max_dd']:.1%}")


if __name__ == "__main__":
    main()
