"""Reproduce the README's backtest tables from the committed data snapshot.

Runs the default settings on every bundled underlying, then a few variants on
SPY (exit rules, costs, the smile, the five structures), and writes
results/backtests.json and results/spy_default.png.

Usage:  python scripts/report.py
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import data_loader as dl  # noqa: E402
from backtest import Config, STRATEGIES, run_backtest  # noqa: E402
from metrics import perf_stats, trade_stats  # noqa: E402

YEARS = 10


def window(market: dl.Market):
    end = market.close.index[-1]
    start = end - pd.Timedelta(days=round(365.25 * YEARS))
    cut = lambda s: None if s is None else s.loc[start:]
    return cut(market.close), cut(market.vol), cut(market.adj_close), cut(market.rate)


def summarise(res: dict) -> dict:
    d, t = res["daily"], res["trades"]
    ps, bs, ts = perf_stats(d["equity"], d["rate"]), perf_stats(d["bench"], d["rate"]), trade_stats(t)
    vrp = (t["entry_iv"] - t["rv_trade"]).dropna()
    return {
        "start": f"{d.index[0]:%Y-%m-%d}", "end": f"{d.index[-1]:%Y-%m-%d}",
        "cagr": ps["CAGR"], "vol": ps["Ann. vol"], "sharpe": ps["Sharpe"],
        "sortino": ps["Sortino"], "max_dd": ps["Max drawdown"], "calmar": ps["Calmar"],
        "daily_skew": ps["Skew (daily)"],
        "bench_cagr": bs["CAGR"], "bench_sharpe": bs["Sharpe"], "bench_max_dd": bs["Max drawdown"],
        "trades": ts.get("Trades", 0), "win_rate": ts.get("Win rate"),
        "avg_win": ts.get("Avg win $"), "avg_loss": ts.get("Avg loss $"),
        "profit_factor": ts.get("Profit factor"), "worst_trade": ts.get("Worst trade $"),
        "trade_skew": ts.get("Trade PnL skew"), "avg_days": ts.get("Avg days held"),
        "interest": res["summary"]["interest_$"],
        "option_pnl": float(t["pnl_$"].sum()) if len(t) else 0.0,
        "iv_minus_rv_mean": float(vrp.mean()) if len(vrp) else None,
        "iv_above_rv_share": float((vrp > 0).mean()) if len(vrp) else None,
        "exits": t["exit_reason"].value_counts().to_dict() if len(t) else {},
    }


def main() -> None:
    out = {"years": YEARS, "defaults": {}, "spy_variants": {}, "spy_structures": {}}
    for name, uni in dl.UNIVERSE.items():
        if not uni["snapshot"] or not dl.snapshot_available(uni["snapshot"]):
            continue
        m = dl.load_market(name, "snapshot")
        close, vol, adj, rate = window(m)
        cfg = Config(q=uni["q"], r=uni["r"], skew=uni["skew"])
        out["defaults"][uni["snapshot"]] = summarise(run_backtest(close, vol, cfg, adj, rate))
        if uni["snapshot"] == "SPY":
            spy = (close, vol, adj, rate, cfg)

    close, vol, adj, rate, base = spy
    vix30 = vol[30] if isinstance(vol, pd.DataFrame) else vol
    variants = {
        "default": (base, vol),
        "hold_to_expiry": (dataclasses.replace(base, pt_enabled=False, sl_enabled=False), vol),
        "no_stop_loss": (dataclasses.replace(base, sl_enabled=False), vol),
        "no_costs": (dataclasses.replace(base, commission=0.0, slippage_pct=0.0), vol),
        "flat_vol_no_smile": (dataclasses.replace(base, skew=False), vol),
        "vix_only_no_term_structure": (base, vix30),
        "delta_0.16": (dataclasses.replace(base, target_delta=0.16), vol),
        "dte_45": (dataclasses.replace(base, dte=45), vol),
    }
    for key, (cfg, v) in variants.items():
        res = run_backtest(close, v, cfg, adj, rate)
        out["spy_variants"][key] = summarise(res)
        if key == "default":
            daily = res["daily"]
    for s in STRATEGIES:
        out["spy_structures"][s] = summarise(run_backtest(close, vol, dataclasses.replace(base, strategy=s), adj, rate))

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "backtests.json").write_text(json.dumps(out, indent=2, default=float) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(daily.index, daily["bench"] / daily["bench"].iloc[0], color="#6b7280", lw=1.2,
            label="SPY buy and hold (total return)")
    ax.plot(daily.index, daily["equity"] / daily["equity"].iloc[0], color="#0f766e", lw=1.6,
            label="Short strangle, default rules")
    tbill = np.exp(np.cumsum(daily["rate"].shift(1).fillna(0) * daily.index.to_series().diff().dt.days.fillna(0) / 365))
    ax.plot(daily.index, tbill, color="#b45309", lw=1.0, ls="--", label="T-bills")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("SPY, 30-day 0.30-delta short strangle with a 50% target and 2x stop, last 10 years")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "spy_default.png", dpi=140)
    plt.close(fig)

    for k, v in out["defaults"].items():
        print(f"{k:5s} CAGR {v['cagr']:6.2%} Sharpe {v['sharpe']:5.2f} maxDD {v['max_dd']:6.1%} win {v['win_rate']:.0%} "
              f"trades {v['trades']} | B&H {v['bench_cagr']:6.2%} | interest ${v['interest']:,.0f} options ${v['option_pnl']:,.0f}")
    for k, v in out["spy_variants"].items():
        print(f"SPY {k:28s} CAGR {v['cagr']:6.2%} Sharpe {v['sharpe']:5.2f} maxDD {v['max_dd']:6.1%} "
              f"options ${v['option_pnl']:,.0f} trades {v['trades']}")
    for k, v in out["spy_structures"].items():
        print(f"SPY {k:28s} CAGR {v['cagr']:6.2%} Sharpe {v['sharpe']:5.2f} maxDD {v['max_dd']:6.1%} "
              f"win {v['win_rate']:.0%} options ${v['option_pnl']:,.0f}")


if __name__ == "__main__":
    main()
