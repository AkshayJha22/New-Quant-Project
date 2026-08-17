"""
smoke_test.py — run `python smoke_test.py` before trusting anything.

Checks, in order:
1. norm_cdf against known reference values; norm_ppf round-trips norm_cdf.
2. Black–Scholes against the classic textbook value and put–call parity.
3. strike_from_delta round-trips through bs_delta.
4. A full backtest on the deterministic synthetic market: engine runs,
   accounting is internally consistent, and one trade's PnL is re-derived
   from its own log fields.
"""

import numpy as np

from pricing import (norm_cdf, norm_ppf, bs_price, bs_delta,
                     strike_from_delta, realized_vol)
from data_loader import synthetic_market
from backtest import Config, run_backtest, MULTIPLIER
from metrics import perf_stats, trade_stats


def check(name, ok):
    status = "ok " if ok else "FAIL"
    print(f"[{status}] {name}")
    assert ok, name


# 1 — normal distribution ----------------------------------------------------
check("N(0) = 0.5", abs(norm_cdf(0.0) - 0.5) < 1e-7)
check("N(1.96) ≈ 0.9750", abs(norm_cdf(1.96) - 0.9750021) < 1e-5)
check("N(-1) ≈ 0.15866", abs(norm_cdf(-1.0) - 0.1586553) < 1e-5)
for p, ref in [(0.975, 1.9599640), (0.95, 1.6448536), (0.01, -2.3263479)]:
    check(f"ppf({p}) ≈ {ref}", abs(float(norm_ppf(p)) - ref) < 1e-6)
xs = np.linspace(-4, 4, 81)
rt = np.max(np.abs(norm_ppf(norm_cdf(xs)) - xs))
check(f"ppf(cdf(x)) round-trip on ±4σ (max err {rt:.1e}; "
      "CDF approx error is tail-amplified)", rt < 5e-4)

# 2 — Black–Scholes ----------------------------------------------------------
c = float(bs_price(100, 100, 1.0, 0.20, r=0.05, q=0.0, kind="call"))
check(f"BS call(100,100,1y,20%,r=5%) = {c:.4f} ≈ 10.4506", abs(c - 10.4506) < 1e-3)

S, K, T, sig, r, q = 105.0, 98.0, 0.4, 0.27, 0.03, 0.012
call = float(bs_price(S, K, T, sig, r, q, "call"))
put = float(bs_price(S, K, T, sig, r, q, "put"))
parity = call - put - (S * np.exp(-q * T) - K * np.exp(-r * T))
check(f"put–call parity residual {parity:.2e}", abs(parity) < 1e-8)

check("T=0 call is intrinsic", float(bs_price(110, 100, 0.0, 0.2, kind="call")) == 10.0)
check("T=0 OTM put is 0", float(bs_price(110, 100, 0.0, 0.2, kind="put")) == 0.0)

# 3 — strike from delta ------------------------------------------------------
for kind in ("call", "put"):
    Kd = strike_from_delta(100.0, 0.30, 30 / 365, 0.22, r=0.04, q=0.01, kind=kind)
    d = abs(float(bs_delta(100.0, Kd, 30 / 365, 0.22, r=0.04, q=0.01, kind=kind)))
    check(f"0.30-delta {kind}: K={Kd:.2f}, |delta| round-trip {d:.4f}",
          abs(d - 0.30) < 1e-4)
check("0.30d put strikes below spot",
      strike_from_delta(100, 0.30, 30 / 365, 0.22, kind="put") < 100)
check("0.30d call strikes above spot",
      strike_from_delta(100, 0.30, 30 / 365, 0.22, kind="call") > 100)

# 4 — full engine on the synthetic market ------------------------------------
prices, vol_idx = synthetic_market(n_years=12, seed=7)
iv = vol_idx

for strat in ("Short Strangle", "Iron Condor", "Covered Call"):
    cfg = Config(strategy=strat, dte=30, alloc_pct=0.5,
                 pt_enabled=True, sl_enabled=True,
                 commission=0.65, slippage_pct=0.02)
    res = run_backtest(prices, iv, cfg)
    daily, trades = res["daily"], res["trades"]

    check(f"{strat}: equity all finite & positive",
          bool(np.isfinite(daily["equity"]).all() and (daily["equity"] > 0).all()))
    check(f"{strat}: {len(trades)} trades completed", len(trades) > 30)

    # Accounting identity: total PnL booked in trades + any open MTM
    # must reconcile with the change in equity.
    open_mtm = daily["equity"].iloc[-1] - daily["cash"].iloc[-1]
    recon = trades["pnl_$"].sum() + open_mtm - (daily["equity"].iloc[-1]
                                                - cfg.start_capital)
    check(f"{strat}: trade log reconciles with equity (residual "
          f"${recon:,.4f})", abs(recon) < 1e-6)

    # Exit-rule sanity: profit-target exits must be profitable before costs.
    pt = trades[trades["exit_reason"] == "Profit target"]
    if len(pt):
        check(f"{strat}: all {len(pt)} profit-target exits have "
              "credit > buy-back", bool((pt["credit_$"] > pt["exit_value_$"]).all()))

# Re-derive one expiry trade's PnL from its own logged fields.
cfg = Config(strategy="Short Straddle", pt_enabled=False, sl_enabled=False,
             commission=0.0, slippage_pct=0.0)
res = run_backtest(prices, iv, cfg)
tr = res["trades"]
exp = tr[tr["exit_reason"] == "Expiry"].iloc[0]
rebuilt = exp["credit_$"] - exp["exit_value_$"] - exp["costs_$"]
check(f"no-cost expiry trade: pnl {exp['pnl_$']:.2f} == credit − intrinsic "
      f"{rebuilt:.2f}", abs(exp["pnl_$"] - rebuilt) < 1e-6)

stats = perf_stats(res["daily"]["equity"], rf=0.04)
tstats = trade_stats(tr)
print(f"\nSynthetic short straddle, no costs: CAGR {stats['CAGR']:.1%}, "
      f"Sharpe {stats['Sharpe']:.2f}, MaxDD {stats['Max drawdown']:.1%}, "
      f"win rate {tstats['Win rate']:.0%}, trades {tstats['Trades']}")

print("\nALL TESTS PASSED")
