"""
backtest.py — event-driven daily backtest engine for short-option strategies.

Design in one paragraph
-----------------------
Because historical option chains aren't freely available, option premiums
are SYNTHESISED with Black–Scholes: the volatility input is either a listed
vol index (VIX / India VIX / VXN / GVZ) or trailing realised vol times a
user-set markup. A position is opened at the model premium, marked to
market every day by repricing with that day's vol, and settled at intrinsic
value at expiry (or closed early on a profit-target / stop-loss). Equity is
always cash + market value of open positions, so the equity curve, the
trade log and the cash account are consistent by construction.

Accounting conventions
----------------------
* A "package" is the set of option legs of one trade, valued long-positive:
  package_value = Σ qty_i · BS_price_i   (qty −1 = short one contract).
  A net-credit structure therefore has NEGATIVE package value.
* entry credit  C0 = −package_value(entry)  > 0 for short-premium trades.
* cost to close  = −package_value(t): what you would pay to buy it back.
* profit target: close when cost_to_close ≤ (1 − pt) · C0
  stop loss:     close when cost_to_close ≥ sl_mult · C0
  (the classic "take profit at 50%, stop at 2× credit" mechanics).
* Everything is settled per contract × MULTIPLIER (100). Covered calls also
  carry 100 shares of stock per contract, liquidated when the trade closes
  so the account is flat between trades and per-trade PnL is unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pricing import (bs_price, strike_from_delta, realized_vol,
                     realized_vol_over, CALENDAR_DAYS)

MULTIPLIER = 100

STRATEGIES = [
    "Short Put",
    "Short Straddle",
    "Short Strangle",
    "Iron Condor",
    "Covered Call",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    strategy: str = "Short Strangle"
    dte: int = 30                      # calendar days to expiry
    alloc_pct: float = 0.50            # notional per trade as fraction of equity
    strike_method: str = "Delta"       # "Delta" | "% OTM"
    target_delta: float = 0.30         # |delta| of short strikes (Delta method)
    otm_pct: float = 0.05              # distance of short strikes (% OTM method)
    wing_pct: float = 0.05             # iron-condor wing width as % of spot
    pt_enabled: bool = True
    pt_pct: float = 0.50               # close at 50% of max profit
    sl_enabled: bool = True
    sl_mult: float = 2.0               # stop at 2x credit received
    cooldown: int = 0                  # trading days to wait before re-entry
    commission: float = 0.65           # $ per contract per leg per side
    slippage_pct: float = 0.02         # fraction of premium lost per side
    r: float = 0.04
    q: float = 0.013
    iv_mode: str = "Vol index"         # "Vol index" | "Realised vol × markup"
    rv_mult: float = 1.10
    rv_window: int = 21
    start_capital: float = 100_000.0
    fractional: bool = True            # allow fractional contracts


@dataclass
class Position:
    legs: list                         # [(kind, K, qty), ...] qty per contract
    contracts: float
    shares: float                      # stock shares (covered call only)
    entry_date: pd.Timestamp
    expiry: pd.Timestamp
    credit0: float                     # per-contract net credit, index points
    entry_iv: float
    s_entry: float
    equity_before: float
    costs: float = 0.0                 # running $ costs charged to this trade


# ---------------------------------------------------------------------------
# Trade construction
# ---------------------------------------------------------------------------

def build_legs(cfg: Config, S: float, sigma: float, T: float) -> tuple[list, float]:
    """Return (option legs, stock shares per contract) for cfg.strategy."""

    def short_call_K():
        if cfg.strike_method == "Delta":
            return strike_from_delta(S, cfg.target_delta, T, sigma, cfg.r, cfg.q, "call")
        return S * (1 + cfg.otm_pct)

    def short_put_K():
        if cfg.strike_method == "Delta":
            return strike_from_delta(S, cfg.target_delta, T, sigma, cfg.r, cfg.q, "put")
        return S * (1 - cfg.otm_pct)

    st = cfg.strategy
    if st == "Short Put":
        return [("put", short_put_K(), -1)], 0.0
    if st == "Short Straddle":
        return [("call", S, -1), ("put", S, -1)], 0.0
    if st == "Short Strangle":
        return [("call", short_call_K(), -1), ("put", short_put_K(), -1)], 0.0
    if st == "Iron Condor":
        kc, kp = short_call_K(), short_put_K()
        w = cfg.wing_pct * S
        return [("call", kc, -1), ("call", kc + w, +1),
                ("put",  kp, -1), ("put",  kp - w, +1)], 0.0
    if st == "Covered Call":
        return [("call", short_call_K(), -1)], float(MULTIPLIER)
    raise ValueError(f"Unknown strategy {st!r}")


def package_value(legs, S: float, T: float, sigma: float, r: float, q: float) -> float:
    """Mark of the option legs of ONE contract, long-positive, index points."""
    return float(sum(qty * bs_price(S, K, T, sigma, r, q, kind)
                     for kind, K, qty in legs))


def legs_label(legs) -> str:
    return " / ".join(f"{'+' if q > 0 else '-'}{k[0].upper()} {K:,.0f}"
                      for k, K, q in legs)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def run_backtest(prices: pd.Series, iv: pd.Series, cfg: Config) -> dict:
    """
    prices : daily closes
    iv     : implied-vol series used for BOTH entry pricing and daily MTM
             (vol index / 100, or trailing RV × markup — chosen upstream)

    Returns {"daily": DataFrame, "trades": DataFrame, "warnings": list}.
    """
    rv = realized_vol(prices, cfg.rv_window)
    df = pd.DataFrame({"S": prices, "iv": iv, "rv": rv}).dropna(subset=["S", "iv"])
    dates = df.index
    if len(dates) < 60:
        raise ValueError("Not enough overlapping price/vol history to backtest.")

    cash = cfg.start_capital
    pos: Position | None = None
    flat_days = cfg.cooldown          # allow entry on day one
    warnings: list[str] = []
    skipped_too_small = 0

    eq_rows, trade_rows = [], []

    def mark(p: Position, S: float, t: pd.Timestamp, sigma: float) -> float:
        """$ market value of the whole position (options + stock)."""
        T = max((p.expiry - t).days, 0) / CALENDAR_DAYS
        opt = package_value(p.legs, S, T, sigma, cfg.r, cfg.q)
        return opt * MULTIPLIER * p.contracts + p.shares * p.contracts * S

    def close(p: Position, t: pd.Timestamp, S: float, sigma: float,
              at_expiry: bool, reason: str):
        nonlocal cash
        T = 0.0 if at_expiry else max((p.expiry - t).days, 0) / CALENDAR_DAYS
        opt_val = package_value(p.legs, S, T, sigma, cfg.r, cfg.q)
        gross = abs(opt_val) * MULTIPLIER * p.contracts
        exit_cost = (cfg.commission * len(p.legs) * p.contracts
                     + cfg.slippage_pct * gross)
        # Buy back the options, sell any stock, pay costs.
        cash += opt_val * MULTIPLIER * p.contracts
        cash += p.shares * p.contracts * S
        cash -= exit_cost
        p.costs += exit_cost

        pnl = cash - p.equity_before
        trade_rows.append({
            "entry_date": p.entry_date, "exit_date": t,
            "days_held": (t - p.entry_date).days,
            "strategy": cfg.strategy, "legs": legs_label(p.legs),
            "contracts": round(p.contracts, 3),
            "credit_$": p.credit0 * MULTIPLIER * p.contracts,
            "exit_value_$": -opt_val * MULTIPLIER * p.contracts,
            "costs_$": p.costs, "pnl_$": pnl,
            "pnl_pct_credit": pnl / (p.credit0 * MULTIPLIER * p.contracts)
                              if p.credit0 > 1e-9 else np.nan,
            "entry_iv": p.entry_iv,
            "rv_trade": realized_vol_over(prices.loc[p.entry_date:t]),
            "underlying_ret": S / p.s_entry - 1.0,
            "exit_reason": reason,
        })

    def try_open(t: pd.Timestamp, S: float, sigma: float):
        nonlocal cash, pos, skipped_too_small
        expiry_target = t + pd.Timedelta(days=cfg.dte)
        if expiry_target > dates[-1]:
            return                                   # would be truncated: skip
        loc = dates.searchsorted(expiry_target, side="right") - 1
        expiry = dates[loc]
        if expiry <= t:
            return
        T = (expiry - t).days / CALENDAR_DAYS

        legs, shares = build_legs(cfg, S, sigma, T)
        equity_now = cash
        contracts = cfg.alloc_pct * equity_now / (S * MULTIPLIER)
        if not cfg.fractional:
            contracts = float(np.floor(contracts))
        if contracts <= 0:
            skipped_too_small += 1
            return

        opt_val = package_value(legs, S, T, sigma, cfg.r, cfg.q)
        credit0 = -opt_val                            # >0 for net-credit trades
        gross = sum(abs(bs_price(S, K, T, sigma, cfg.r, cfg.q, kind))
                    for kind, K, _ in legs) * MULTIPLIER * contracts
        entry_cost = cfg.commission * len(legs) * contracts + cfg.slippage_pct * gross

        equity_before = cash
        cash -= opt_val * MULTIPLIER * contracts      # receive credit into cash
        cash -= shares * contracts * S                # buy stock (covered call)
        cash -= entry_cost

        pos = Position(legs=legs, contracts=contracts, shares=shares,
                       entry_date=t, expiry=expiry, credit0=credit0,
                       entry_iv=sigma, s_entry=S,
                       equity_before=equity_before, costs=entry_cost)

    # ---------------- main daily loop ----------------
    for t in dates:
        S, sigma = float(df.at[t, "S"]), float(df.at[t, "iv"])

        if pos is not None:
            if t >= pos.expiry:
                close(pos, t, S, sigma, at_expiry=True, reason="Expiry")
                pos, flat_days = None, 0
            elif pos.credit0 > 1e-9:
                cost_to_close = -package_value(
                    pos.legs, S, max((pos.expiry - t).days, 0) / CALENDAR_DAYS,
                    sigma, cfg.r, cfg.q)
                if cfg.pt_enabled and cost_to_close <= (1 - cfg.pt_pct) * pos.credit0:
                    close(pos, t, S, sigma, False, "Profit target")
                    pos, flat_days = None, 0
                elif cfg.sl_enabled and cost_to_close >= cfg.sl_mult * pos.credit0:
                    close(pos, t, S, sigma, False, "Stop loss")
                    pos, flat_days = None, 0

        if pos is None:
            if flat_days >= cfg.cooldown:
                try_open(t, S, sigma)
            flat_days += 1

        mv = mark(pos, S, t, sigma) if pos is not None else 0.0
        eq_rows.append({"date": t, "equity": cash + mv, "cash": cash,
                        "S": S, "iv": sigma, "rv": df.at[t, "rv"],
                        "in_market": pos is not None})

    daily = pd.DataFrame(eq_rows).set_index("date")
    daily["bench"] = cfg.start_capital * daily["S"] / daily["S"].iloc[0]

    trades = pd.DataFrame(trade_rows)
    if skipped_too_small:
        warnings.append(f"{skipped_too_small} entries skipped: equity too small "
                        f"for one whole contract (enable fractional contracts).")
    if len(trades) < 5:
        warnings.append("Fewer than 5 completed trades — statistics will be noisy.")

    return {"daily": daily, "trades": trades, "warnings": warnings}


# ---------------------------------------------------------------------------
# Payoff curves for the strategy diagram (pure functions, no state)
# ---------------------------------------------------------------------------

def payoff_curves(cfg: Config, S0: float, sigma: float) -> dict:
    """
    Expiry payoff and entry-day (T-0) PnL of one contract of the configured
    structure, per underlying price — feeds the payoff diagram.
    """
    T = cfg.dte / CALENDAR_DAYS
    legs, shares = build_legs(cfg, S0, sigma, T)
    credit0 = -package_value(legs, S0, T, sigma, cfg.r, cfg.q)

    grid = np.linspace(0.70 * S0, 1.30 * S0, 241)
    expiry_pnl, t0_pnl = [], []
    for s in grid:
        v_exp = package_value(legs, s, 0.0, sigma, cfg.r, cfg.q)
        v_now = package_value(legs, s, T, sigma, cfg.r, cfg.q)
        stock = shares * (s - S0)
        expiry_pnl.append((credit0 + v_exp) * MULTIPLIER + stock)
        t0_pnl.append((credit0 + v_now) * MULTIPLIER + stock)

    return {"grid": grid, "expiry": np.array(expiry_pnl),
            "t0": np.array(t0_pnl), "legs": legs,
            "credit_$": credit0 * MULTIPLIER, "label": legs_label(legs)}
