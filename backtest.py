"""
backtest.py: event-driven daily backtest engine for short-option strategies.

Design in one paragraph
-----------------------
Free historical option chains are hard to come by, so option premiums are
SYNTHESISED with Black-Scholes. The volatility input is a listed volatility
index (VIX / India VIX / VXN / GVZ, or for SPY the VIX9D, VIX and VIX3M term
structure) or trailing realised volatility times a user-set markup.
Optionally a volatility smile prices every strike at its own volatility
(pricing.EQUITY_INDEX_SKEW). A position is opened at the model premium, marked
to market every day with that day's volatility, and settled at intrinsic
value at expiry (or closed early on a profit target or stop loss). Equity is
always cash plus the market value of open positions, so the equity curve, the
trade log and the cash account are consistent by construction.

Accounting conventions
----------------------
* A "package" is the set of option legs of one trade, valued long-positive:
  package_value = sum of qty_i * price_i   (qty -1 = short one contract).
  A net-credit structure therefore has NEGATIVE package value.
* entry credit  C0 = -package_value(entry)  > 0 for short-premium trades.
* cost to close  = -package_value(t): what you would pay to buy it back.
* profit target: close when cost_to_close <= (1 - pt) * C0
  stop loss:     close when cost_to_close >= sl_mult * C0
* Everything is settled per contract * MULTIPLIER (100). Covered calls also
  carry 100 shares of stock per contract, liquidated when the trade closes
  so the account is flat between trades and per-trade P&L is unambiguous.
  The shares earn the dividend yield q while held and pay stock_cost_bps of
  their value each time they are bought or sold.
* Cash earns the risk-free rate every day: a rate series when one is given
  (for example the 13-week T-bill), otherwise the constant cfg.r. Interest is
  tracked separately and is not part of any trade's P&L, so trade statistics
  measure the options alone while the equity curve includes the interest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pricing import (bs_price, strike_from_delta, realized_vol, realized_vol_over,
                     CALENDAR_DAYS, SkewParams, EQUITY_INDEX_SKEW, smile_price,
                     strike_from_delta_smile, atm_vol_from_index, index_vol_at)

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
    stock_cost_bps: float = 1.0        # covered-call stock trades, per side
    r: float = 0.04                    # used when no rate series is supplied
    q: float = 0.013
    iv_mode: str = "Vol index"         # "Vol index" | "Realised vol x markup"
    rv_mult: float = 1.10
    rv_window: int = 21
    start_capital: float = 100_000.0
    fractional: bool = True            # allow fractional contracts
    skew: bool = False                 # price each strike off a volatility smile
    skew_params: SkewParams = EQUITY_INDEX_SKEW
    interest_on_cash: bool = True      # cash earns the risk-free rate
    stock_dividends: bool = True       # covered-call shares earn the dividend yield q


@dataclass
class Position:
    legs: list                         # [(kind, K, qty), ...] qty per contract
    contracts: float
    shares: float                      # stock shares per contract (covered call only)
    entry_date: pd.Timestamp
    expiry: pd.Timestamp
    credit0: float                     # per-contract net credit, index points
    entry_iv: float                    # ATM volatility used at entry
    s_entry: float
    equity_before: float
    costs: float = 0.0                 # running $ costs charged to this trade
    interest: float = 0.0              # interest earned on cash while open (excluded from P&L)
    dividends: float = 0.0             # dividends on covered-call shares (part of P&L)


# ---------------------------------------------------------------------------
# Trade construction
# ---------------------------------------------------------------------------

def _params(cfg: Config) -> SkewParams | None:
    return cfg.skew_params if cfg.skew else None


def build_legs(cfg: Config, S: float, sigma: float, T: float,
               r: float | None = None) -> tuple[list, float]:
    """
    Return (option legs, stock shares per contract) for cfg.strategy.
    sigma is the at-the-money volatility; with the smile on, delta strikes are
    solved at each strike's own volatility.
    """
    r = cfg.r if r is None else r
    params = _params(cfg)

    def delta_strike(kind):
        if params is None:
            return strike_from_delta(S, cfg.target_delta, T, sigma, r, cfg.q, kind)
        return strike_from_delta_smile(S, cfg.target_delta, T, sigma, r, cfg.q, kind, params)

    def short_call_K():
        return delta_strike("call") if cfg.strike_method == "Delta" else S * (1 + cfg.otm_pct)

    def short_put_K():
        return delta_strike("put") if cfg.strike_method == "Delta" else S * (1 - cfg.otm_pct)

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


def package_value(legs, S: float, T: float, sigma: float, r: float, q: float,
                  params: SkewParams | None = None) -> float:
    """Mark of the option legs of ONE contract, long-positive, index points.
    sigma is the ATM volatility; params=None prices every leg at sigma."""
    if params is None:
        return float(sum(qty * bs_price(S, K, T, sigma, r, q, kind)
                         for kind, K, qty in legs))
    return float(sum(qty * smile_price(S, K, T, sigma, r, q, kind, params)
                     for kind, K, qty in legs))


def legs_label(legs) -> str:
    return " / ".join(f"{'+' if q > 0 else '-'}{k[0].upper()} {K:,.0f}"
                      for k, K, q in legs)


# ---------------------------------------------------------------------------
# Volatility input helpers
# ---------------------------------------------------------------------------

def _thirty_day_column(iv: pd.DataFrame):
    cols = [float(c) for c in iv.columns]
    return iv.columns[int(np.argmin([abs(c - 30.0) for c in cols]))]


def headline_vol(iv) -> pd.Series:
    """The 30-day volatility series (the single series, or the column nearest 30 days)."""
    if isinstance(iv, pd.DataFrame):
        return iv[_thirty_day_column(iv)]
    return iv


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def run_backtest(prices: pd.Series, iv, cfg: Config, bench: pd.Series | None = None,
                 rates: pd.Series | None = None) -> dict:
    """
    prices : daily closes the options are written on (raw, not dividend-adjusted)
    iv     : volatility input in decimals, used for entry pricing and daily marks.
             A Series is one 30-day index applied to every maturity. A DataFrame
             with numeric columns (tenor in calendar days, e.g. 9, 30, 93) is a
             term structure, interpolated in total variance at each option's
             remaining life (pricing.index_vol_at).
    bench  : optional total-return series (adjusted closes) for the buy-and-hold
             benchmark; defaults to prices.
    rates  : optional annualised risk-free rate in decimals (e.g. T-bill / 100),
             used for interest on cash and for discounting; defaults to cfg.r.

    Returns {"daily": DataFrame, "trades": DataFrame, "warnings": list, "summary": dict}.
    """
    rv = realized_vol(prices, cfg.rv_window)
    iv30 = headline_vol(iv)
    df = pd.DataFrame({"S": prices, "iv": iv30, "rv": rv}).dropna(subset=["S", "iv"])
    dates = df.index
    if len(dates) < 60:
        raise ValueError("Not enough overlapping price/vol history to backtest.")

    if isinstance(iv, pd.DataFrame):
        tenors = [float(c) for c in iv.columns]
        iv_tab = iv.reindex(dates).ffill()

        def index_vol(t, days):
            return index_vol_at(tenors, iv_tab.loc[t].to_numpy(dtype=float), days)
    else:
        def index_vol(t, days):
            return float(df.at[t, "iv"])

    params = _params(cfg)
    convert_index = params is not None and cfg.iv_mode == "Vol index"

    def atm_vol(t, days):
        v = index_vol(t, days)
        if convert_index:
            v = atm_vol_from_index(v, max(days, 1) / CALENDAR_DAYS, params)
        return v

    if rates is not None:
        rate_tab = rates.reindex(dates.union(rates.index)).sort_index().ffill().reindex(dates)
        rate_tab = rate_tab.fillna(cfg.r)

        def rate(t):
            return float(rate_tab.at[t])
    else:
        def rate(t):
            return cfg.r

    cash = cfg.start_capital
    pos: Position | None = None
    flat_days = cfg.cooldown          # allow entry on day one
    warnings: list[str] = []
    skipped_too_small = 0
    total_interest = 0.0

    eq_rows, trade_rows = [], []

    def days_left(p: Position, t: pd.Timestamp) -> int:
        return max((p.expiry - t).days, 0)

    def mark(p: Position, S: float, t: pd.Timestamp) -> float:
        """$ market value of the whole position (options + stock)."""
        d = days_left(p, t)
        opt = package_value(p.legs, S, d / CALENDAR_DAYS, atm_vol(t, d), rate(t), cfg.q, params)
        return opt * MULTIPLIER * p.contracts + p.shares * p.contracts * S

    def close(p: Position, t: pd.Timestamp, S: float, at_expiry: bool, reason: str):
        nonlocal cash
        d = 0 if at_expiry else days_left(p, t)
        opt_val = package_value(p.legs, S, d / CALENDAR_DAYS, atm_vol(t, max(d, 1)),
                                rate(t), cfg.q, params)
        gross = abs(opt_val) * MULTIPLIER * p.contracts
        stock_value = p.shares * p.contracts * S
        exit_cost = (cfg.commission * len(p.legs) * p.contracts
                     + cfg.slippage_pct * gross
                     + cfg.stock_cost_bps / 1e4 * stock_value)
        # Buy back the options, sell any stock, pay costs.
        cash += opt_val * MULTIPLIER * p.contracts
        cash += stock_value
        cash -= exit_cost
        p.costs += exit_cost

        pnl = cash - p.equity_before - p.interest
        trade_rows.append({
            "entry_date": p.entry_date, "exit_date": t,
            "days_held": (t - p.entry_date).days,
            "strategy": cfg.strategy, "legs": legs_label(p.legs),
            "contracts": round(p.contracts, 3),
            "credit_$": p.credit0 * MULTIPLIER * p.contracts,
            "exit_value_$": -opt_val * MULTIPLIER * p.contracts,
            "costs_$": p.costs, "dividends_$": p.dividends, "pnl_$": pnl,
            "pnl_pct_credit": pnl / (p.credit0 * MULTIPLIER * p.contracts)
                              if p.credit0 > 1e-9 else np.nan,
            "entry_iv": p.entry_iv,
            "rv_trade": realized_vol_over(prices.loc[p.entry_date:t]),
            "underlying_ret": S / p.s_entry - 1.0,
            "exit_reason": reason,
        })

    def try_open(t: pd.Timestamp, S: float):
        nonlocal cash, pos, skipped_too_small
        expiry_target = t + pd.Timedelta(days=cfg.dte)
        if expiry_target > dates[-1]:
            return                                   # would be truncated: skip
        loc = dates.searchsorted(expiry_target, side="right") - 1
        expiry = dates[loc]
        if expiry <= t:
            return
        d = (expiry - t).days
        T = d / CALENDAR_DAYS
        sigma = atm_vol(t, d)
        r_t = rate(t)

        legs, shares = build_legs(cfg, S, sigma, T, r_t)
        equity_now = cash
        contracts = cfg.alloc_pct * equity_now / (S * MULTIPLIER)
        if not cfg.fractional:
            contracts = float(np.floor(contracts))
        if contracts <= 0:
            skipped_too_small += 1
            return

        opt_val = package_value(legs, S, T, sigma, r_t, cfg.q, params)
        credit0 = -opt_val                            # >0 for net-credit trades
        leg_prices = [(bs_price(S, K, T, sigma, r_t, cfg.q, kind) if params is None
                       else smile_price(S, K, T, sigma, r_t, cfg.q, kind, params))
                      for kind, K, _ in legs]
        gross = sum(abs(float(p)) for p in leg_prices) * MULTIPLIER * contracts
        stock_value = shares * contracts * S
        entry_cost = (cfg.commission * len(legs) * contracts + cfg.slippage_pct * gross
                      + cfg.stock_cost_bps / 1e4 * stock_value)

        equity_before = cash
        cash -= opt_val * MULTIPLIER * contracts      # receive credit into cash
        cash -= stock_value                           # buy stock (covered call)
        cash -= entry_cost

        pos = Position(legs=legs, contracts=contracts, shares=shares,
                       entry_date=t, expiry=expiry, credit0=credit0,
                       entry_iv=sigma, s_entry=S,
                       equity_before=equity_before, costs=entry_cost)

    # ---------------- main daily loop ----------------
    prev_t, prev_S = None, None
    for t in dates:
        S = float(df.at[t, "S"])

        # Interest on cash and dividends on held shares since the last close.
        if prev_t is not None:
            dt_years = (t - prev_t).days / CALENDAR_DAYS
            if cfg.interest_on_cash:
                interest = cash * (np.exp(rate(prev_t) * dt_years) - 1.0)
                cash += interest
                total_interest += interest
                if pos is not None:
                    pos.interest += interest
            if pos is not None and pos.shares and cfg.stock_dividends:
                div = pos.shares * pos.contracts * prev_S * cfg.q * dt_years
                cash += div
                pos.dividends += div

        if pos is not None:
            if t >= pos.expiry:
                close(pos, t, S, at_expiry=True, reason="Expiry")
                pos, flat_days = None, 0
            elif pos.credit0 > 1e-9:
                d = days_left(pos, t)
                cost_to_close = -package_value(pos.legs, S, d / CALENDAR_DAYS,
                                               atm_vol(t, d), rate(t), cfg.q, params)
                if cfg.pt_enabled and cost_to_close <= (1 - cfg.pt_pct) * pos.credit0:
                    close(pos, t, S, False, "Profit target")
                    pos, flat_days = None, 0
                elif cfg.sl_enabled and cost_to_close >= cfg.sl_mult * pos.credit0:
                    close(pos, t, S, False, "Stop loss")
                    pos, flat_days = None, 0

        if pos is None:
            if flat_days >= cfg.cooldown:
                try_open(t, S)
            flat_days += 1

        mv = mark(pos, S, t) if pos is not None else 0.0
        eq_rows.append({"date": t, "equity": cash + mv, "cash": cash,
                        "S": S, "iv": float(df.at[t, "iv"]), "rv": df.at[t, "rv"],
                        "rate": rate(t), "interest_cum": total_interest,
                        "in_market": pos is not None})
        prev_t, prev_S = t, S

    daily = pd.DataFrame(eq_rows).set_index("date")
    if bench is not None:
        b = bench.reindex(daily.index).ffill().bfill()
        daily["bench"] = cfg.start_capital * b / b.iloc[0]
    else:
        daily["bench"] = cfg.start_capital * daily["S"] / daily["S"].iloc[0]

    trades = pd.DataFrame(trade_rows)
    if skipped_too_small:
        warnings.append(f"{skipped_too_small} entries skipped: equity too small "
                        f"for one whole contract (enable fractional contracts).")
    if len(trades) < 5:
        warnings.append("Fewer than 5 completed trades: statistics will be noisy.")

    open_unrealized = 0.0
    if pos is not None:
        open_unrealized = daily["equity"].iloc[-1] - pos.equity_before - pos.interest
    summary = {"interest_$": total_interest, "open_unrealized_$": open_unrealized}
    return {"daily": daily, "trades": trades, "warnings": warnings, "summary": summary}


# ---------------------------------------------------------------------------
# Payoff curves for the strategy diagram (pure functions, no state)
# ---------------------------------------------------------------------------

def payoff_curves(cfg: Config, S0: float, sigma: float) -> dict:
    """
    Expiry payoff and entry-day (T-0) P&L of one contract of the configured
    structure, per underlying price; feeds the payoff diagram. sigma is the
    ATM volatility.
    """
    T = cfg.dte / CALENDAR_DAYS
    params = _params(cfg)
    legs, shares = build_legs(cfg, S0, sigma, T)
    credit0 = -package_value(legs, S0, T, sigma, cfg.r, cfg.q, params)

    grid = np.linspace(0.70 * S0, 1.30 * S0, 241)
    expiry_pnl, t0_pnl = [], []
    for s in grid:
        v_exp = package_value(legs, s, 0.0, sigma, cfg.r, cfg.q, params)
        v_now = package_value(legs, s, T, sigma, cfg.r, cfg.q, params)
        stock = shares * (s - S0)
        expiry_pnl.append((credit0 + v_exp) * MULTIPLIER + stock)
        t0_pnl.append((credit0 + v_now) * MULTIPLIER + stock)

    return {"grid": grid, "expiry": np.array(expiry_pnl),
            "t0": np.array(t0_pnl), "legs": legs,
            "credit_$": credit0 * MULTIPLIER, "label": legs_label(legs)}
