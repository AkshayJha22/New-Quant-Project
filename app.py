"""
app.py — Theta Lab: a systematic option-selling backtester.

Streamlit entry point. This file only does UI and orchestration; all maths
lives in pricing.py / backtest.py / metrics.py and all figures in plots.py,
so every layer can be tested without a browser (see smoke_test.py).

Run with:  streamlit run app.py
"""

from __future__ import annotations

import dataclasses
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

import data_loader as dl
import plots
from backtest import Config, STRATEGIES, run_backtest, payoff_curves
from metrics import (perf_stats, trade_stats, drawdown_series,
                     monthly_return_table, time_in_market)
from pricing import realized_vol

st.set_page_config(page_title="Theta Lab", page_icon="⏳", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&display=swap');
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }
[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; }
div[data-testid="stSidebarHeader"] { padding-bottom: 0; }
</style>
""", unsafe_allow_html=True)

PLOTLY_CFG = {"displayModeBar": False}


# ---------------------------------------------------------------------------
# Cached data / compute wrappers (all heavy lifting is memoised)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def cached_prices(ticker: str, start: date) -> pd.Series:
    return dl.fetch_prices(ticker, start=start)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_vol_index(vol_ticker: str | None, index_key: tuple) -> pd.Series | None:
    idx = pd.DatetimeIndex(list(index_key))
    return dl.fetch_vol_index(vol_ticker, idx)


@st.cache_data(show_spinner=False)
def cached_backtest(prices: pd.Series, iv: pd.Series, cfg: Config) -> dict:
    return run_backtest(prices, iv, cfg)


@st.cache_data(show_spinner=False)
def cached_sensitivity(prices: pd.Series, iv: pd.Series, cfg: Config) -> pd.DataFrame:
    base = cfg.slippage_pct if cfg.slippage_pct > 0 else 0.02
    grid = sorted({0.0, 0.5 * base, base, 1.5 * base, 2 * base, 3 * base})
    rows = []
    for s in grid:
        res = run_backtest(prices, iv, dataclasses.replace(cfg, slippage_pct=s))
        ps = perf_stats(res["daily"]["equity"], rf=cfg.r)
        rows.append({"slippage_pct": s, "sharpe": ps["Sharpe"], "cagr": ps["CAGR"]})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def cached_comparison(prices: pd.Series, iv: pd.Series, cfg: Config) -> dict:
    out = {}
    for name in STRATEGIES:
        res = run_backtest(prices, iv, dataclasses.replace(cfg, strategy=name))
        out[name] = res["daily"]["equity"]
    return out


# ---------------------------------------------------------------------------
# Sidebar — every assumption is a visible, adjustable input
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⏳ Theta Lab")
    st.caption("Systematic option-selling backtester")

    st.markdown("#### Market")
    uni_name = st.selectbox("Underlying", list(dl.UNIVERSE.keys()), index=0)
    uni = dl.UNIVERSE[uni_name]
    ticker = uni["ticker"]
    if ticker is None:
        ticker = st.text_input("Yahoo Finance ticker", value="AAPL").strip() or "AAPL"

    c1, c2 = st.columns(2)
    start_date = c1.date_input("Start", value=date.today() - timedelta(days=365 * 10))
    end_date = c2.date_input("End", value=date.today())
    start_capital = st.number_input("Starting capital ($)", 1_000, 10_000_000,
                                    100_000, step=10_000)

    st.markdown("#### Strategy")
    strategy = st.selectbox("Structure", STRATEGIES, index=2)
    dte = st.slider("Days to expiry (DTE)", 7, 90, 30)
    strike_method = st.radio("Strike selection", ["Delta", "% OTM"], horizontal=True)
    if strike_method == "Delta":
        target_delta = st.slider("Short-strike |delta|", 0.05, 0.50, 0.30, 0.05)
        otm_pct = 0.05
    else:
        otm_pct = st.slider("Short strikes % OTM", 0.01, 0.15, 0.05, 0.01)
        target_delta = 0.30
    wing_pct = (st.slider("Condor wing width (% of spot)", 0.01, 0.15, 0.05, 0.01)
                if strategy == "Iron Condor" else 0.05)
    alloc_pct = st.slider("Notional per trade (% of equity)", 10, 100, 50, 5) / 100

    st.markdown("#### Volatility & rates")
    st.caption("Premiums are synthesised with Black–Scholes from this vol input.")
    iv_options = []
    if uni["vol_index"] is not None:
        iv_options.append("Vol index")
    iv_options.append("Realised vol × markup")
    iv_mode = st.radio("Implied-vol source", iv_options, horizontal=True)
    rv_mult = st.slider("RV markup (proxy for the vol premium)", 0.80, 1.60, 1.10, 0.05,
                        disabled=(iv_mode == "Vol index"))
    rv_window = st.slider("Realised-vol window (days)", 10, 63, 21,
                          disabled=(iv_mode == "Vol index"))
    r = st.slider("Risk-free rate (%)", 0.0, 8.0, 4.0, 0.25) / 100
    q = st.slider("Dividend yield (%)", 0.0, 5.0, 100 * uni["q"], 0.1) / 100

    st.markdown("#### Exits")
    pt_enabled = st.toggle("Profit target", value=True)
    pt_pct = st.slider("Close at % of max profit", 10, 90, 50, 5,
                       disabled=not pt_enabled) / 100
    sl_enabled = st.toggle("Stop loss", value=True)
    sl_mult = st.slider("Stop at × credit received", 1.0, 5.0, 2.0, 0.25,
                        disabled=not sl_enabled)
    cooldown = st.slider("Days flat before re-entry", 0, 10, 0)

    st.markdown("#### Frictions")
    commission = st.number_input("Commission $/contract/leg/side", 0.0, 5.0, 0.65, 0.05)
    slippage_pct = st.slider("Slippage (% of premium per side)", 0.0, 5.0, 2.0, 0.5) / 100
    fractional = st.toggle("Allow fractional contracts", value=True,
                           help="Keeps sizing smooth for small accounts; turn off "
                                "for whole-contract realism.")

    st.divider()
    st.caption("Educational tool — synthetic Black–Scholes premiums, "
               "not investment advice.")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

fetch_start = start_date - timedelta(days=400)   # warm-up for RV / vol index
synthetic_fallback = False
try:
    with st.spinner(f"Loading {ticker} prices…"):
        prices_full = cached_prices(ticker, fetch_start)
except Exception:
    synthetic_fallback = True
    prices_full = cached_prices(dl.SYNTH_TICKER, fetch_start)

vol_ticker = dl.UNIVERSE["Demo — synthetic market (offline)"]["vol_index"] \
    if synthetic_fallback else uni["vol_index"]

vol_series = None
if iv_mode == "Vol index":
    with st.spinner("Loading vol index…"):
        vol_series = cached_vol_index(vol_ticker, tuple(prices_full.index))
    if vol_series is None:
        st.warning(f"No usable vol index for {ticker} — using realised vol × markup.")
        iv_mode = "Realised vol × markup"

if iv_mode == "Vol index":
    iv_full = vol_series
    iv_label = f"vol index ({vol_ticker})"
else:
    iv_full = realized_vol(prices_full, rv_window) * rv_mult
    iv_label = f"trailing {rv_window}d realised vol × {rv_mult:.2f}"

prices = prices_full.loc[str(start_date):str(end_date)]
iv = iv_full.loc[str(start_date):str(end_date)]
if len(prices) < 60:
    prices, iv = prices_full, iv_full   # degenerate window: use everything

cfg = Config(strategy=strategy, dte=dte, alloc_pct=alloc_pct,
             strike_method=strike_method, target_delta=target_delta,
             otm_pct=otm_pct, wing_pct=wing_pct,
             pt_enabled=pt_enabled, pt_pct=pt_pct,
             sl_enabled=sl_enabled, sl_mult=sl_mult, cooldown=cooldown,
             commission=commission, slippage_pct=slippage_pct,
             r=r, q=q, iv_mode=iv_mode, rv_mult=rv_mult, rv_window=rv_window,
             start_capital=float(start_capital), fractional=fractional)

res = cached_backtest(prices, iv, cfg)
daily, trades = res["daily"], res["trades"]


# ---------------------------------------------------------------------------
# Header + KPI row
# ---------------------------------------------------------------------------

display_name = "Synthetic market (offline demo)" if synthetic_fallback else uni_name
st.markdown(f"# <span style='color:#3FD0B6'>⏳</span> Theta Lab",
            unsafe_allow_html=True)
st.caption(f"**{strategy}** on **{display_name}** · {dte} DTE · premiums from "
           f"{iv_label} · {daily.index[0]:%b %Y} → {daily.index[-1]:%b %Y}")

if synthetic_fallback:
    st.info("Live data was unreachable, so you're looking at the built-in "
            "synthetic market — same engine, simulated prices.", icon="📡")
for w in res["warnings"]:
    st.warning(w, icon="⚠️")

stats = perf_stats(daily["equity"], rf=r)
bstats = perf_stats(daily["bench"], rf=r)
tstats = trade_stats(trades)

k = st.columns(6)
k[0].metric("CAGR", f"{stats['CAGR']:.1%}",
            f"{stats['CAGR'] - bstats['CAGR']:+.1%} vs B&H", border=True)
k[1].metric("Sharpe", f"{stats['Sharpe']:.2f}",
            f"{stats['Sharpe'] - bstats['Sharpe']:+.2f} vs B&H", border=True)
k[2].metric("Max drawdown", f"{stats['Max drawdown']:.1%}",
            f"B&H {bstats['Max drawdown']:.1%}", delta_color="off", border=True)
k[3].metric("Win rate", f"{tstats.get('Win rate', float('nan')):.0%}"
            if tstats else "—", f"{tstats.get('Trades', 0)} trades",
            delta_color="off", border=True)
k[4].metric("Profit factor",
            f"{tstats.get('Profit factor', float('nan')):.2f}" if tstats else "—",
            f"skew {tstats.get('Trade PnL skew', float('nan')):.2f}"
            if tstats else "", delta_color="off", border=True)
k[5].metric("Time in market", f"{time_in_market(daily):.0%}",
            f"avg hold {tstats.get('Avg days held', float('nan')):.0f}d"
            if tstats else "", delta_color="off", border=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_ov, tab_tr, tab_vol, tab_strat, tab_meth = st.tabs(
    ["Overview", "Trades", "Volatility", "Strategy & stress", "Methodology"])

with tab_ov:
    log_scale = st.toggle("Log scale", value=False)
    st.plotly_chart(plots.fig_equity(daily, log=log_scale),
                    width="stretch", config=PLOTLY_CFG)
    st.plotly_chart(plots.fig_drawdown(drawdown_series(daily["equity"]),
                                       drawdown_series(daily["bench"])),
                    width="stretch", config=PLOTLY_CFG)
    st.plotly_chart(plots.fig_monthly_heatmap(monthly_return_table(daily["equity"])),
                    width="stretch", config=PLOTLY_CFG)
    with st.expander("Full statistics table"):
        st.dataframe(pd.DataFrame({"Strategy": stats, "Buy & hold": bstats})
                     .style.format("{:.3f}"), width="stretch")

with tab_tr:
    if len(trades) == 0:
        st.info("No completed trades in this window — widen the dates or "
                "loosen the entry settings.")
    else:
        st.plotly_chart(plots.fig_trade_bars(trades), width="stretch",
                        config=PLOTLY_CFG)
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(plots.fig_pnl_hist(trades), width="stretch",
                        config=PLOTLY_CFG)
        c2.plotly_chart(plots.fig_exit_donut(trades), width="stretch",
                        config=PLOTLY_CFG)
        st.plotly_chart(plots.fig_iv_pnl_scatter(trades), width="stretch",
                        config=PLOTLY_CFG)
        with st.expander("Trade log"):
            show = trades.copy()
            for col in ("credit_$", "exit_value_$", "costs_$", "pnl_$"):
                show[col] = show[col].round(0)
            st.dataframe(show, width="stretch", hide_index=True)
            st.download_button("Download trade log (CSV)",
                               trades.to_csv(index=False).encode(),
                               "theta_lab_trades.csv", "text/csv")

with tab_vol:
    st.caption("The whole thesis of option selling: is implied vol (what you "
               "sell) systematically above realised vol (what is delivered)?")
    if len(trades):
        st.plotly_chart(plots.fig_vrp_timeseries(trades), width="stretch",
                        config=PLOTLY_CFG)
        st.plotly_chart(plots.fig_vrp_scatter(trades), width="stretch",
                        config=PLOTLY_CFG)
    st.plotly_chart(plots.fig_rolling_vol(daily, trades), width="stretch",
                    config=PLOTLY_CFG)

with tab_strat:
    S0 = float(daily["S"].iloc[-1])
    sigma0 = float(daily["iv"].iloc[-1])
    st.plotly_chart(plots.fig_payoff(payoff_curves(cfg, S0, sigma0)),
                    width="stretch", config=PLOTLY_CFG)
    st.caption(f"Structure priced at today's spot ({S0:,.0f}) and vol "
               f"({sigma0:.0%}).")
    with st.spinner("Stress-testing costs…"):
        st.plotly_chart(plots.fig_cost_sensitivity(
            cached_sensitivity(prices, iv, cfg)), width="stretch",
            config=PLOTLY_CFG)
    if st.toggle("Compare all five structures on these settings", value=True):
        with st.spinner("Running all strategies…"):
            comp = cached_comparison(prices, iv, cfg)
        st.plotly_chart(plots.fig_compare(comp, daily["bench"]),
                        width="stretch", config=PLOTLY_CFG)

with tab_meth:
    st.markdown(f"""
### How this backtest works

**Premium synthesis.** Free historical option chains don't exist, so every
premium is *modelled*: entry prices, daily marks and early-exit values all
come from Black–Scholes using **{iv_label}** as the volatility input.
Expiry settlement is exact intrinsic value. This is the single biggest
assumption in the tool — which is why the vol source and the RV markup are
user inputs, and why the stress tab exists.

**Accounting.** Equity = cash + mark-to-market of open positions, every
day. Credits land in cash at entry; positions are bought back (or settled)
at exit; covered calls carry 100 shares per contract, liquidated at each
close so the account is flat between trades and per-trade PnL is exact.

**Exits.** Hold to expiry, or close early at a profit target
({pt_pct:.0%} of max profit) / stop loss ({sl_mult:.1f}× credit) evaluated
on daily closes.

**Sizing.** Each trade's notional (spot × 100 × contracts) is
{alloc_pct:.0%} of current equity — conservative, roughly cash-secured
sizing; no margin or leverage is modelled.

**Costs.** ${commission:.2f} per contract per leg per side, plus
{slippage_pct:.1%} of premium per side as slippage.

### Known limitations (worth saying out loud)

- Flat vol surface: one vol number per day, **no skew or smile** — real
  short puts collect more (and risk more) than this model shows.
- European exercise, daily closes only: no intraday stop-outs, no early
  assignment.
- No margin modelling; sizing is notional-based.
- Strikes are continuous (real chains are gridded).
- Vol indices are ~30-day measures applied to all DTEs.

### Extensions I'd build next

Skew via a risk-reversal adjustment · delta-hedged variant to isolate the
vol premium · weekly expiries · a margin model with buying-power limits.
""")
