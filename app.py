"""
app.py: Systematic Option Selling Backtester (Streamlit entry point).

This file only does UI and orchestration; all maths lives in pricing.py,
backtest.py and metrics.py, and all figures in plots.py, so every layer can
be tested without a browser (see tests/).

Run with:  streamlit run app.py
"""

from __future__ import annotations

import dataclasses
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

import data_loader as dl
import plots
from backtest import Config, STRATEGIES, run_backtest, payoff_curves, headline_vol
from metrics import (perf_stats, trade_stats, drawdown_series,
                     monthly_return_table, time_in_market)
from pricing import realized_vol, atm_vol_from_index, CALENDAR_DAYS

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Systematic Option Selling Backtester", page_icon="📉",
                   layout="wide")

st.markdown("""
<style>
h1, h2, h3 { letter-spacing: -0.01em; }
div[data-testid="stSidebarHeader"] { padding-bottom: 0; }
</style>
""", unsafe_allow_html=True)

PLOTLY_CFG = {"displayModeBar": False}


# ---------------------------------------------------------------------------
# Cached data / compute wrappers (all heavy lifting is memoised)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def cached_market(uni_name: str, source: str, custom: str | None, start: date,
                  use_ts: bool) -> dl.Market:
    return dl.load_market(uni_name, source, custom, start, use_ts)


@st.cache_data(show_spinner=False)
def cached_backtest(prices, iv, cfg: Config, bench, rates) -> dict:
    return run_backtest(prices, iv, cfg, bench=bench, rates=rates)


@st.cache_data(show_spinner=False)
def cached_sensitivity(prices, iv, cfg: Config, bench, rates) -> pd.DataFrame:
    base = cfg.slippage_pct if cfg.slippage_pct > 0 else 0.02
    grid = sorted({0.0, 0.5 * base, base, 1.5 * base, 2 * base, 3 * base})
    rows = []
    for s in grid:
        res = run_backtest(prices, iv, dataclasses.replace(cfg, slippage_pct=s),
                           bench=bench, rates=rates)
        d = res["daily"]
        ps = perf_stats(d["equity"], rf=d["rate"])
        rows.append({"slippage_pct": s, "sharpe": ps["Sharpe"], "cagr": ps["CAGR"]})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def cached_comparison(prices, iv, cfg: Config, bench, rates) -> dict:
    out = {}
    for name in STRATEGIES:
        res = run_backtest(prices, iv, dataclasses.replace(cfg, strategy=name),
                           bench=bench, rates=rates)
        out[name] = res["daily"]["equity"]
    return out


@st.cache_data(show_spinner=False)
def cached_validation() -> tuple[dict, dict]:
    import validation
    return validation.run()


# ---------------------------------------------------------------------------
# Sidebar: every assumption is a visible, adjustable input
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📉 Option selling backtester")
    st.caption("Systematic short-premium strategies, priced from volatility indices")

    st.markdown("#### Market")
    uni_name = st.selectbox("Underlying", list(dl.UNIVERSE.keys()), index=0)
    uni = dl.UNIVERSE[uni_name]
    custom = None
    if uni["ticker"] is None:
        custom = st.text_input("Yahoo Finance ticker", value="AAPL").strip() or "AAPL"

    if uni["ticker"] == dl.SYNTH_TICKER:
        source = "synthetic"
    elif uni["snapshot"] and dl.snapshot_available(uni["snapshot"]):
        source = st.radio("Data", ["snapshot", "live"], horizontal=True,
                          format_func=lambda s: {"snapshot": "Bundled snapshot",
                                                 "live": "Live (Yahoo)"}[s],
                          help="The snapshot under data/ works offline and matches the "
                               "README. Live data needs internet.")
    else:
        source = "live"

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
        otm_pct = st.slider("Short strikes % OTM", 0.0, 0.15, 0.05, 0.01)
        target_delta = 0.30
    wing_pct = (st.slider("Condor wing width (% of spot)", 0.01, 0.15, 0.05, 0.01)
                if strategy == "Iron Condor" else 0.05)
    alloc_pct = st.slider("Notional per trade (% of equity)", 10, 100, 50, 5) / 100

    st.markdown("#### Volatility and rates")
    st.caption("Premiums are synthesised with Black-Scholes from this volatility input.")
    iv_options = (["Vol index"] if uni["vol"] else []) + ["Realised vol x markup"]
    iv_mode = st.radio("Implied-vol source", iv_options, horizontal=True)
    use_ts = False
    if iv_mode == "Vol index" and len(uni["vol"]) > 1:
        use_ts = st.toggle("Term structure (VIX9D / VIX / VIX3M)", value=True,
                           help="Each option is priced at the volatility for its remaining "
                                "life, interpolated in total variance between the 9-day, "
                                "30-day and 93-day indices.")
    skew = st.toggle("Volatility smile (skew)", value=uni["skew"],
                     help="Price each strike at its own volatility. With a volatility "
                          "index as input, ATM volatility is backed out so the smile's "
                          "variance-swap level equals the index. Off: one volatility "
                          "prices every strike.")
    rv_mult = st.slider("RV markup (proxy for the vol premium)", 0.80, 1.60, 1.10, 0.05,
                        disabled=(iv_mode == "Vol index"))
    rv_window = st.slider("Realised-vol window (days)", 10, 63, 21,
                          disabled=(iv_mode == "Vol index"))
    has_rate_series = bool(uni["rate"]) and source != "synthetic"
    r = st.slider("Risk-free rate (%)", 0.0, 10.0, 100 * uni["r"], 0.25,
                  help="Used for discounting and interest on cash when no T-bill "
                       "series is available.") / 100
    use_rate_series = has_rate_series and st.toggle(
        "Cash earns the 13-week T-bill", value=True,
        help="Interest on cash and discounting follow the daily ^IRX yield.")
    q = st.slider("Dividend yield (%)", 0.0, 5.0, 100 * uni["q"], 0.1) / 100

    st.markdown("#### Exits")
    pt_enabled = st.toggle("Profit target", value=True)
    pt_pct = st.slider("Close at % of max profit", 10, 90, 50, 5,
                       disabled=not pt_enabled) / 100
    sl_enabled = st.toggle("Stop loss", value=True)
    sl_mult = st.slider("Stop at x credit received", 1.0, 5.0, 2.0, 0.25,
                        disabled=not sl_enabled)
    cooldown = st.slider("Days flat before re-entry", 0, 10, 0)

    st.markdown("#### Frictions")
    commission = st.number_input("Commission $/contract/leg/side", 0.0, 5.0, 0.65, 0.05)
    slippage_pct = st.slider("Slippage (% of premium per side)", 0.0, 5.0, 2.0, 0.5) / 100
    stock_cost_bps = st.number_input("Stock trading cost (bps per side, covered call)",
                                     0.0, 50.0, 1.0, 0.5)
    fractional = st.toggle("Allow fractional contracts", value=True,
                           help="Keeps sizing smooth for small accounts; turn off "
                                "for whole-contract realism.")

    st.divider()
    st.caption("Educational tool: synthetic Black-Scholes premiums, "
               "not investment advice.")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

fetch_start = start_date - timedelta(days=400)   # warm-up for RV / vol index
fallback_note = None
try:
    with st.spinner("Loading market data..."):
        mkt = cached_market(uni_name, source, custom, fetch_start, use_ts)
except Exception:
    if uni["snapshot"] and dl.snapshot_available(uni["snapshot"]):
        mkt = cached_market(uni_name, "snapshot", None, fetch_start, use_ts)
        fallback_note = "Live data was unreachable, so the bundled snapshot is shown."
    else:
        mkt = cached_market("Demo: synthetic market (offline)", "synthetic", None,
                            fetch_start, False)
        fallback_note = ("Live data was unreachable, so you're looking at the built-in "
                         "synthetic market: same engine, simulated prices.")

if iv_mode == "Vol index" and mkt.vol is None:
    st.warning("No usable volatility index for this ticker: using realised vol x markup.")
    iv_mode = "Realised vol x markup"

if iv_mode == "Vol index":
    iv_full = mkt.vol
    if isinstance(iv_full, pd.DataFrame):
        iv_label = "VIX9D / VIX / VIX3M term structure"
    else:
        iv_label = f"volatility index ({uni['vol'][30][0] if 30 in uni['vol'] else 'synthetic'})"
else:
    iv_full = realized_vol(mkt.close, rv_window) * rv_mult
    iv_label = f"trailing {rv_window}d realised vol x {rv_mult:.2f}"

window = slice(str(start_date), str(end_date))
prices = mkt.close.loc[window]
bench = mkt.adj_close.loc[window]
iv = iv_full.loc[window]
rates = mkt.rate.loc[window] if (use_rate_series and mkt.rate is not None) else None
if len(prices) < 60:
    prices, bench, iv = mkt.close, mkt.adj_close, iv_full   # degenerate window: use everything
    rates = mkt.rate if (use_rate_series and mkt.rate is not None) else None

cfg = Config(strategy=strategy, dte=dte, alloc_pct=alloc_pct,
             strike_method=strike_method, target_delta=target_delta,
             otm_pct=otm_pct, wing_pct=wing_pct,
             pt_enabled=pt_enabled, pt_pct=pt_pct,
             sl_enabled=sl_enabled, sl_mult=sl_mult, cooldown=cooldown,
             commission=commission, slippage_pct=slippage_pct,
             stock_cost_bps=stock_cost_bps, r=r, q=q, iv_mode=iv_mode,
             rv_mult=rv_mult, rv_window=rv_window,
             start_capital=float(start_capital), fractional=fractional, skew=skew)

res = cached_backtest(prices, iv, cfg, bench, rates)
daily, trades = res["daily"], res["trades"]


# ---------------------------------------------------------------------------
# Header + KPI row
# ---------------------------------------------------------------------------

display_name = {"synthetic": "Synthetic market (offline demo)"}.get(mkt.source, uni_name)
if custom:
    display_name = custom
st.markdown("# Systematic Option Selling Backtester")
st.caption(f"**{strategy}** on **{display_name}** ({mkt.source} data) · {dte} DTE · "
           f"premiums from {iv_label}{' with a volatility smile' if skew else ''} · "
           f"{daily.index[0]:%b %Y} to {daily.index[-1]:%b %Y}")

if fallback_note:
    st.info(fallback_note, icon="📡")
for w in res["warnings"]:
    st.warning(w, icon="⚠️")

stats = perf_stats(daily["equity"], rf=daily["rate"])
bstats = perf_stats(daily["bench"], rf=daily["rate"])
tstats = trade_stats(trades)

k = st.columns(6)
k[0].metric("CAGR", f"{stats['CAGR']:.1%}",
            f"{stats['CAGR'] - bstats['CAGR']:+.1%} vs B&H", border=True)
k[1].metric("Sharpe", f"{stats['Sharpe']:.2f}",
            f"{stats['Sharpe'] - bstats['Sharpe']:+.2f} vs B&H", border=True)
k[2].metric("Max drawdown", f"{stats['Max drawdown']:.1%}",
            f"B&H {bstats['Max drawdown']:.1%}", delta_color="off", border=True)
k[3].metric("Win rate", f"{tstats.get('Win rate', float('nan')):.0%}"
            if tstats else "n/a", f"{tstats.get('Trades', 0)} trades",
            delta_color="off", border=True)
k[4].metric("Profit factor",
            f"{tstats.get('Profit factor', float('nan')):.2f}" if tstats else "n/a",
            f"skew {tstats.get('Trade PnL skew', float('nan')):.2f}"
            if tstats else "", delta_color="off", border=True)
k[5].metric("Interest on cash", f"${res['summary']['interest_$']:,.0f}",
            f"in market {time_in_market(daily):.0%}", delta_color="off", border=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_ov, tab_tr, tab_vol, tab_strat, tab_val, tab_meth = st.tabs(
    ["Overview", "Trades", "Volatility", "Strategy & stress", "Validation vs Cboe",
     "Methodology"])

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
        st.info("No completed trades in this window: widen the dates or "
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
            for col in ("credit_$", "exit_value_$", "costs_$", "dividends_$", "pnl_$"):
                show[col] = show[col].round(0)
            st.dataframe(show, width="stretch", hide_index=True)
            st.download_button("Download trade log (CSV)",
                               trades.to_csv(index=False).encode(),
                               "option_selling_trades.csv", "text/csv")

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
    index0 = float(daily["iv"].iloc[-1])
    sigma0 = (atm_vol_from_index(index0, dte / CALENDAR_DAYS, cfg.skew_params)
              if (skew and iv_mode == "Vol index") else index0)
    st.plotly_chart(plots.fig_payoff(payoff_curves(cfg, S0, sigma0)),
                    width="stretch", config=PLOTLY_CFG)
    st.caption(f"Structure priced at the latest spot ({S0:,.0f}) and ATM vol "
               f"({sigma0:.0%}).")
    with st.spinner("Stress-testing costs..."):
        st.plotly_chart(plots.fig_cost_sensitivity(
            cached_sensitivity(prices, iv, cfg, bench, rates)), width="stretch",
            config=PLOTLY_CFG)
    if st.toggle("Compare all five structures on these settings", value=True):
        with st.spinner("Running all strategies..."):
            comp = cached_comparison(prices, iv, cfg, bench, rates)
        st.plotly_chart(plots.fig_compare(comp, daily["bench"]),
                        width="stretch", config=PLOTLY_CFG)

with tab_val:
    st.markdown("### Validation against real option prices")
    st.caption("Cboe's PutWrite (PUT) and BuyWrite (BXM) indices are built from real "
               "S&P 500 option prices. The same strategies run through this engine, with "
               "premiums synthesised from the VIX, show how far modelled premiums are "
               "from real ones. Settings: monthly at-the-money options on the S&P 500, "
               "held to expiry, no costs, cash at the 13-week T-bill.")
    json_path, png_path = ROOT / "results" / "validation.json", ROOT / "results" / "validation.png"
    summary = None
    if st.button("Recompute from the data snapshot"):
        with st.spinner("Running the four replications..."):
            summary, curves = cached_validation()
        st.plotly_chart(plots.fig_validation(curves), width="stretch", config=PLOTLY_CFG)
    elif json_path.exists():
        summary = json.loads(json_path.read_text())
        if png_path.exists():
            st.image(str(png_path), width="stretch")
    if summary:
        table = pd.DataFrame(summary["results"]).T
        table = table[["model_cagr", "index_cagr", "cagr_gap", "monthly_corr",
                       "tracking_error", "model_max_dd", "index_max_dd"]]
        table.columns = ["Model CAGR", "Index CAGR", "Gap", "Monthly corr",
                         "Tracking error", "Model max DD", "Index max DD"]
        st.dataframe(table.style.format({c: "{:.2%}" for c in table.columns
                                         if c != "Monthly corr"} | {"Monthly corr": "{:.3f}"}),
                     width="stretch")
        st.caption(f"Period {summary['period']}. Dividend yield used for the covered call: "
                   f"{summary['dividend_yield_used']:.2%}.")
    else:
        st.info("No validation results yet: run `python validation.py`.")

with tab_meth:
    st.markdown(f"""
### How this backtest works

**Premium synthesis.** Historical option chains are not used: entry prices,
daily marks and early-exit values all come from Black-Scholes using
**{iv_label}** as the volatility input. Expiry settlement is exact intrinsic
value. This is the single biggest assumption in the tool, which is why the
volatility source, the smile and the RV markup are user inputs, and why the
validation tab compares the engine with real option-based indices.

**Volatility smile.** With the smile on, each strike is priced at its own
volatility: ATM volatility times a smile multiplier in standardised
moneyness, fitted to a real SPY option chain. A volatility index is a
variance-swap level, which sits above ATM volatility when there is skew, so
the ATM volatility is backed out of the index. On the Cboe PUT index this
cut the replication's overstatement of put-selling returns from about 6 points
a year to about 1.

**Term structure.** For SPY, each option uses the volatility for its
remaining life, interpolated in total variance between VIX9D, VIX and VIX3M.

**Accounting.** Equity = cash + mark-to-market of open positions, every
day. Cash earns the risk-free rate (the 13-week T-bill where available).
Credits land in cash at entry; positions are bought back (or settled) at
exit; covered calls carry 100 shares per contract, earn the dividend yield
while held and are liquidated at each close, so the account is flat between
trades. Trade P&L excludes interest; the equity curve includes it.

**Prices.** Options are priced off raw closes with the dividend yield q;
the buy-and-hold benchmark uses dividend-adjusted closes (total return).

**Exits.** Hold to expiry, or close early at a profit target
({pt_pct:.0%} of max profit) or stop loss ({sl_mult:.1f}x credit) evaluated
on daily closes.

**Sizing.** Each trade's notional (spot x 100 x contracts) is
{alloc_pct:.0%} of current equity: conservative, roughly cash-secured
sizing; no margin or leverage is modelled.

**Costs.** ${commission:.2f} per contract per leg per side, plus
{slippage_pct:.1%} of premium per side as slippage, plus
{stock_cost_bps:.1f} bps per side on covered-call stock trades.

### Known limitations

- The smile is one fitted shape (sticky delta): skew level changes over time
  are not modelled.
- European exercise, daily closes only: no intraday stop-outs, no early
  assignment.
- No margin modelling; sizing is notional-based.
- Strikes are continuous (real chains are gridded).
- One history: ten years contain only a handful of crashes.
""")
