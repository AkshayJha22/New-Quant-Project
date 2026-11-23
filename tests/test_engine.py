"""Engine accounting on the deterministic synthetic market (formerly check 4 of smoke_test.py)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest import Config, STRATEGIES, run_backtest
from data_loader import synthetic_market

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "engine_v1.json").read_text())


@pytest.fixture(scope="module")
def market():
    prices, iv = synthetic_market(n_years=12, seed=7)
    return prices, iv


def reconcile(res, cfg):
    """Closed-trade P&L + interest + the open trade's unrealised P&L = change in equity."""
    d, t = res["daily"], res["trades"]
    total = t["pnl_$"].sum() if len(t) else 0.0
    total += res["summary"]["interest_$"] + res["summary"]["open_unrealized_$"]
    return total - (d["equity"].iloc[-1] - cfg.start_capital)


@pytest.mark.parametrize("strategy", ["Short Strangle", "Iron Condor", "Covered Call"])
@pytest.mark.parametrize("skew", [False, True])
def test_engine_runs_and_reconciles(market, strategy, skew):
    prices, iv = market
    cfg = Config(strategy=strategy, dte=30, alloc_pct=0.5, skew=skew)
    res = run_backtest(prices, iv, cfg)
    d, trades = res["daily"], res["trades"]
    assert np.isfinite(d["equity"]).all() and (d["equity"] > 0).all()
    assert len(trades) > 30
    assert abs(reconcile(res, cfg)) < 1e-6
    pt = trades[trades["exit_reason"] == "Profit target"]
    if len(pt):  # a profit-target exit must be profitable before costs
        assert (pt["credit_$"] > pt["exit_value_$"]).all()


def test_expiry_trade_pnl_rebuilt_from_its_own_fields(market):
    prices, iv = market
    cfg = Config(strategy="Short Straddle", pt_enabled=False, sl_enabled=False,
                 commission=0.0, slippage_pct=0.0)
    trades = run_backtest(prices, iv, cfg)["trades"]
    tr = trades[trades["exit_reason"] == "Expiry"].iloc[0]
    rebuilt = tr["credit_$"] - tr["exit_value_$"] - tr["costs_$"]
    assert tr["pnl_$"] == pytest.approx(rebuilt, abs=1e-6)


def test_interest_is_paid_on_cash_but_kept_out_of_trade_pnl(market):
    prices, iv = market
    cfg = Config(r=0.04)
    rates = pd.Series(0.04, index=prices.index)       # same rate as cfg.r: pricing unchanged
    flat = run_backtest(prices, iv, Config(r=0.04, interest_on_cash=False))
    paid = run_backtest(prices, iv, cfg, rates=rates)
    assert paid["summary"]["interest_$"] > 0
    assert paid["daily"]["equity"].iloc[-1] > flat["daily"]["equity"].iloc[-1]
    # interest grows equity, hence position size, but not P&L per contract
    a = (flat["trades"]["pnl_$"] / flat["trades"]["contracts"]).values[:30]
    b = (paid["trades"]["pnl_$"] / paid["trades"]["contracts"]).values[:30]
    assert np.allclose(a, b, rtol=2e-2, atol=1.0)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_old_engine_results_reproduced_exactly(strategy):
    """Smile, interest, stock costs and stock dividends off: identical to version 1."""
    prices, iv = synthetic_market(n_years=12, seed=7)
    prices, iv = prices.iloc[-1500:], iv.iloc[-1500:]
    cfg = Config(strategy=strategy, interest_on_cash=False, stock_cost_bps=0.0,
                 stock_dividends=False)
    res = run_backtest(prices, iv, cfg)
    assert res["daily"]["equity"].iloc[-1] == pytest.approx(FIXTURE[strategy]["final_equity"], rel=1e-12)
    assert len(res["trades"]) == FIXTURE[strategy]["trades"]


def test_term_structure_input(market):
    prices, iv = market
    ts = pd.DataFrame({9: iv * 0.95, 30: iv, 93: iv * 1.05})
    res = run_backtest(prices, ts, Config(dte=45))
    assert len(res["trades"]) > 20 and abs(reconcile(res, Config(dte=45))) < 1e-6
