"""The committed data snapshot and the Cboe validation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import data_loader as dl
import validation

ROOT = Path(__file__).resolve().parents[1]
HAS_SNAPSHOT = dl.snapshot_available("SPY")


@pytest.mark.skipif(not HAS_SNAPSHOT, reason="no data snapshot")
def test_snapshot_loads_with_term_structure_and_rates():
    m = dl.load_market("SPY: S&P 500 ETF", "snapshot")
    assert m.source == "snapshot" and len(m.close) > 4000
    assert (m.adj_close <= m.close * 1.0001).all()          # dividends adjust history down
    assert isinstance(m.vol, pd.DataFrame) and list(m.vol.columns) == [9, 30, 93]
    recent = m.vol.loc["2012":]
    assert recent.notna().all().all() and recent.median().between(0.08, 0.4).all()
    assert m.rate.dropna().between(-0.01, 0.08).all()


@pytest.mark.skipif(not HAS_SNAPSHOT, reason="no data snapshot")
@pytest.mark.parametrize("name", ["QQQ: Nasdaq-100 ETF", "NIFTY 50 (^NSEI)", "GLD: Gold ETF"])
def test_other_snapshots_load(name):
    m = dl.load_market(name, "snapshot")
    assert isinstance(m.vol, pd.Series) and m.vol.dropna().median() > 0.05


def test_compare_statistics_on_identical_and_shifted_curves():
    idx = pd.bdate_range("2015-01-01", periods=1500)
    rng = np.random.default_rng(0)
    s = pd.Series(np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx)))), index=idx)
    same = validation.compare(s, s)
    assert same["cagr_gap"] == pytest.approx(0.0) and same["monthly_corr"] == pytest.approx(1.0)
    assert same["tracking_error"] == pytest.approx(0.0, abs=1e-12)
    drift = s * np.exp(0.02 * np.arange(len(idx)) / 252)
    assert validation.compare(drift, s)["cagr_gap"] > 0.015


@pytest.mark.skipif(not HAS_SNAPSHOT, reason="no data snapshot")
def test_validation_matches_committed_results():
    summary, curves = validation.run()
    committed = json.loads((ROOT / "results" / "validation.json").read_text())
    for label, stats in committed["results"].items():
        assert summary["results"][label]["model_cagr"] == pytest.approx(stats["model_cagr"], abs=1e-9)
    # the smile brings the put-writing replication closer to the real index
    flat, smile = summary["results"]["PUT model, flat"], summary["results"]["PUT model, smile"]
    assert abs(smile["cagr_gap"]) < abs(flat["cagr_gap"])
