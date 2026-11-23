"""The optional volatility smile, the index-to-ATM conversion and the term structure."""
import numpy as np
import pytest

from pricing import (EQUITY_INDEX_SKEW, atm_vol_from_index, bs_delta, bs_price, index_vol_at,
                     smile_delta, smile_price, smile_vol, strike_from_delta,
                     strike_from_delta_smile, variance_swap_ratio)

P = EQUITY_INDEX_SKEW
T30 = 31 / 365


def test_flat_smile_reproduces_flat_prices_exactly():
    K = np.array([80.0, 95.0, 100.0, 105.0, 120.0])
    for kind in ("call", "put"):
        flat = bs_price(100, K, T30, 0.2, 0.04, 0.013, kind)
        assert np.array_equal(smile_price(100, K, T30, 0.2, 0.04, 0.013, kind, None), flat)
    assert strike_from_delta_smile(100, 0.3, T30, 0.2, 0.04, 0.013, "put", None) == \
        strike_from_delta(100, 0.3, T30, 0.2, 0.04, 0.013, "put")


def test_smile_makes_downside_puts_richer_and_upside_calls_cheaper():
    for K in (85.0, 90.0, 95.0):
        assert smile_price(100, K, T30, 0.2, 0, 0, "put", P) > bs_price(100, K, T30, 0.2, 0, 0, "put")
    for K in (105.0, 110.0):
        assert smile_price(100, K, T30, 0.2, 0, 0, "call", P) < bs_price(100, K, T30, 0.2, 0, 0, "call")
    # ATM volatility is unchanged by the smile
    assert abs(float(smile_vol(100, 100 * np.exp(0), T30, 0.2, 0, 0, P)) - 0.2) < 1e-12


def test_calibration_matches_the_spy_snapshot_targets():
    """At the snapshot's ATM vol (22.1%, 31 days) the fitted smile should land near
    the market's 25-delta and 10-delta skews (+4.8, -4.1 and +11.2 points)."""
    atm = 0.2208
    def premium(delta, kind):
        K = strike_from_delta_smile(100, delta, T30, atm, 0, 0, kind, P)
        return 100 * (float(smile_vol(100, K, T30, atm, 0, 0, P)) - atm)
    assert 3.5 <= premium(0.25, "put") <= 5.5
    assert -4.5 <= premium(0.25, "call") <= -2.0
    assert 9.0 <= premium(0.10, "put") <= 13.0
    assert 1.12 <= variance_swap_ratio(atm, T30, P) <= 1.28


@pytest.mark.parametrize("kind", ["put", "call"])
@pytest.mark.parametrize("atm", [0.10, 0.25, 0.60])
@pytest.mark.parametrize("days", [7, 30, 90])
def test_strike_from_delta_with_smile_round_trips(kind, atm, days):
    T = days / 365
    K = strike_from_delta_smile(100, 0.30, T, atm, 0.03, 0.01, kind, P)
    d = abs(float(smile_delta(100, K, T, atm, 0.03, 0.01, kind, P)))
    assert abs(d - 0.30) < 1e-6


@pytest.mark.parametrize("atm", [0.10, 0.25, 0.60])
def test_delta_is_monotone_in_strike_with_smile(atm):
    a = atm * np.sqrt(T30)
    K = 100 * np.exp(np.linspace(-6 * a, 6 * a, 801))
    put = np.abs(smile_delta(100, K, T30, atm, 0, 0, "put", P))
    call = smile_delta(100, K, T30, atm, 0, 0, "call", P)
    inside = (put[:-1] > 1e-9) & (put[1:] < 1 - 1e-9)      # ignore floating-point saturation
    assert np.all(np.diff(put)[inside] > 0) and np.all(np.diff(call)[inside] < 0)
    assert np.all(np.diff(put) >= -1e-12) and np.all(np.diff(call) <= 1e-12)


def test_index_to_atm_conversion():
    assert atm_vol_from_index(0.2, T30, None) == 0.2          # flat: the index itself
    atm = atm_vol_from_index(0.25, T30, P)
    assert atm < 0.25
    assert abs(atm * variance_swap_ratio(atm, T30, P) - 0.25) < 1e-6


def test_term_structure_interpolation():
    tenors, vols = [9, 30, 93], [0.15, 0.18, 0.20]
    assert index_vol_at(tenors, vols, 30) == pytest.approx(0.18)
    assert index_vol_at(tenors, vols, 3) == pytest.approx(0.15)      # flat below
    assert index_vol_at(tenors, vols, 200) == pytest.approx(0.20)    # flat above
    v = index_vol_at(tenors, vols, 14)
    w = 0.15 ** 2 * 9 + (0.18 ** 2 * 30 - 0.15 ** 2 * 9) * (14 - 9) / (30 - 9)
    assert v == pytest.approx(np.sqrt(w / 14))                      # linear in total variance
    assert index_vol_at(tenors, [np.nan, 0.18, np.nan], 60) == pytest.approx(0.18)
