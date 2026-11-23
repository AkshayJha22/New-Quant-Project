"""Pricing maths against reference values (formerly checks 1-3 of smoke_test.py)."""
import numpy as np
import pytest

from pricing import bs_delta, bs_price, norm_cdf, norm_ppf, strike_from_delta


def test_normal_cdf_reference_values():
    assert abs(norm_cdf(0.0) - 0.5) < 1e-7
    assert abs(norm_cdf(1.96) - 0.9750021) < 1e-5
    assert abs(norm_cdf(-1.0) - 0.1586553) < 1e-5


@pytest.mark.parametrize("p, ref", [(0.975, 1.959964), (0.95, 1.6448536), (0.01, -2.3263479)])
def test_normal_ppf_reference_values(p, ref):
    assert abs(float(norm_ppf(p)) - ref) < 1e-6


def test_ppf_cdf_round_trip():
    x = np.linspace(-4, 4, 161)
    # the CDF approximation error (< 7.5e-8) is amplified in the tails
    assert np.max(np.abs(norm_ppf(norm_cdf(x)) - x)) < 2e-4


def test_black_scholes_reference_and_parity():
    c = float(bs_price(100, 100, 1.0, 0.20, r=0.05, kind="call"))
    assert abs(c - 10.4506) < 1e-3
    S, K, T, sig, r, q = 100, 95, 0.5, 0.25, 0.03, 0.01
    call = float(bs_price(S, K, T, sig, r, q, "call"))
    put = float(bs_price(S, K, T, sig, r, q, "put"))
    assert abs(call - put - (S * np.exp(-q * T) - K * np.exp(-r * T))) < 1e-8


def test_expiry_settles_at_intrinsic():
    assert float(bs_price(110, 100, 0.0, 0.2, kind="call")) == 10.0
    assert float(bs_price(110, 100, 0.0, 0.2, kind="put")) == 0.0


@pytest.mark.parametrize("kind", ["call", "put"])
def test_strike_from_delta_round_trip(kind):
    T, sig = 30 / 365, 0.22
    K = strike_from_delta(100, 0.30, T, sig, kind=kind)
    assert abs(abs(float(bs_delta(100, K, T, sig, kind=kind))) - 0.30) < 1e-6
    assert (K > 100) if kind == "call" else (K < 100)
