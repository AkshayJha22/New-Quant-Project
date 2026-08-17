"""
pricing.py — Black–Scholes pricing utilities (NumPy only, no SciPy).

Everything here is vectorised: scalars, arrays and pandas objects all work.
The normal CDF and inverse CDF are implemented directly so the project has
no dependency beyond NumPy for its maths. Accuracy of both approximations
is verified in smoke_test.py against known reference values.

Conventions
-----------
S      : spot price of the underlying
K      : strike
T      : time to expiry in YEARS (calendar-day convention, i.e. days / 365)
sigma  : annualised volatility (e.g. 0.20 for 20%)
r      : continuously-compounded risk-free rate
q      : continuous dividend yield
kind   : "call" or "put"
"""

from __future__ import annotations

import numpy as np

SQRT_2PI = np.sqrt(2.0 * np.pi)
TRADING_DAYS = 252
CALENDAR_DAYS = 365.0


# ---------------------------------------------------------------------------
# Normal distribution helpers
# ---------------------------------------------------------------------------

def norm_pdf(x):
    """Standard normal density."""
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / SQRT_2PI


def norm_cdf(x):
    """
    Standard normal CDF via the Abramowitz & Stegun 26.2.17 rational
    approximation. Absolute error < 7.5e-8 — more than enough for pricing.
    """
    x = np.asarray(x, dtype=float)
    ax = np.abs(x)
    k = 1.0 / (1.0 + 0.2316419 * ax)
    poly = k * (0.319381530 + k * (-0.356563782 + k * (1.781477937
             + k * (-1.821255978 + k * 1.330274429))))
    approx = 1.0 - norm_pdf(ax) * poly
    return np.where(x >= 0.0, approx, 1.0 - approx)


def norm_ppf(p):
    """
    Inverse standard normal CDF using Acklam's rational approximation
    (relative error ~1.15e-9). Vectorised; input clipped away from {0, 1}.
    """
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]

    plow, phigh = 0.02425, 1.0 - 0.02425
    out = np.empty_like(p)

    # Lower tail
    lo = p < plow
    if np.any(lo):
        qq = np.sqrt(-2.0 * np.log(p[lo]))
        out[lo] = (((((c[0] * qq + c[1]) * qq + c[2]) * qq + c[3]) * qq + c[4]) * qq + c[5]) / \
                  ((((d[0] * qq + d[1]) * qq + d[2]) * qq + d[3]) * qq + 1.0)

    # Central region
    mid = (~lo) & (p <= phigh)
    if np.any(mid):
        qq = p[mid] - 0.5
        rr = qq * qq
        out[mid] = (((((a[0] * rr + a[1]) * rr + a[2]) * rr + a[3]) * rr + a[4]) * rr + a[5]) * qq / \
                   (((((b[0] * rr + b[1]) * rr + b[2]) * rr + b[3]) * rr + b[4]) * rr + 1.0)

    # Upper tail (mirror of lower)
    hi = p > phigh
    if np.any(hi):
        qq = np.sqrt(-2.0 * np.log(1.0 - p[hi]))
        out[hi] = -(((((c[0] * qq + c[1]) * qq + c[2]) * qq + c[3]) * qq + c[4]) * qq + c[5]) / \
                   ((((d[0] * qq + d[1]) * qq + d[2]) * qq + d[3]) * qq + 1.0)

    return out


# ---------------------------------------------------------------------------
# Black–Scholes
# ---------------------------------------------------------------------------

def bs_price(S, K, T, sigma, r=0.0, q=0.0, kind="call"):
    """
    Black–Scholes price of a European option with continuous dividend yield.
    At T <= 0 (or sigma <= 0) the price collapses to intrinsic value, which is
    exactly how expiring positions are settled in the backtest.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    intrinsic = np.maximum(S - K, 0.0) if kind == "call" else np.maximum(K - S, 0.0)

    T_safe = np.maximum(T, 1e-12)
    sig_safe = np.maximum(sigma, 1e-9)
    vol_sqrt_t = sig_safe * np.sqrt(T_safe)

    d1 = (np.log(S / K) + (r - q + 0.5 * sig_safe ** 2) * T_safe) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    disc_r = np.exp(-r * T_safe)
    disc_q = np.exp(-q * T_safe)

    if kind == "call":
        px = S * disc_q * norm_cdf(d1) - K * disc_r * norm_cdf(d2)
    else:
        px = K * disc_r * norm_cdf(-d2) - S * disc_q * norm_cdf(-d1)

    degenerate = (T <= 0.0) | (sigma <= 0.0)
    return np.where(degenerate, intrinsic, px)


def bs_delta(S, K, T, sigma, r=0.0, q=0.0, kind="call"):
    """Black–Scholes delta (spot delta, includes the e^{-qT} carry factor)."""
    T_safe = np.maximum(np.asarray(T, dtype=float), 1e-12)
    sig_safe = np.maximum(np.asarray(sigma, dtype=float), 1e-9)
    d1 = (np.log(np.asarray(S, float) / np.asarray(K, float))
          + (r - q + 0.5 * sig_safe ** 2) * T_safe) / (sig_safe * np.sqrt(T_safe))
    if kind == "call":
        return np.exp(-q * T_safe) * norm_cdf(d1)
    return np.exp(-q * T_safe) * (norm_cdf(d1) - 1.0)


def strike_from_delta(S, target_abs_delta, T, sigma, r=0.0, q=0.0, kind="call"):
    """
    Closed-form strike for a target |delta| — inverts BS delta analytically.

        call:  N(d1) = Δ · e^{qT}          →  d1 = Φ⁻¹(Δ e^{qT})
        put :  N(-d1) = |Δ| · e^{qT}       →  d1 = -Φ⁻¹(|Δ| e^{qT})

    then  K = S · exp[(r - q + σ²/2)T - d1·σ√T].

    A 0.30-delta put therefore lands below spot and a 0.30-delta call above
    it, which smoke_test.py verifies by round-tripping through bs_delta.
    """
    T = float(T)
    adj = float(target_abs_delta) * np.exp(q * T)
    adj = min(max(adj, 1e-6), 1.0 - 1e-6)
    d1 = float(norm_ppf(adj)) if kind == "call" else -float(norm_ppf(adj))
    return float(S) * np.exp((r - q + 0.5 * sigma ** 2) * T - d1 * sigma * np.sqrt(T))


# ---------------------------------------------------------------------------
# Volatility estimators
# ---------------------------------------------------------------------------

def realized_vol(prices, window: int = 21):
    """Rolling annualised realised volatility from close-to-close log returns."""
    logret = np.log(prices / prices.shift(1))
    return logret.rolling(window).std() * np.sqrt(TRADING_DAYS)


def realized_vol_over(prices) -> float:
    """Annualised realised vol over one specific window of prices (a trade's life)."""
    logret = np.log(prices / prices.shift(1)).dropna()
    if len(logret) < 3:
        return float("nan")
    return float(logret.std() * np.sqrt(TRADING_DAYS))
