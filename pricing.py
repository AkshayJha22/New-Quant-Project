"""
pricing.py: Black-Scholes pricing utilities (NumPy only, no SciPy).

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
    approximation. Absolute error < 7.5e-8, more than enough for pricing.
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
    Closed-form strike for a target |delta|: inverts BS delta analytically.

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


# ---------------------------------------------------------------------------
# Volatility smile (optional) and volatility-index conversion
# ---------------------------------------------------------------------------
#
# With the smile switched off, every strike is priced at one volatility, as in
# the original engine, and nothing below is used.
#
# With it on, a strike's volatility is the at-the-money volatility times a
# smile multiplier g(z), where z is the strike's distance from the forward in
# at-the-money standard deviations:
#
#     z    = ln(K / F) / (atm_vol * sqrt(T)),       F = S * exp((r - q) T)
#     g(z) = 1 + slope * z + curve * (sqrt(z^2 + tip^2) - tip)
#
# This is a smoothed "V": linear wings with a rounded tip at the money. A
# negative slope makes downside strikes dearer than upside ones (equity skew).
# Because z is measured in standard deviations, the smile keeps its shape in
# delta terms when volatility moves (a sticky-delta assumption), and the
# multiplier is bounded to [floor, cap] so far wings stay finite.
#
# A volatility index (VIX, VXN, India VIX, GVZ) is not at-the-money volatility:
# it is the square root of the fair variance-swap rate, computed from a whole
# strip of out-of-the-money options, so with skew it sits above ATM. When the
# smile is on, the ATM volatility is chosen so the smile's variance-swap level
# equals the index (atm_vol_from_index).

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class SkewParams:
    slope: float
    curve: float
    tip: float
    floor: float = 0.5
    cap: float = 3.0


# Fitted by least squares (scripts/calibrate_skew.py) to the 31-day SPY smile in
# the option-chain snapshot of 23 Nov 2026 (data/spy_smile_2026-11-23.csv),
# 117 out-of-the-money options with mid prices. At that day's ATM volatility of
# 22.1% the fit gives: 25-delta put +4.2 vol points over ATM (market +4.8),
# 25-delta call -3.0 (market -4.1), 10-delta put +10.9 (market +11.2), and a
# variance-swap / ATM ratio of 1.20 (market 1.16). Equity indices use it;
# gold (GLD) defaults to a flat smile.
EQUITY_INDEX_SKEW = SkewParams(slope=-0.233, curve=0.082, tip=1.84)


def smile_multiplier(z, p: SkewParams):
    z = np.asarray(z, dtype=float)
    g = 1.0 + p.slope * z + p.curve * (np.sqrt(z * z + p.tip * p.tip) - p.tip)
    return np.clip(g, p.floor, p.cap)


def smile_vol(S, K, T, atm_vol, r=0.0, q=0.0, params: SkewParams | None = None):
    """Implied volatility at strike K. params=None gives a flat smile (atm_vol)."""
    if params is None:
        return np.asarray(atm_vol, dtype=float) * np.ones_like(np.asarray(K, dtype=float))
    T_safe = np.maximum(np.asarray(T, dtype=float), 1e-12)
    atm = np.maximum(np.asarray(atm_vol, dtype=float), 1e-9)
    F = np.asarray(S, dtype=float) * np.exp((r - q) * T_safe)
    z = np.log(np.asarray(K, dtype=float) / F) / (atm * np.sqrt(T_safe))
    return atm * smile_multiplier(z, params)


def smile_price(S, K, T, atm_vol, r=0.0, q=0.0, kind="call", params: SkewParams | None = None):
    """Black-Scholes price with the strike's own smile volatility."""
    return bs_price(S, K, T, smile_vol(S, K, T, atm_vol, r, q, params), r, q, kind)


def smile_delta(S, K, T, atm_vol, r=0.0, q=0.0, kind="call", params: SkewParams | None = None):
    """Delta at the strike's own smile volatility (sticky strike within the day)."""
    return bs_delta(S, K, T, smile_vol(S, K, T, atm_vol, r, q, params), r, q, kind)


def strike_from_delta_smile(S, target_abs_delta, T, atm_vol, r=0.0, q=0.0, kind="call",
                            params: SkewParams | None = None) -> float:
    """
    Strike whose |delta|, computed at that strike's own smile volatility,
    equals the target. Flat smile: the closed form above. With a smile there is
    no closed form, so bisect on log-strike; |delta| is monotone in the strike
    over the bracket (checked in tests/test_skew.py).
    """
    if params is None:
        return strike_from_delta(S, target_abs_delta, T, atm_vol, r, q, kind)
    T = max(float(T), 1e-9)
    a = max(float(atm_vol), 1e-6) * np.sqrt(T)
    F = float(S) * np.exp((r - q) * T)
    lo, hi = (np.log(F) - 10 * a, np.log(F) + 2 * a) if kind == "put" \
        else (np.log(F) - 2 * a, np.log(F) + 10 * a)
    target = float(target_abs_delta)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        d = abs(float(smile_delta(S, np.exp(mid), T, atm_vol, r, q, kind, params)))
        # put: |delta| rises with the strike; call: |delta| falls with the strike
        if (d < target) == (kind == "put"):
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


def _variance_swap_ratio_exact(a: float, p: SkewParams) -> float:
    """
    sqrt(fair variance) / ATM vol for one maturity, with a = atm_vol * sqrt(T).
    Fair variance = (2/T) * integral of OTM option prices / K^2 dK (forward 1,
    no discounting), the replication behind the VIX formula.
    """
    T = 1.0                                   # only a = atm_vol*sqrt(T) matters
    atm = a
    u = np.linspace(-14 * a, 14 * a, 6001)    # log-strikes around the forward
    K = np.exp(u)
    sig = smile_vol(1.0, K, T, atm, 0.0, 0.0, p)
    srt = sig * np.sqrt(T)
    d1 = (-u + 0.5 * srt * srt) / srt
    d2 = d1 - srt
    put = K * norm_cdf(-d2) - norm_cdf(-d1)
    call = norm_cdf(d1) - K * norm_cdf(d2)
    otm = np.where(K < 1.0, put, call)
    var = 2.0 / T * np.trapezoid(otm / K, u)   # dK / K^2 = du / K
    return float(np.sqrt(var) / atm)


@lru_cache(maxsize=8)
def _ratio_table(p: SkewParams) -> tuple[np.ndarray, np.ndarray]:
    grid = np.exp(np.linspace(np.log(0.003), np.log(1.5), 60))
    return grid, np.array([_variance_swap_ratio_exact(a, p) for a in grid])


def variance_swap_ratio(atm_vol, T, params: SkewParams | None) -> float:
    """Variance-swap volatility divided by ATM volatility (1 for a flat smile)."""
    if params is None:
        return 1.0
    grid, ratio = _ratio_table(params)
    a = float(np.clip(float(atm_vol) * np.sqrt(max(float(T), 1e-9)), grid[0], grid[-1]))
    return float(np.interp(np.log(a), np.log(grid), ratio))


def atm_vol_from_index(index_vol, T, params: SkewParams | None) -> float:
    """
    ATM volatility whose smile has a variance-swap level equal to the index.
    Flat smile: the index itself. Fixed-point iteration; the ratio moves
    slowly with the ATM level, so a few steps converge.
    """
    index_vol = float(index_vol)
    if params is None or not np.isfinite(index_vol) or index_vol <= 0:
        return index_vol
    atm = index_vol / 1.2
    for _ in range(8):
        atm = index_vol / variance_swap_ratio(atm, T, params)
    return float(atm)


# ---------------------------------------------------------------------------
# Volatility term structure
# ---------------------------------------------------------------------------

def index_vol_at(tenor_days, vols, days) -> float:
    """
    Interpolate volatility-index levels quoted at fixed tenors (in calendar
    days, e.g. 9, 30 and 93 for VIX9D, VIX and VIX3M) at another maturity.
    Linear in total variance (vol^2 * T) between tenors, flat in volatility
    outside the quoted range. NaN tenors are ignored.
    """
    t = np.asarray(tenor_days, dtype=float)
    v = np.asarray(vols, dtype=float)
    ok = np.isfinite(v)
    t, v = t[ok], v[ok]
    if len(t) == 0:
        return float("nan")
    if len(t) == 1:
        return float(v[0])
    order = np.argsort(t)
    t, v = t[order], v[order]
    d = float(np.clip(days, t[0], t[-1]))
    w = np.interp(d, t, v * v * t)            # total variance, linear in maturity
    return float(np.sqrt(w / d))
