"""
data_loader.py — market data access.

Two sources, one interface:

1. Live data via yfinance (daily closes, plus the matching CBOE/NSE vol
   index where one exists, used as the implied-vol input).
2. A fully offline synthetic market: geometric Brownian motion driven by a
   mean-reverting stochastic volatility process with a leverage effect
   (vol spikes when prices fall), plus a synthetic "vol index" quoted at a
   premium to true vol. This keeps the app demo-able with no internet and
   gives the smoke test a deterministic fixture.

No Streamlit imports here — the module stays UI-free so it can be unit
tested directly. Caching wrappers live in app.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SYNTH_TICKER = "__SYNTH__"

# Display name -> config. vol_index=None means no listed vol index; the app
# then falls back to trailing realised vol x a user-set multiplier.
UNIVERSE: dict[str, dict] = {
    "SPY — S&P 500 ETF":        {"ticker": "SPY",   "vol_index": "^VIX",      "q": 0.013},
    "QQQ — Nasdaq-100 ETF":     {"ticker": "QQQ",   "vol_index": "^VXN",      "q": 0.006},
    "NIFTY 50 (^NSEI)":         {"ticker": "^NSEI", "vol_index": "^INDIAVIX", "q": 0.012},
    "GLD — Gold ETF":           {"ticker": "GLD",   "vol_index": "^GVZ",      "q": 0.000},
    "Demo — synthetic market (offline)":
                                {"ticker": SYNTH_TICKER, "vol_index": SYNTH_TICKER, "q": 0.000},
    "Custom ticker…":           {"ticker": None,    "vol_index": None,        "q": 0.000},
}


# ---------------------------------------------------------------------------
# Synthetic market
# ---------------------------------------------------------------------------

def synthetic_market(n_years: int = 15, seed: int = 7,
                     s0: float = 100.0) -> tuple[pd.Series, pd.Series]:
    """
    Simulate (price, vol-index) with realistic short-vol dynamics:

      d ln(sigma) = kappa (ln(sigma_bar) - ln(sigma)) dt + xi dW2
      d ln(S)     = (mu - sigma^2 / 2) dt + sigma dW1,   corr(dW1, dW2) = rho < 0

    rho < 0 is the leverage effect: sell-offs come with vol spikes, which is
    precisely what makes systematic option selling dangerous. The synthetic
    vol index quotes at a ~10% premium to true instantaneous vol so the
    demo exhibits a volatility risk premium by construction.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(),
                         periods=int(n_years * 252))
    n = len(idx)                      # authoritative: end date may be a weekend
    dt = 1.0 / 252

    mu, sigma_bar, kappa, xi, rho = 0.07, 0.16, 3.0, 1.1, -0.7

    z1 = rng.standard_normal(n)
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)

    log_sig = np.empty(n)
    log_sig[0] = np.log(sigma_bar)
    for t in range(1, n):
        log_sig[t] = (log_sig[t - 1]
                      + kappa * (np.log(sigma_bar) - log_sig[t - 1]) * dt
                      + xi * np.sqrt(dt) * z2[t])
    sig = np.clip(np.exp(log_sig), 0.06, 1.2)

    rets = (mu - 0.5 * sig ** 2) * dt + sig * np.sqrt(dt) * z1
    prices = s0 * np.exp(np.cumsum(rets))

    price = pd.Series(prices, index=idx, name="SYNTH")

    vrp_markup = 1.10
    noise = rng.normal(0, 0.01, n)
    vol_index = pd.Series(np.clip(sig * vrp_markup + noise, 0.05, 1.5),
                          index=idx, name="SYNTH_IV")
    return price, vol_index


# ---------------------------------------------------------------------------
# Live data
# ---------------------------------------------------------------------------

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance sometimes returns MultiIndex columns even for one ticker."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_prices(ticker: str, start=None, end=None) -> pd.Series:
    """
    Daily adjusted closes as a float Series indexed by date.
    Raises RuntimeError if nothing comes back — the app catches this and
    offers the synthetic market instead of dying.
    """
    if ticker == SYNTH_TICKER:
        price, _ = synthetic_market()
        return price

    import yfinance as yf
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError(f"No price data returned for {ticker!r}.")
    df = _flatten(df)
    close = df["Close"].dropna().astype(float)
    close.name = ticker
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    return close


def fetch_vol_index(vol_ticker: str | None,
                    price_index: pd.DatetimeIndex) -> pd.Series | None:
    """
    Vol index as a decimal (VIX 20 -> 0.20), forward-filled onto the price
    calendar. Returns None when unavailable so the caller can fall back to
    realised vol.
    """
    if vol_ticker is None:
        return None
    if vol_ticker == SYNTH_TICKER:
        _, iv = synthetic_market()
        return iv.reindex(price_index).ffill()

    import yfinance as yf
    try:
        df = yf.download(vol_ticker, start=price_index[0], end=None,
                         auto_adjust=True, progress=False)
        if df is None or len(df) == 0:
            return None
        df = _flatten(df)
        iv = (df["Close"].dropna().astype(float) / 100.0)
        iv.index = pd.DatetimeIndex(iv.index).tz_localize(None)
        iv = iv.reindex(price_index).ffill()
        return iv if iv.notna().sum() > 50 else None
    except Exception:
        return None
