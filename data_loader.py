"""
data_loader.py: market data access.

Three sources, one interface (load_market):

1. "Snapshot": CSVs committed under data/, downloaded on GitHub's runners by
   .github/workflows/data.yml (scripts/fetch_data.py). Works offline and makes
   every number in the README reproducible.
2. "Live": Yahoo Finance via yfinance at runtime (needs internet).
3. "Synthetic": an offline simulated market (geometric Brownian motion with
   mean-reverting stochastic volatility and a leverage effect, plus a
   synthetic vol index quoted at a premium to true vol). Used for the demo
   and as a deterministic fixture for the tests.

For every source the loader returns raw closes (the price options are written
on), adjusted closes (dividends folded in, for the buy-and-hold benchmark),
the volatility index (one 30-day series, or for SPY the VIX9D / VIX / VIX3M
term structure keyed by tenor in days) and, where available, a risk-free
rate series (13-week T-bill) for interest on cash.

No Streamlit imports here, so the module can be unit tested directly. Caching
wrappers live in app.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
SYNTH_TICKER = "__SYNTH__"

# Display name -> config. "snapshot" names files under data/prices; "vol" maps
# tenor in days -> volatility index (snapshot file name, Yahoo ticker);
# "rate" is the risk-free series, or None to use the constant from the sidebar.
UNIVERSE: dict[str, dict] = {
    "SPY: S&P 500 ETF": {
        "ticker": "SPY", "snapshot": "SPY", "q": 0.013, "r": 0.04, "skew": True,
        "vol": {9: ("VIX9D", "^VIX9D"), 30: ("VIX", "^VIX"), 93: ("VIX3M", "^VIX3M")},
        "rate": ("IRX", "^IRX")},
    "QQQ: Nasdaq-100 ETF": {
        "ticker": "QQQ", "snapshot": "QQQ", "q": 0.006, "r": 0.04, "skew": True,
        "vol": {30: ("VXN", "^VXN")}, "rate": ("IRX", "^IRX")},
    "NIFTY 50 (^NSEI)": {
        "ticker": "^NSEI", "snapshot": "NSEI", "q": 0.012, "r": 0.065, "skew": True,
        "vol": {30: ("INDIAVIX", "^INDIAVIX")}, "rate": None},
    "GLD: Gold ETF": {
        "ticker": "GLD", "snapshot": "GLD", "q": 0.000, "r": 0.04, "skew": False,
        "vol": {30: ("GVZ", "^GVZ")}, "rate": ("IRX", "^IRX")},
    "Demo: synthetic market (offline)": {
        "ticker": SYNTH_TICKER, "snapshot": None, "q": 0.000, "r": 0.04, "skew": True,
        "vol": {30: ("SYNTH", SYNTH_TICKER)}, "rate": None},
    "Custom ticker...": {
        "ticker": None, "snapshot": None, "q": 0.000, "r": 0.04, "skew": False,
        "vol": {}, "rate": None},
}


@dataclass
class Market:
    close: pd.Series                     # raw closes: what options are written on
    adj_close: pd.Series                 # total-return closes: the benchmark
    vol: pd.Series | pd.DataFrame | None  # decimals; DataFrame columns = tenor days
    rate: pd.Series | None               # annualised decimals, or None
    source: str                          # "snapshot" | "live" | "synthetic"
    notes: list = field(default_factory=list)


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
    vol index quotes at a ~10% premium to true instantaneous vol.
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
# Snapshot (data/ committed in the repo)
# ---------------------------------------------------------------------------

def snapshot_available(name: str | None) -> bool:
    return bool(name) and (DATA_DIR / "prices" / f"{name}.csv").exists()


def _read_series(path: Path, col: str = "close") -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df[col].astype(float).dropna()


def load_snapshot_prices(name: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(DATA_DIR / "prices" / f"{name}.csv", parse_dates=["date"], index_col="date")
    return df["close"].astype(float).dropna(), df["adj_close"].astype(float).dropna()


def load_snapshot_series(group: str, name: str) -> pd.Series | None:
    path = DATA_DIR / group / f"{name}.csv"
    return _read_series(path) if path.exists() else None


def build_vol(series_by_tenor: dict[int, pd.Series | None],
              calendar: pd.DatetimeIndex) -> pd.Series | pd.DataFrame | None:
    """
    Align volatility-index series (in percent) to the price calendar, as
    decimals. One tenor gives a Series. Several tenors give a DataFrame whose
    gaps (for example VIX9D before 2011) are filled from the 30-day index.
    """
    aligned = {}
    for tenor, s in series_by_tenor.items():
        if s is None or s.notna().sum() < 50:
            continue
        aligned[tenor] = (s / 100.0).reindex(calendar.union(s.index)).sort_index().ffill().reindex(calendar)
    if not aligned:
        return None
    if len(aligned) == 1 or 30 not in aligned:
        return next(iter(aligned.values())) if 30 not in aligned else aligned[30]
    frame = pd.DataFrame(aligned)
    for tenor in frame.columns:
        frame[tenor] = frame[tenor].fillna(frame[30])
    return frame[sorted(frame.columns)]


def load_market(uni_name: str, source: str = "snapshot", custom_ticker: str | None = None,
                start=None, use_term_structure: bool = True) -> Market:
    """
    Load closes, volatility and rates for one universe entry.
    source: "snapshot", "live" or "synthetic". A live failure raises
    RuntimeError so the caller can fall back.
    """
    uni = UNIVERSE[uni_name]
    if uni["ticker"] == SYNTH_TICKER or source == "synthetic":
        price, iv = synthetic_market()
        return Market(price, price, iv, None, "synthetic")

    vol_map = dict(uni["vol"])
    if not use_term_structure and 30 in vol_map:
        vol_map = {30: vol_map[30]}

    if source == "snapshot" and snapshot_available(uni["snapshot"]):
        close, adj = load_snapshot_prices(uni["snapshot"])
        vols = {t: load_snapshot_series("vol", name) for t, (name, _) in vol_map.items()}
        vol = build_vol(vols, close.index)
        rate = None
        if uni["rate"]:
            irx = load_snapshot_series("rates", uni["rate"][0])
            if irx is not None:
                rate = (irx / 100.0).reindex(close.index.union(irx.index)).sort_index().ffill().reindex(close.index)
        return Market(close, adj, vol, rate, "snapshot")

    ticker = custom_ticker if uni["ticker"] is None else uni["ticker"]
    close, adj = fetch_prices(ticker, start=start)
    vols = {t: fetch_series(yahoo) for t, (_, yahoo) in vol_map.items()}
    vol = build_vol(vols, close.index)
    rate = None
    if uni["rate"]:
        irx = fetch_series(uni["rate"][1])
        if irx is not None:
            rate = (irx / 100.0).reindex(close.index.union(irx.index)).sort_index().ffill().reindex(close.index)
    return Market(close, adj, vol, rate, "live")


# ---------------------------------------------------------------------------
# Live data (Yahoo Finance)
# ---------------------------------------------------------------------------

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance sometimes returns MultiIndex columns even for one ticker."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_prices(ticker: str, start=None, end=None) -> tuple[pd.Series, pd.Series]:
    """
    Raw and dividend-adjusted daily closes. Raises RuntimeError if nothing
    comes back; the app catches it and falls back to the snapshot or the
    synthetic market.
    """
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError(f"No price data returned for {ticker!r}.")
    df = _flatten(df)
    close = df["Close"].dropna().astype(float)
    adj = df["Adj Close"].dropna().astype(float) if "Adj Close" in df.columns else close
    for s in (close, adj):
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        s.name = ticker
    return close, adj


def fetch_series(ticker: str | None) -> pd.Series | None:
    """A Yahoo series of closes (for example a vol index in percent), or None."""
    if ticker is None:
        return None
    try:
        import yfinance as yf
        df = yf.download(ticker, start="2004-01-01", auto_adjust=False, progress=False)
        if df is None or len(df) == 0:
            return None
        s = _flatten(df)["Close"].dropna().astype(float)
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        return s if s.notna().sum() > 50 else None
    except Exception:
        return None
