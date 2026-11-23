"""Download the market-data snapshot that ships with this repository.

GitHub's runners can reach Yahoo Finance and Cboe, so this script is run by
.github/workflows/data.yml, which uploads the CSVs as a workflow artifact.
The snapshot is then committed by hand, so the app, the tests and
validation.py work offline and reproducibly.

Files written:
  data/prices/<NAME>.csv  date, close, adj_close   SPY, QQQ, NSEI, GLD, GSPC
  data/vol/<NAME>.csv     date, close              VIX, VIX9D, VIX3M, VXN, INDIAVIX, GVZ
  data/cboe/<NAME>.csv    date, close              PUT, BXM (Cboe option-strategy benchmarks)
  data/rates/IRX.csv      date, close              13-week US T-bill yield, in percent
  data/manifest.json      source, rows, first and last date of every series

`close` is the raw closing price (adjusted for splits only): the price options
are written on. `adj_close` also folds dividends back in, so it is a
total-return series, used only for the buy-and-hold benchmark.

Usage:  python scripts/fetch_data.py [--start 2004-01-01]
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

PRICES = {"SPY": "SPY", "QQQ": "QQQ", "NSEI": "^NSEI", "GLD": "GLD", "GSPC": "^GSPC"}
VOL = {"VIX": "^VIX", "VIX9D": "^VIX9D", "VIX3M": "^VIX3M", "VXN": "^VXN",
       "INDIAVIX": "^INDIAVIX", "GVZ": "^GVZ"}
CBOE_BENCH = {"PUT": "^PUT", "BXM": "^BXM"}
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) research-data-snapshot"}


def from_yahoo(ticker: str, start: str, tries: int = 4) -> pd.DataFrame | None:
    """Daily raw and adjusted closes from Yahoo, with retries for rate limits."""
    import yfinance as yf
    for attempt in range(1, tries + 1):
        try:
            df = yf.download(ticker, start=start, auto_adjust=False,
                             progress=False, threads=False)
            if df is not None and len(df):
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                adj = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
                out = pd.DataFrame({"close": df["Close"].astype(float),
                                    "adj_close": adj.astype(float)})
                out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
                return out.dropna(subset=["close"])
            print(f"  yahoo {ticker}: empty (attempt {attempt})")
        except Exception as exc:  # network errors and rate limits
            print(f"  yahoo {ticker}: {type(exc).__name__}: {exc} (attempt {attempt})")
        time.sleep(6 * attempt)
    return None


def from_cboe(sym: str, start: str) -> pd.DataFrame | None:
    """Cboe's public daily history CSV (DATE plus CLOSE, or one value column)."""
    try:
        r = requests.get(CBOE_URL.format(sym=sym), headers=HEADERS, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip().upper() for c in df.columns]
        value_col = "CLOSE" if "CLOSE" in df.columns else df.columns[-1]
        out = pd.DataFrame({"close": pd.to_numeric(df[value_col], errors="coerce").values},
                           index=pd.to_datetime(df["DATE"], format="mixed"))
        out = out[out.index >= pd.Timestamp(start)].dropna()
        out["adj_close"] = out["close"]
        return out if len(out) else None
    except Exception as exc:
        print(f"  cboe {sym}: {type(exc).__name__}: {exc}")
        return None


def save(df: pd.DataFrame, path: Path, with_adj: bool) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    cols = ["close", "adj_close"] if with_adj else ["close"]
    out = df[cols].round(6)
    out.index.name = "date"
    out.to_csv(path, date_format="%Y-%m-%d")
    return {"rows": int(len(out)), "first": f"{out.index[0]:%Y-%m-%d}",
            "last": f"{out.index[-1]:%Y-%m-%d}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2004-01-01")
    args = ap.parse_args()

    manifest: dict = {"generated_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M"),
                      "series": {}}

    def record(group, name, df, source, with_adj):
        if df is None or not len(df):
            manifest["series"][f"{group}/{name}"] = {"source": None, "error": "no data"}
            print(f"FAILED {group}/{name}")
            return
        info = save(df, DATA / group / f"{name}.csv", with_adj)
        manifest["series"][f"{group}/{name}"] = {"source": source, **info}
        print(f"ok {group}/{name}: {source}, {info['rows']} rows, "
              f"{info['first']} to {info['last']}")

    for name, ticker in PRICES.items():
        record("prices", name, from_yahoo(ticker, args.start), f"yahoo:{ticker}", True)
        time.sleep(2)

    for name, ticker in VOL.items():
        df, src = None, None
        if name != "INDIAVIX":                      # Cboe publishes the US vol indices
            df, src = from_cboe(name, args.start), f"cboe:{name}"
        if df is None:
            df, src = from_yahoo(ticker, args.start), f"yahoo:{ticker}"
        record("vol", name, df, src, False)
        time.sleep(2)

    for name, ticker in CBOE_BENCH.items():
        df, src = from_cboe(name, args.start), f"cboe:{name}"
        if df is None:
            df, src = from_yahoo(ticker, args.start), f"yahoo:{ticker}"
        record("cboe", name, df, src, False)
        time.sleep(2)

    record("rates", "IRX", from_yahoo("^IRX", args.start), "yahoo:^IRX", False)

    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    missing = [k for k, v in manifest["series"].items() if not v.get("source")]
    print("missing:", missing or "none")


if __name__ == "__main__":
    main()
