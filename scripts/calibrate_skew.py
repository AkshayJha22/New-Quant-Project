"""Fit the smile multiplier in pricing.py to a real option chain.

Input: an option-chain CSV in the format written by the
Implied-Volatility-Surface-Explorer repo's chain snapshot (expiry, days, type,
strike, bid, ask, ..., spot) and the 13-week T-bill yield. The script picks the
expiry closest to the target maturity, estimates the forward from put-call
parity, backs out implied volatilities of out-of-the-money mid prices, and fits

    vol(K) / ATM = 1 + slope*z + curve*(sqrt(z^2 + tip^2) - tip),
    z = ln(K / F) / (ATM * sqrt(T))

by least squares. It also reports the market's 25-delta and 10-delta skews and
the model-free variance-swap volatility (the VIX formula) of that expiry.

Usage:
  python scripts/calibrate_skew.py --chain SPY_chain_snapshot.csv --rate 0.0415 [--days 30]
  python scripts/calibrate_skew.py --smile data/spy_smile_2026-11-23.csv   # refit the stored slice
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pricing import SkewParams, smile_multiplier  # noqa: E402


def black(F, K, T, sig, df, kind):
    s = sig * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * s * s) / s
    d2 = d1 - s
    if kind == "C":
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_vol(price, F, K, T, df, kind):
    intrinsic = max(0.0, df * (F - K)) if kind == "C" else max(0.0, df * (K - F))
    if not price > intrinsic + 1e-6:
        return np.nan
    try:
        return brentq(lambda s: black(F, K, T, s, df, kind) - price, 1e-4, 5.0)
    except ValueError:
        return np.nan


def smile_from_chain(chain: pd.DataFrame, rate: float, target_days: int) -> tuple[pd.DataFrame, dict]:
    chain = chain.copy()
    chain["mid"] = (chain["bid"] + chain["ask"]) / 2
    days = chain["days"].unique()
    pick = int(days[np.argmin(np.abs(days - target_days))])
    g = chain[chain["days"] == pick]
    T = pick / 365
    df = np.exp(-rate * T)
    calls = g[g["type"] == "C"].set_index("strike")
    puts = g[g["type"] == "P"].set_index("strike")
    common = calls.index.intersection(puts.index)
    both = pd.DataFrame({"c": calls.loc[common, "mid"], "p": puts.loc[common, "mid"]}).dropna()
    both = both[(both.c > 0) & (both.p > 0)]
    k0 = (both.c - both.p).abs().idxmin()
    F = float(k0 + (both.loc[k0, "c"] - both.loc[k0, "p"]) / df)   # put-call parity
    otm = pd.concat([puts[puts.index < F].assign(type="P"), calls[calls.index >= F].assign(type="C")])
    otm = otm[(otm["bid"] > 0) & (otm["ask"] > otm["bid"])]
    otm = otm[(otm["ask"] - otm["bid"]) / otm["mid"] < 0.6]
    otm["iv"] = [implied_vol(m, F, K, T, df, k) for K, m, k in zip(otm.index, otm["mid"], otm["type"])]
    otm = otm.dropna(subset=["iv"]).sort_index()
    atm = float(np.interp(F, otm.index.values, otm["iv"].values))
    otm["z"] = np.log(otm.index.values / F) / (atm * np.sqrt(T))
    otm["multiplier"] = otm["iv"] / atm

    # market skews by delta, and the model-free variance-swap volatility
    d1 = (np.log(F / otm.index.values) + 0.5 * otm["iv"] ** 2 * T) / (otm["iv"] * np.sqrt(T))
    otm["delta"] = np.where(otm["type"] == "C", norm.cdf(d1), norm.cdf(d1) - 1) * df
    p, c = otm[otm["type"] == "P"], otm[otm["type"] == "C"]
    at = lambda frame, d: float(np.interp(d, np.abs(frame["delta"].values)[::-1] if frame is c else np.abs(frame["delta"].values),
                                          frame["iv"].values[::-1] if frame is c else frame["iv"].values))
    variance = 2 / T / df * np.trapezoid(otm["mid"].values / otm.index.values ** 2, otm.index.values)
    meta = {"expiry": str(g["expiry"].iloc[0]), "days": pick, "forward": F, "atm_vol": atm,
            "rate": rate, "spot": float(g["spot"].iloc[0]),
            "skew_points": {"put25": 100 * (at(p, 0.25) - atm), "call25": 100 * (at(c, 0.25) - atm),
                            "put10": 100 * (at(p, 0.10) - atm), "call10": 100 * (at(c, 0.10) - atm)},
            "variance_swap_vol": float(np.sqrt(variance)),
            "variance_swap_to_atm": float(np.sqrt(variance) / atm)}
    out = otm.reset_index().rename(columns={"index": "strike"})
    return out[["strike", "type", "bid", "ask", "mid", "iv", "delta", "z", "multiplier"]], meta


def fit(smile: pd.DataFrame) -> SkewParams:
    m = (smile["z"] > -4) & (smile["z"] < 3)
    z, y = smile.loc[m, "z"].values, smile.loc[m, "multiplier"].values
    sol = least_squares(lambda p: smile_multiplier(z, SkewParams(*p)) - y, [-0.25, 0.1, 0.3],
                        bounds=([-1, 0, 0.05], [0, 2, 3]))
    return SkewParams(*[round(float(v), 4) for v in sol.x])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain")
    ap.add_argument("--rate", type=float, default=0.04)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--smile", help="a stored smile slice (skips the chain step)")
    ap.add_argument("--save", help="write the smile slice and its metadata here")
    args = ap.parse_args()

    if args.smile:
        smile = pd.read_csv(args.smile)
        meta_path = Path(args.smile).with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    else:
        smile, meta = smile_from_chain(pd.read_csv(args.chain), args.rate, args.days)
        if args.save:
            smile.round(6).to_csv(args.save, index=False)
            Path(args.save).with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    params = fit(smile)
    print(json.dumps(meta, indent=2))
    print("fitted:", params)


if __name__ == "__main__":
    main()
