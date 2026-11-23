"""
smoke_test.py: run `python smoke_test.py` before trusting anything.

The checks now live in tests/ and run under pytest; this script runs the fast
ones (everything except the Streamlit app test):

1. tests/test_pricing.py   normal CDF and inverse, Black-Scholes reference
                           values, put-call parity, strike from delta.
2. tests/test_skew.py      the volatility smile, the index-to-ATM conversion
                           and the term structure.
3. tests/test_engine.py    full backtests on the deterministic synthetic
                           market: accounting reconciles to the cent, and the
                           original engine's results are reproduced exactly
                           when the new features are off.
4. tests/test_data_and_validation.py   the data snapshot and the Cboe
                           validation.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    sys.exit(pytest.main(["-q", str(ROOT / "tests"), "--ignore", str(ROOT / "tests" / "test_app.py")]))
