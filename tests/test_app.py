"""The Streamlit app renders without exceptions (main page, all tabs built)."""
import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import ROOT


@pytest.mark.parametrize("underlying", ["SPY: S&P 500 ETF", "Demo: synthetic market (offline)"])
def test_app_renders(underlying):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    if underlying != at.selectbox[0].value:
        at.selectbox[0].set_value(underlying).run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert "CAGR" in labels and "Sharpe" in labels
    assert "Validation vs Cboe" in [t.label for t in at.tabs]
