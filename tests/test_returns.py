import numpy as np
import pandas as pd

from qqq_sentiment.returns import attach_forward_abnormal_returns


def test_market_adjusted_forward_return():
    times = pd.to_datetime(["2025-01-01T10:00:00Z", "2025-01-01T10:05:00Z"], utc=True)
    prices = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "QQQ", "QQQ"],
            "timestamp_utc": [times[0], times[1], times[0], times[1]],
            "close": [100.0, 102.0, 200.0, 202.0],
        }
    )
    obs = pd.DataFrame({"ticker": ["AAPL"], "asof": [times[0]], "anchor_close": [100.0]})
    out = attach_forward_abnormal_returns(obs, prices, ["5min"], "QQQ")
    expected = np.log(102 / 100) - np.log(202 / 200)
    assert np.isclose(out.loc[0, "abret_5min"], expected)
