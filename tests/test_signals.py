import pandas as pd

from qqq_sentiment.io import normalize_news
from qqq_sentiment.signals import build_signal_panel


def test_signal_uses_only_past_news():
    news = normalize_news(
        pd.DataFrame(
            [
                {"story_id": "a", "first_timestamp_utc": "2025-01-01T10:00:00Z", "ticker": "AAPL", "sentiment": 1.0},
                {"story_id": "b", "first_timestamp_utc": "2025-01-01T10:10:00Z", "ticker": "AAPL", "sentiment": -1.0},
            ]
        )
    )
    obs = pd.DataFrame({"ticker": ["AAPL"], "asof": pd.to_datetime(["2025-01-01T10:05:00Z"], utc=True)})
    out = build_signal_panel(news, obs, ["15min"], half_life_fraction=1.0)
    assert out.loc[0, "news_count_15min"] == 1
    assert out.loc[0, "signal_15min"] > 0


def test_short_half_life_is_stable_over_long_history():
    news = normalize_news(
        pd.DataFrame(
            [
                {"story_id": "old", "first_timestamp_utc": "2020-01-01T10:00:00Z", "ticker": "AAPL", "sentiment": -1.0},
                {"story_id": "new", "first_timestamp_utc": "2026-01-01T10:00:00Z", "ticker": "AAPL", "sentiment": 0.8},
            ]
        )
    )
    obs = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "asof": pd.to_datetime(["2020-01-01T10:01:00Z", "2026-01-01T10:01:00Z"], utc=True),
        }
    )
    out = build_signal_panel(news, obs, ["5min"], half_life_fraction=0.5)
    assert out["signal_5min"].notna().all()
    assert out.loc[0, "signal_5min"] < 0
    assert out.loc[1, "signal_5min"] > 0
