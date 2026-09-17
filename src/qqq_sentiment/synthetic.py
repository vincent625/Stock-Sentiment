from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_TICKERS = ["NVDA", "AAPL", "MSFT", "MU", "AMZN", "AMD", "GOOGL", "GOOG", "AVGO", "TSLA"]


def generate_demo(seed: int = 7, n_days: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a deterministic intraday demo with a known, decaying sentiment response."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-01-06", periods=n_days, tz="America/New_York")
    session_times = []
    for day in days:
        start = day.replace(hour=9, minute=30)
        session_times.append(pd.date_range(start, periods=390, freq="1min"))
    local_index = session_times[0].append(session_times[1:])
    timestamps = local_index.tz_convert("UTC")

    bench_ret = rng.normal(0.0, 0.00035, len(timestamps))
    price_rows = []
    qqq_close = 500.0 * np.exp(np.cumsum(bench_ret))
    for ts, close in zip(timestamps, qqq_close):
        price_rows.append(("QQQ", ts, close))

    news_rows = []
    story_counter = 0

    for ticker_idx, ticker in enumerate(DEFAULT_TICKERS):
        beta = 0.8 + 0.05 * ticker_idx
        stock_ret = beta * bench_ret + rng.normal(0.0, 0.00055, len(timestamps))

        candidate_idx = np.concatenate(
            [np.arange(day_idx * 390 + 60, day_idx * 390 + 300) for day_idx in range(n_days)]
        )
        chosen = np.sort(rng.choice(candidate_idx, size=max(25, n_days * 3), replace=False))
        for event_idx in chosen:
            sentiment = float(rng.uniform(-1.0, 1.0))
            story_counter += 1
            event_ts = timestamps[event_idx] + pd.Timedelta(seconds=20)
            news_rows.append(
                {
                    "story_id": f"demo-{story_counter}",
                    "first_timestamp_utc": event_ts,
                    "ticker": ticker,
                    "issuer_id": "ALPHABET" if ticker in {"GOOG", "GOOGL"} else ticker,
                    "sentiment": sentiment,
                    "relevance": float(rng.uniform(0.75, 1.0)),
                    "novelty": float(rng.uniform(0.75, 1.0)),
                    "source_quality": 1.0,
                    "event_type": "demo_event",
                    "source": "synthetic",
                }
            )

            # Most response arrives quickly, with a decaying tail through ~60 minutes.
            for lag in range(1, 61):
                idx = event_idx + lag
                if idx >= len(stock_ret):
                    break
                stock_ret[idx] += 0.00022 * sentiment * np.exp(-lag / 12.0)

        start_price = 80.0 + 15.0 * ticker_idx
        closes = start_price * np.exp(np.cumsum(stock_ret))
        for ts, close in zip(timestamps, closes):
            price_rows.append((ticker, ts, close))

    news = pd.DataFrame(news_rows)
    prices = pd.DataFrame(price_rows, columns=["ticker", "timestamp_utc", "close"])
    return news, prices
