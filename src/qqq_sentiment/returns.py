from __future__ import annotations

import numpy as np
import pandas as pd


def _merge_price(
    rows: pd.DataFrame,
    price_rows: pd.DataFrame,
    left_time_col: str,
    output_time_col: str,
    output_price_col: str,
    direction: str,
    allow_exact_matches: bool,
) -> pd.DataFrame:
    left = rows.sort_values(left_time_col).copy()
    right = price_rows[["timestamp_utc", "close"]].sort_values("timestamp_utc").copy()
    right = right.rename(columns={"timestamp_utc": output_time_col, "close": output_price_col})
    return pd.merge_asof(
        left,
        right,
        left_on=left_time_col,
        right_on=output_time_col,
        direction=direction,
        allow_exact_matches=allow_exact_matches,
    )


def anchor_events_to_next_bar(news: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Map each news event to the first stock bar strictly after publication."""
    out = []
    for ticker, events in news.groupby("ticker", sort=False):
        price_rows = prices[prices["ticker"] == ticker]
        if price_rows.empty:
            continue
        rows = events[["story_id", "ticker", "first_timestamp_utc"]].copy()
        merged = _merge_price(
            rows,
            price_rows,
            "first_timestamp_utc",
            "asof",
            "anchor_close",
            direction="forward",
            allow_exact_matches=False,
        )
        out.append(merged.dropna(subset=["asof", "anchor_close"]))
    if not out:
        return pd.DataFrame(columns=["story_id", "ticker", "first_timestamp_utc", "asof", "anchor_close"])
    return pd.concat(out, ignore_index=True)


def event_baselines(news: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Attach the last known stock price at or before each news timestamp."""
    out = []
    for ticker, events in news.groupby("ticker", sort=False):
        price_rows = prices[prices["ticker"] == ticker]
        if price_rows.empty:
            continue
        rows = events[["story_id", "ticker", "first_timestamp_utc"]].copy()
        merged = _merge_price(
            rows,
            price_rows,
            "first_timestamp_utc",
            "baseline_timestamp",
            "baseline_close",
            direction="backward",
            allow_exact_matches=True,
        )
        out.append(merged.dropna(subset=["baseline_timestamp", "baseline_close"]))
    if not out:
        return pd.DataFrame(columns=["story_id", "ticker", "first_timestamp_utc", "baseline_timestamp", "baseline_close"])
    return pd.concat(out, ignore_index=True)


def rolling_beta_table(
    prices: pd.DataFrame,
    benchmark_ticker: str,
    window_bars: int,
    min_periods: int,
) -> pd.DataFrame:
    bench = prices[prices["ticker"] == benchmark_ticker][["timestamp_utc", "close"]].copy()
    bench = bench.sort_values("timestamp_utc")
    bench["benchmark_return"] = np.log(bench["close"]).diff()

    chunks = []
    for ticker, px in prices[prices["ticker"] != benchmark_ticker].groupby("ticker", sort=False):
        px = px[["timestamp_utc", "close"]].sort_values("timestamp_utc").copy()
        px["stock_return"] = np.log(px["close"]).diff()
        joined = px.merge(bench[["timestamp_utc", "benchmark_return"]], on="timestamp_utc", how="inner")
        cov = joined["stock_return"].rolling(window_bars, min_periods=min_periods).cov(joined["benchmark_return"])
        var = joined["benchmark_return"].rolling(window_bars, min_periods=min_periods).var()
        joined["beta"] = cov / var
        joined["ticker"] = ticker
        chunks.append(joined[["ticker", "timestamp_utc", "beta"]])
    if not chunks:
        return pd.DataFrame(columns=["ticker", "timestamp_utc", "beta"])
    return pd.concat(chunks, ignore_index=True)


def attach_forward_abnormal_returns(
    observations: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: list[str] | tuple[str, ...],
    benchmark_ticker: str,
    base_time_col: str = "asof",
    base_price_col: str = "anchor_close",
    abnormal_method: str = "market_adjusted",
    beta_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if abnormal_method not in {"market_adjusted", "beta_adjusted"}:
        raise ValueError("abnormal_method must be market_adjusted or beta_adjusted")
    if abnormal_method == "beta_adjusted" and beta_table is None:
        raise ValueError("beta_table is required for beta_adjusted returns")

    benchmark = prices[prices["ticker"] == benchmark_ticker]
    if benchmark.empty:
        raise ValueError(f"Benchmark ticker {benchmark_ticker} is missing from prices")

    result_chunks = []
    for ticker, rows in observations.groupby("ticker", sort=False):
        stock = prices[prices["ticker"] == ticker]
        if stock.empty:
            continue

        part = rows.sort_values(base_time_col).copy()
        if base_price_col not in part:
            part = _merge_price(
                part,
                stock,
                base_time_col,
                "base_price_timestamp",
                base_price_col,
                direction="backward",
                allow_exact_matches=True,
            )

        part = _merge_price(
            part,
            benchmark,
            base_time_col,
            "benchmark_base_timestamp",
            "benchmark_base_close",
            direction="backward",
            allow_exact_matches=True,
        )

        if abnormal_method == "beta_adjusted":
            betas = beta_table[beta_table["ticker"] == ticker][["timestamp_utc", "beta"]].sort_values("timestamp_utc")
            part = pd.merge_asof(
                part.sort_values(base_time_col),
                betas,
                left_on=base_time_col,
                right_on="timestamp_utc",
                direction="backward",
                allow_exact_matches=True,
            ).drop(columns=["timestamp_utc"])
        else:
            part["beta"] = 1.0

        for horizon_label in horizons:
            horizon = pd.Timedelta(horizon_label)
            target_col = f"target_{_safe_label(horizon_label)}"
            part[target_col] = part[base_time_col] + horizon

            stock_target = _merge_price(
                part[[target_col]].copy(),
                stock,
                target_col,
                f"stock_target_timestamp_{_safe_label(horizon_label)}",
                f"stock_target_close_{_safe_label(horizon_label)}",
                direction="forward",
                allow_exact_matches=True,
            )
            bench_target = _merge_price(
                part[[target_col]].copy(),
                benchmark,
                target_col,
                f"benchmark_target_timestamp_{_safe_label(horizon_label)}",
                f"benchmark_target_close_{_safe_label(horizon_label)}",
                direction="forward",
                allow_exact_matches=True,
            )

            s_close = stock_target[f"stock_target_close_{_safe_label(horizon_label)}"].to_numpy()
            b_close = bench_target[f"benchmark_target_close_{_safe_label(horizon_label)}"].to_numpy()
            stock_return = np.log(s_close / part[base_price_col].to_numpy())
            benchmark_return = np.log(b_close / part["benchmark_base_close"].to_numpy())
            abnormal = stock_return - part["beta"].to_numpy() * benchmark_return
            part[f"abret_{_safe_label(horizon_label)}"] = abnormal

        result_chunks.append(part)

    if not result_chunks:
        return observations.copy()
    return pd.concat(result_chunks, ignore_index=True)


def _safe_label(value: str) -> str:
    return str(value).lower().replace(" ", "").replace("minutes", "min")
