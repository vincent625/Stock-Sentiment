from __future__ import annotations

import numpy as np
import pandas as pd


def _weighted_window_signal(
    event_times_ns: np.ndarray,
    article_scores: np.ndarray,
    article_weights: np.ndarray,
    asof_times_ns: np.ndarray,
    lookback: pd.Timedelta,
    half_life: pd.Timedelta | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerically stable finite-window exponential weighting.

    Work is proportional to the number of events actually inside each lookback
    window; no exponent is evaluated over the full historical span.
    """
    signal = np.full(len(asof_times_ns), np.nan, dtype=float)
    counts = np.zeros(len(asof_times_ns), dtype=int)
    if len(event_times_ns) == 0:
        return signal, counts

    right = np.searchsorted(event_times_ns, asof_times_ns, side="right")
    left = np.searchsorted(event_times_ns, asof_times_ns - lookback.value, side="right")
    counts = right - left

    if half_life is None:
        half_life_seconds = None
    else:
        half_life_seconds = half_life.total_seconds()
        if half_life_seconds <= 0:
            raise ValueError("half_life must be positive")

    for idx, (lo, hi, asof_ns) in enumerate(zip(left, right, asof_times_ns)):
        if hi <= lo:
            continue
        base_weights = article_weights[lo:hi]
        if half_life_seconds is None:
            decay = 1.0
        else:
            age_seconds = (asof_ns - event_times_ns[lo:hi]) / 1e9
            decay = np.exp(-np.log(2.0) * age_seconds / half_life_seconds)
        weights = base_weights * decay
        denominator = float(np.sum(weights))
        if denominator <= 0:
            continue
        numerator = float(np.sum(article_scores[lo:hi] * decay))
        signal[idx] = numerator / denominator

    return signal, counts


def build_signal_panel(
    news: pd.DataFrame,
    observations: pd.DataFrame,
    lookbacks: list[str] | tuple[str, ...],
    half_life_fraction: float = 0.5,
    min_relevance: float = 0.0,
    min_novelty: float = 0.0,
    asof_col: str = "asof",
) -> pd.DataFrame:
    """Attach one sentiment signal column per lookback to ticker/as-of observations."""
    if not 0 < half_life_fraction <= 1:
        raise ValueError("half_life_fraction must be in (0, 1]")

    filtered = news[
        (news["relevance"] >= min_relevance) & (news["novelty"] >= min_novelty)
    ].copy()
    filtered["article_weight"] = (
        filtered["relevance"] * filtered["novelty"] * filtered["source_quality"]
    )
    filtered["article_score"] = filtered["sentiment"] * filtered["article_weight"]

    chunks: list[pd.DataFrame] = []
    for ticker, obs in observations.groupby("ticker", sort=False):
        obs = obs.sort_values(asof_col).copy()
        events = filtered[filtered["ticker"] == ticker].sort_values("first_timestamp_utc")

        event_times = events["first_timestamp_utc"].astype("int64").to_numpy()
        asof_times = obs[asof_col].astype("int64").to_numpy()
        article_scores = events["article_score"].to_numpy(dtype=float)
        article_weights = events["article_weight"].to_numpy(dtype=float)

        for lookback_label in lookbacks:
            lookback = pd.Timedelta(lookback_label)
            half_life = lookback * half_life_fraction
            signal, count = _weighted_window_signal(
                event_times,
                article_scores,
                article_weights,
                asof_times,
                lookback,
                half_life,
            )
            key = _safe_label(lookback_label)
            obs[f"signal_{key}"] = signal
            obs[f"news_count_{key}"] = count
        chunks.append(obs)

    if not chunks:
        return observations.copy()
    return pd.concat(chunks, ignore_index=True)


def _safe_label(value: str) -> str:
    return str(value).lower().replace(" ", "").replace("minutes", "min")
