from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .config import StudyConfig
from .io import normalize_news, normalize_prices
from .returns import (
    anchor_events_to_next_bar,
    attach_forward_abnormal_returns,
    event_baselines,
    rolling_beta_table,
)
from .signals import _safe_label, build_signal_panel
from .stats import add_fdr, chronological_fold_labels, summarize_cell


def _beta_table_if_needed(prices: pd.DataFrame, config: StudyConfig) -> pd.DataFrame | None:
    if config.returns.abnormal_method != "beta_adjusted":
        return None
    return rolling_beta_table(
        prices,
        benchmark_ticker=config.benchmark_ticker,
        window_bars=config.returns.beta_window_bars,
        min_periods=config.returns.beta_min_periods,
    )


def build_predictive_panel(news: pd.DataFrame, prices: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    news = normalize_news(news)
    prices = normalize_prices(prices)

    anchors = anchor_events_to_next_bar(news, prices)
    observations = (
        anchors[["ticker", "asof", "anchor_close"]]
        .drop_duplicates(["ticker", "asof"])
        .sort_values(["asof", "ticker"])
        .reset_index(drop=True)
    )
    signals = build_signal_panel(
        news,
        observations,
        lookbacks=config.lookbacks,
        half_life_fraction=config.signal.half_life_fraction,
        min_relevance=config.signal.min_relevance,
        min_novelty=config.signal.min_novelty,
        asof_col="asof",
    )
    betas = _beta_table_if_needed(prices, config)
    panel = attach_forward_abnormal_returns(
        signals,
        prices,
        horizons=config.horizons,
        benchmark_ticker=config.benchmark_ticker,
        base_time_col="asof",
        base_price_col="anchor_close",
        abnormal_method=config.returns.abnormal_method,
        beta_table=betas,
    )
    panel["block_date"] = panel["asof"].dt.tz_convert(config.market_timezone).dt.date
    panel["fold"] = chronological_fold_labels(
        panel["asof"], config.validation.n_folds, timezone=config.market_timezone
    )
    return panel


def grid_metrics(panel: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    """Cheap chronological-fold metrics used for hyperparameter stability/selection."""
    rows = []
    for fold in sorted(panel["fold"].dropna().unique()):
        subset = panel[panel["fold"] == fold]
        for lookback in config.lookbacks:
            signal_col = f"signal_{_safe_label(lookback)}"
            for horizon in config.horizons:
                return_col = f"abret_{_safe_label(horizon)}"
                clean = subset[[signal_col, return_col]].dropna()
                if len(clean) < 3 or clean[signal_col].nunique() < 2 or clean[return_col].nunique() < 2:
                    spearman = np.nan
                    pearson = np.nan
                else:
                    spearman = float(spearmanr(clean[signal_col], clean[return_col]).statistic)
                    pearson = float(pearsonr(clean[signal_col], clean[return_col]).statistic)
                rows.append(
                    {
                        "fold": int(fold),
                        "lookback": lookback,
                        "horizon": horizon,
                        "n": int(len(clean)),
                        "spearman": spearman,
                        "pearson": pearson,
                    }
                )
    return pd.DataFrame(rows)


def robust_test_grid_metrics(panel: pd.DataFrame, config: StudyConfig, test_fold: int) -> pd.DataFrame:
    """Bootstrap/HAC/FDR inference for the untouched final fold only."""
    subset = panel[panel["fold"] == test_fold]
    rows = []
    for lookback in config.lookbacks:
        signal_col = f"signal_{_safe_label(lookback)}"
        for horizon in config.horizons:
            return_col = f"abret_{_safe_label(horizon)}"
            metrics = summarize_cell(
                subset,
                signal_col=signal_col,
                return_col=return_col,
                block_col="block_date",
                n_boot=config.validation.bootstrap_samples,
                seed=config.validation.bootstrap_seed,
                hac_maxlags=config.validation.hac_maxlags,
            )
            rows.append({"fold": int(test_fold), "lookback": lookback, "horizon": horizon, **metrics})
    return add_fdr(pd.DataFrame(rows), p_col="bootstrap_p")


def aggregate_grid_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        fold_metrics.groupby(["lookback", "horizon"], as_index=False)
        .agg(
            mean_spearman=("spearman", "mean"),
            median_spearman=("spearman", "median"),
            std_spearman=("spearman", "std"),
            positive_fold_fraction=("spearman", lambda s: float((s > 0).mean())),
            mean_pearson=("pearson", "mean"),
            mean_n=("n", "mean"),
        )
    )
    grouped["stability_score"] = (
        grouped["mean_spearman"]
        * grouped["positive_fold_fraction"]
        / (1.0 + grouped["std_spearman"].fillna(0.0))
    )
    return grouped


def build_event_reaction_panel(news: pd.DataFrame, prices: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    news = normalize_news(news)
    prices = normalize_prices(prices)
    base = event_baselines(news, prices)
    events = news.merge(base, on=["story_id", "ticker", "first_timestamp_utc"], how="inner")
    observations = events[["story_id", "ticker", "first_timestamp_utc", "baseline_timestamp", "baseline_close"]].copy()
    observations = observations.rename(columns={"first_timestamp_utc": "asof"})

    signals = build_signal_panel(
        news,
        observations,
        lookbacks=config.lookbacks,
        half_life_fraction=config.signal.half_life_fraction,
        min_relevance=config.signal.min_relevance,
        min_novelty=config.signal.min_novelty,
        asof_col="asof",
    )
    betas = _beta_table_if_needed(prices, config)
    panel = attach_forward_abnormal_returns(
        signals,
        prices,
        horizons=config.horizons,
        benchmark_ticker=config.benchmark_ticker,
        base_time_col="asof",
        base_price_col="baseline_close",
        abnormal_method=config.returns.abnormal_method,
        beta_table=betas,
    )
    panel = panel.merge(
        news[["story_id", "ticker", "sentiment", "event_type", "relevance", "novelty", "source"]],
        on=["story_id", "ticker"],
        how="left",
    )
    panel["block_date"] = panel["asof"].dt.tz_convert(config.market_timezone).dt.date
    panel["fold"] = chronological_fold_labels(
        panel["asof"], config.validation.n_folds, timezone=config.market_timezone
    )
    return panel


def reaction_curve(
    event_panel: pd.DataFrame,
    lookback: str,
    horizons: list[str] | tuple[str, ...],
    fold: int | None = None,
    min_observations: int = 30,
) -> pd.DataFrame:
    df = event_panel.copy()
    if fold is not None:
        df = df[df["fold"] == fold]
    signal_col = f"signal_{_safe_label(lookback)}"

    clean_signal = df[signal_col].dropna()
    if len(clean_signal) < min_observations:
        return pd.DataFrame(columns=["horizon", "horizon_minutes", "spread_car", "n_top", "n_bottom"])

    try:
        df["sentiment_bucket"] = pd.qcut(df[signal_col], 5, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame(columns=["horizon", "horizon_minutes", "spread_car", "n_top", "n_bottom"])

    if df["sentiment_bucket"].nunique() < 5:
        return pd.DataFrame(columns=["horizon", "horizon_minutes", "spread_car", "n_top", "n_bottom"])

    rows = []
    for horizon in horizons:
        ret_col = f"abret_{_safe_label(horizon)}"
        top = df[df["sentiment_bucket"] == df["sentiment_bucket"].max()][ret_col].dropna()
        bottom = df[df["sentiment_bucket"] == df["sentiment_bucket"].min()][ret_col].dropna()
        if len(top) == 0 or len(bottom) == 0:
            continue
        rows.append(
            {
                "horizon": horizon,
                "horizon_minutes": pd.Timedelta(horizon).total_seconds() / 60.0,
                "spread_car": float(top.mean() - bottom.mean()),
                "n_top": int(len(top)),
                "n_bottom": int(len(bottom)),
            }
        )
    return pd.DataFrame(rows).sort_values("horizon_minutes").reset_index(drop=True)


def reaction_time_metrics(curve: pd.DataFrame, terminal_horizon: str | None = None) -> dict[str, float]:
    c = curve.sort_values("horizon_minutes").dropna(subset=["spread_car"]).copy()
    if c.empty:
        return {"T50_min": np.nan, "T80_min": np.nan, "terminal_spread": np.nan}

    if terminal_horizon is None:
        terminal = c.iloc[-1]
    else:
        terminal_minutes = pd.Timedelta(terminal_horizon).total_seconds() / 60.0
        eligible = c[c["horizon_minutes"] <= terminal_minutes]
        if eligible.empty:
            return {"T50_min": np.nan, "T80_min": np.nan, "terminal_spread": np.nan}
        terminal = eligible.iloc[-1]

    terminal_abs = abs(float(terminal["spread_car"]))
    if terminal_abs == 0:
        return {"T50_min": np.nan, "T80_min": np.nan, "terminal_spread": float(terminal["spread_car"])}

    out: dict[str, float] = {"terminal_spread": float(terminal["spread_car"])}
    abs_path = c["spread_car"].abs()
    for q in (0.5, 0.8):
        hit = c[abs_path >= q * terminal_abs]
        out[f"T{int(q * 100)}_min"] = float(hit.iloc[0]["horizon_minutes"]) if len(hit) else np.nan
    return out
