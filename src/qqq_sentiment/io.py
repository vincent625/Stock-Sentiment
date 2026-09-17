from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


NEWS_REQUIRED = {"story_id", "first_timestamp_utc", "ticker", "sentiment"}
PRICE_REQUIRED = {"ticker", "timestamp_utc", "close"}


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".csv.gz"} or path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output type: {path}")


def normalize_news(df: pd.DataFrame) -> pd.DataFrame:
    missing = NEWS_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"News data is missing required columns: {sorted(missing)}")

    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["story_id"] = out["story_id"].astype(str)
    out["first_timestamp_utc"] = pd.to_datetime(out["first_timestamp_utc"], utc=True)
    out["sentiment"] = pd.to_numeric(out["sentiment"], errors="coerce")

    for col in ("relevance", "novelty", "source_quality"):
        if col not in out:
            out[col] = 1.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(1.0).clip(0.0, 1.0)

    if "issuer_id" not in out:
        out["issuer_id"] = out["ticker"]
    if "event_type" not in out:
        out["event_type"] = "unknown"
    if "source" not in out:
        out["source"] = "unknown"

    out = out.dropna(subset=["first_timestamp_utc", "sentiment", "ticker"])
    out["sentiment"] = out["sentiment"].clip(-1.0, 1.0)

    # Keep only the earliest observation of the same story/security pair.
    out = (
        out.sort_values("first_timestamp_utc")
        .drop_duplicates(["story_id", "ticker"], keep="first")
        .reset_index(drop=True)
    )
    return out


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    missing = PRICE_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Price data is missing required columns: {sorted(missing)}")

    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["ticker", "timestamp_utc", "close"])
    out = out[out["close"] > 0]
    out = (
        out.sort_values(["ticker", "timestamp_utc"])
        .drop_duplicates(["ticker", "timestamp_utc"], keep="last")
        .reset_index(drop=True)
    )
    return out
