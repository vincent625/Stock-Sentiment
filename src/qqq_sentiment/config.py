from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SignalConfig:
    half_life_fraction: float = 0.5
    min_relevance: float = 0.0
    min_novelty: float = 0.0


@dataclass(frozen=True)
class ReturnConfig:
    abnormal_method: str = "market_adjusted"
    beta_window_bars: int = 3900
    beta_min_periods: int = 390


@dataclass(frozen=True)
class ValidationConfig:
    n_folds: int = 5
    bootstrap_samples: int = 500
    bootstrap_seed: int = 42
    hac_maxlags: int = 5
    min_observations: int = 30


@dataclass(frozen=True)
class StudyConfig:
    benchmark_ticker: str = "QQQ"
    market_timezone: str = "America/New_York"
    lookbacks: tuple[str, ...] = (
        "5min", "15min", "30min", "1h", "2h", "4h", "1d", "3d", "5d"
    )
    horizons: tuple[str, ...] = (
        "1min", "5min", "15min", "30min", "1h", "2h", "4h", "1d"
    )
    signal: SignalConfig = field(default_factory=SignalConfig)
    returns: ReturnConfig = field(default_factory=ReturnConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)


def _section(cls, raw: dict[str, Any] | None):
    return cls(**(raw or {}))


def load_config(path: str | Path | None = None) -> StudyConfig:
    if path is None:
        return StudyConfig()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return StudyConfig(
        benchmark_ticker=raw.get("benchmark_ticker", "QQQ"),
        market_timezone=raw.get("market_timezone", "America/New_York"),
        lookbacks=tuple(raw.get("lookbacks", StudyConfig().lookbacks)),
        horizons=tuple(raw.get("horizons", StudyConfig().horizons)),
        signal=_section(SignalConfig, raw.get("signal")),
        returns=_section(ReturnConfig, raw.get("returns")),
        validation=_section(ValidationConfig, raw.get("validation")),
    )
