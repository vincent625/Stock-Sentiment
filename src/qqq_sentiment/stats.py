from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests


def chronological_fold_labels(times: pd.Series, n_folds: int, timezone: str = "UTC") -> pd.Series:
    """Assign chronological folds by whole market-calendar dates."""
    ts = pd.to_datetime(times, utc=True).dt.tz_convert(timezone)
    date_keys = ts.dt.floor("D")
    unique_dates = np.array(sorted(date_keys.dropna().unique()))
    if len(unique_dates) == 0:
        return pd.Series(index=times.index, dtype="Int64")
    n_folds = max(1, min(int(n_folds), len(unique_dates)))
    mapping: dict[pd.Timestamp, int] = {}
    for fold, date_block in enumerate(np.array_split(unique_dates, n_folds)):
        for date_key in date_block:
            mapping[pd.Timestamp(date_key)] = fold
    return date_keys.map(mapping).astype(int)


def block_bootstrap_spearman(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    block_col: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    clean = df[[x_col, y_col, block_col]].dropna()
    if clean.empty:
        return np.nan, np.nan, np.nan
    groups = [g for _, g in clean.groupby(block_col, sort=False)]
    if len(groups) < 2:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        indices = rng.integers(0, len(groups), size=len(groups))
        sample = pd.concat([groups[i] for i in indices], ignore_index=True)
        rho = spearmanr(sample[x_col], sample[y_col]).statistic
        if np.isfinite(rho):
            stats.append(rho)
    if not stats:
        return np.nan, np.nan, np.nan

    values = np.asarray(stats)
    lo, hi = np.quantile(values, [0.025, 0.975])
    left = int((values <= 0).sum())
    right = int((values >= 0).sum())
    p = 2.0 * (min(left, right) + 1) / (len(values) + 1)
    return float(lo), float(hi), float(min(p, 1.0))


def hac_standardized_slope(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    maxlags: int,
) -> tuple[float, float, float]:
    clean = df[[x_col, y_col]].dropna()
    if len(clean) < max(10, maxlags + 3):
        return np.nan, np.nan, np.nan

    x = clean[x_col].to_numpy(dtype=float)
    y = clean[y_col].to_numpy(dtype=float)
    x_std = x.std(ddof=0)
    y_std = y.std(ddof=0)
    if x_std == 0 or y_std == 0:
        return np.nan, np.nan, np.nan
    x = (x - x.mean()) / x_std
    y = (y - y.mean()) / y_std
    fit = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(fit.params[1]), float(fit.tvalues[1]), float(fit.pvalues[1])


def summarize_cell(
    df: pd.DataFrame,
    signal_col: str,
    return_col: str,
    block_col: str,
    n_boot: int,
    seed: int,
    hac_maxlags: int,
) -> dict[str, float]:
    clean = df[[signal_col, return_col, block_col]].dropna()
    n = len(clean)
    if n < 3 or clean[signal_col].nunique() < 2 or clean[return_col].nunique() < 2:
        return {
            "n": n,
            "spearman": np.nan,
            "pearson": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "bootstrap_p": np.nan,
            "hac_beta": np.nan,
            "hac_t": np.nan,
            "hac_p": np.nan,
        }

    sp = spearmanr(clean[signal_col], clean[return_col]).statistic
    pe = pearsonr(clean[signal_col], clean[return_col]).statistic
    lo, hi, boot_p = block_bootstrap_spearman(
        clean, signal_col, return_col, block_col, n_boot=n_boot, seed=seed
    )
    beta, hac_t, hac_p = hac_standardized_slope(clean, signal_col, return_col, maxlags=hac_maxlags)
    return {
        "n": int(n),
        "spearman": float(sp),
        "pearson": float(pe),
        "ci_low": lo,
        "ci_high": hi,
        "bootstrap_p": boot_p,
        "hac_beta": beta,
        "hac_t": hac_t,
        "hac_p": hac_p,
    }


def add_fdr(df: pd.DataFrame, p_col: str = "bootstrap_p") -> pd.DataFrame:
    out = df.copy()
    out["fdr_p"] = np.nan
    mask = out[p_col].notna()
    if mask.any():
        _, adjusted, _, _ = multipletests(out.loc[mask, p_col], method="fdr_bh")
        out.loc[mask, "fdr_p"] = adjusted
    return out
