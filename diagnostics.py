"""Covariate balance and stabilized-weight diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _weighted_mean(x: np.ndarray, w: np.ndarray | None = None) -> float:
    mask = ~np.isnan(x)
    x = x[mask]
    if w is not None:
        w = w[mask]
    if len(x) == 0:
        return np.nan
    if w is None:
        return float(np.mean(x))
    return float(np.average(x, weights=w))


def _weighted_var(x: np.ndarray, w: np.ndarray | None = None) -> float:
    mask = ~np.isnan(x)
    x = x[mask]
    if w is not None:
        w = w[mask]
    if len(x) <= 1:
        return np.nan
    m = _weighted_mean(x, w)
    if w is None:
        return float(np.var(x, ddof=1))
    return float(np.average((x - m) ** 2, weights=w))


def max_pairwise_smd(df: pd.DataFrame, X: pd.DataFrame, weights_col: str | None = None) -> pd.DataFrame:
    """Compute maximum pairwise absolute SMD across exposure groups."""
    rows = []
    groups = sorted(df["Group"].dropna().astype(int).unique())
    weights = df[weights_col].to_numpy(float) if weights_col else None

    for col in X.columns:
        vals = X[col].to_numpy(float)
        max_smd = 0.0
        for i, g1 in enumerate(groups):
            for g2 in groups[i + 1:]:
                m1 = df["Group"].astype(int).eq(g1).to_numpy()
                m2 = df["Group"].astype(int).eq(g2).to_numpy()
                w1 = weights[m1] if weights is not None else None
                w2 = weights[m2] if weights is not None else None
                mean1 = _weighted_mean(vals[m1], w1)
                mean2 = _weighted_mean(vals[m2], w2)
                var1 = _weighted_var(vals[m1], w1)
                var2 = _weighted_var(vals[m2], w2)
                pooled_sd = np.sqrt((var1 + var2) / 2) if np.isfinite(var1) and np.isfinite(var2) else np.nan
                smd = 0.0 if not np.isfinite(pooled_sd) or pooled_sd == 0 else abs(mean1 - mean2) / pooled_sd
                max_smd = max(max_smd, float(smd))
        rows.append({"covariate": col, "max_abs_smd": max_smd})
    return pd.DataFrame(rows)


def effective_sample_size(weights: pd.Series | np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    return float((w.sum() ** 2) / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else np.nan


def weight_diagnostics(df: pd.DataFrame, weights_col: str = "w_Group") -> pd.DataFrame:
    """Summarise stabilized weights by exposure group."""
    rows = []
    for group, d in df.groupby("Group", dropna=False):
        w = d[weights_col].astype(float)
        rows.append({
            "Group": group,
            "n": int(len(d)),
            "prevalence": float(len(d) / len(df)),
            "median_weight": float(w.median()),
            "iqr_low": float(w.quantile(0.25)),
            "iqr_high": float(w.quantile(0.75)),
            "p01": float(w.quantile(0.01)),
            "p99": float(w.quantile(0.99)),
            "min": float(w.min()),
            "max": float(w.max()),
            "prop_weights_gt_10": float((w > 10).mean()),
            "effective_sample_size": effective_sample_size(w),
            "ess_over_n": float(effective_sample_size(w) / len(d)) if len(d) else np.nan,
        })
    all_w = df[weights_col].astype(float)
    rows.append({
        "Group": "Overall",
        "n": int(len(df)),
        "prevalence": 1.0,
        "median_weight": float(all_w.median()),
        "iqr_low": float(all_w.quantile(0.25)),
        "iqr_high": float(all_w.quantile(0.75)),
        "p01": float(all_w.quantile(0.01)),
        "p99": float(all_w.quantile(0.99)),
        "min": float(all_w.min()),
        "max": float(all_w.max()),
        "prop_weights_gt_10": float((all_w > 10).mean()),
        "effective_sample_size": effective_sample_size(all_w),
        "ess_over_n": float(effective_sample_size(all_w) / len(df)) if len(df) else np.nan,
    })
    return pd.DataFrame(rows)
