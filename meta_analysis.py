"""Cohort-level random-effects pooling for log-ratio estimates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


def random_effects_pool(table: pd.DataFrame, estimate_col="aOR", lower_col="ci_low", upper_col="ci_high") -> dict:
    """DerSimonian-Laird random-effects pooling for cohort-level ratio estimates."""
    if table.empty:
        raise ValueError("Cohort-level estimate table is empty.")

    log_est = np.log(table[estimate_col].astype(float).values)
    se = (np.log(table[upper_col].astype(float).values) - np.log(table[lower_col].astype(float).values)) / (2 * 1.96)
    vi = se ** 2
    wi = 1 / vi
    fixed = np.sum(wi * log_est) / np.sum(wi)
    q = np.sum(wi * (log_est - fixed) ** 2)
    df = len(log_est) - 1
    c = np.sum(wi) - np.sum(wi ** 2) / np.sum(wi)
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    w_re = 1 / (vi + tau2)
    pooled = np.sum(w_re * log_est) / np.sum(w_re)
    se_pooled = np.sqrt(1 / np.sum(w_re))
    z = pooled / se_pooled
    return {
        "n_cohorts": int(len(table)),
        "pooled_aOR": float(np.exp(pooled)),
        "ci_low": float(np.exp(pooled - 1.96 * se_pooled)),
        "ci_high": float(np.exp(pooled + 1.96 * se_pooled)),
        "tau2": float(tau2),
        "I2_pct": float(max(0, (q - df) / q) * 100) if q > 0 else 0.0,
        "Q": float(q),
        "Q_p": float(chi2.sf(q, df)) if df > 0 else np.nan,
        "z_p": float(2 * norm.sf(abs(z))),
    }
