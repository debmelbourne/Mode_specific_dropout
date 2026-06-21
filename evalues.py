"""E-value utilities for ratio estimates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def evalue_rr(rr: float) -> float:
    """Compute E-value for a ratio estimate.

    Ratios below 1 are inverted so the E-value is expressed on the harmful-ratio
    scale. Confidence intervals crossing the null should have a lower-bound
    E-value of 1.00.
    """
    rr = float(rr)
    if np.isnan(rr):
        return np.nan
    if rr < 1:
        rr = 1 / rr
    if rr == 1:
        return 1.0
    return float(rr + np.sqrt(rr * (rr - 1)))


def evalue_table(estimates: pd.DataFrame, estimate_col="aOR", lower_col="ci_low", upper_col="ci_high") -> pd.DataFrame:
    """Add E-values and lower-bound E-values to a contrast-estimate table."""
    rows = []
    for _, row in estimates.iterrows():
        est = float(row[estimate_col])
        low = float(row[lower_col])
        high = float(row[upper_col])
        crosses_null = low <= 1 <= high
        closest_to_null = low if est >= 1 else high
        rows.append({
            **row.to_dict(),
            "E_value": evalue_rr(est),
            "Lower_bound_E_value": 1.0 if crosses_null else evalue_rr(closest_to_null),
        })
    return pd.DataFrame(rows)


def add_bh_fdr(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR-adjusted q-values."""
    p = pd.Series(p_values, dtype=float)
    order = p.sort_values().index
    ranked = p.loc[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = q.iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
    out = pd.Series(index=p.index, dtype=float)
    out.loc[order] = q.values
    return out
