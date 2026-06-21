"""Survival-analysis template utilities."""

from __future__ import annotations

import pandas as pd
from lifelines import CoxPHFitter


def fit_ipw_cox(df: pd.DataFrame, weights_col: str = "w_Group", time_col: str = "Duration_stage_total") -> pd.DataFrame:
    """Fit IPW-adjusted Cox model with exposure groups.

    The time origin and event definition must be verified locally before use.
    The manuscript interprets survival analyses as supportive rather than the
    sole basis for inference.
    """
    if time_col not in df.columns:
        raise KeyError(f"Time column not found: {time_col}")

    analysis = df.dropna(subset=[time_col, "asphyxia", "Group", weights_col]).copy()
    analysis = analysis[analysis[time_col] > 0].copy()
    analysis["time"] = analysis[time_col]
    analysis["event"] = analysis["asphyxia"].astype(int)
    analysis["Group"] = pd.Categorical(analysis["Group"], categories=[0, 1, 2, 3], ordered=True)
    cox_df = pd.get_dummies(analysis[["time", "event", "Group", weights_col]], columns=["Group"], drop_first=True)

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="time", event_col="event", weights_col=weights_col, robust=True)
    out = cph.summary.reset_index().rename(columns={"covariate": "term", "exp(coef)": "HR"})
    return out
