"""Inverse-probability weighting utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from .preprocessing import AnalysisConfig, design_matrix


def estimate_stabilized_ipw(
    df: pd.DataFrame,
    X: pd.DataFrame,
    config: AnalysisConfig = AnalysisConfig(),
) -> pd.DataFrame:
    """Fit multinomial exposure model and compute stabilized IP weights."""
    out = df.copy()
    y_group = out["Group"].astype(int)

    model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=5000)
    model.fit(X, y_group)

    proba = pd.DataFrame(model.predict_proba(X), columns=model.classes_, index=out.index)
    observed_ps = np.array([proba.loc[idx, grp] for idx, grp in y_group.items()])
    observed_ps = np.clip(observed_ps, config.propensity_clip_lower, config.propensity_clip_upper)

    prevalence = y_group.value_counts(normalize=True).to_dict()
    numerator = y_group.map(prevalence).astype(float).values
    out["pscore_Group"] = observed_ps
    out["w_Group"] = numerator / observed_ps
    return out


def fit_ipw_logistic(
    df: pd.DataFrame,
    weights_col: str = "w_Group",
) -> pd.DataFrame:
    """Fit weighted logistic regression of outcome on exposure group.

    The reference category is US Low Dropout (Group 0). HC3 robust standard
    errors are used, matching the manuscript's non-proprietary statistical
    workflow.
    """
    analysis = df.dropna(subset=["asphyxia", "Group", weights_col]).copy()
    analysis["Group"] = pd.Categorical(analysis["Group"], categories=[0, 1, 2, 3], ordered=True)

    X = pd.get_dummies(analysis["Group"], prefix="Group", drop_first=True).astype(float)
    X = sm.add_constant(X, has_constant="add")
    y = analysis["asphyxia"].astype(float)
    weights = analysis[weights_col].astype(float)

    model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weights)
    result = model.fit(cov_type="HC3")

    params = result.params
    ci = result.conf_int()
    table = pd.DataFrame({
        "term": params.index,
        "log_or": params.values,
        "aOR": np.exp(params.values),
        "ci_low": np.exp(ci[0].values),
        "ci_high": np.exp(ci[1].values),
        "p_value": result.pvalues.values,
    })
    term_map = {
        "const": "Intercept",
        "Group_1": "US High Dropout vs US Low Dropout",
        "Group_2": "ECG Low Dropout vs US Low Dropout",
        "Group_3": "ECG High Dropout vs US Low Dropout",
    }
    table["contrast"] = table["term"].map(term_map).fillna(table["term"])
    return table


def run_primary_ipw(
    df: pd.DataFrame,
    config: AnalysisConfig = AnalysisConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end primary IPW analysis."""
    X, _ = design_matrix(df)
    weighted = estimate_stabilized_ipw(df, X, config=config)
    or_table = fit_ipw_logistic(weighted)
    return weighted, or_table


def likelihood_ratio_test_group(df: pd.DataFrame, X: pd.DataFrame, weights_col: str = "w_Group") -> dict:
    """Nested likelihood-ratio test comparing models with vs without exposure group."""
    from scipy.stats import chi2

    analysis = df.dropna(subset=["asphyxia", "Group", weights_col]).copy()
    y = analysis["asphyxia"].astype(float)
    weights = analysis[weights_col].astype(float)

    group_X = pd.get_dummies(pd.Categorical(analysis["Group"], categories=[0, 1, 2, 3], ordered=True),
                             prefix="Group", drop_first=True).astype(float)
    X_aligned = X.loc[analysis.index].astype(float)

    full = sm.add_constant(pd.concat([group_X.reset_index(drop=True), X_aligned.reset_index(drop=True)], axis=1), has_constant="add")
    nested = sm.add_constant(X_aligned.reset_index(drop=True), has_constant="add")

    full_result = sm.GLM(y.reset_index(drop=True), full, family=sm.families.Binomial(), freq_weights=weights.reset_index(drop=True)).fit()
    nested_result = sm.GLM(y.reset_index(drop=True), nested, family=sm.families.Binomial(), freq_weights=weights.reset_index(drop=True)).fit()

    lr_stat = 2 * (full_result.llf - nested_result.llf)
    df_diff = full_result.df_model - nested_result.df_model
    return {"chi_square": float(lr_stat), "df": float(df_diff), "p_value": float(chi2.sf(lr_stat, df_diff))}
