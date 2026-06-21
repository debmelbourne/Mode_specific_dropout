"""TMLE-style binary-contrast template.

This module provides a non-proprietary template for the ECG High Dropout versus
US Low Dropout contrast. It uses gradient boosting nuisance learners with the
hyperparameters reported in the manuscript (learning_rate=0.1, max_depth=3,
n_estimators=100) and a simple logistic targeting step.

The function is intended for workflow transparency. It is not a clinical model
and is not deployment code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from .preprocessing import design_matrix


def _expit(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def run_binary_tmle_template(
    df: pd.DataFrame,
    treated_group: int = 3,
    reference_group: int = 0,
    clip: tuple[float, float] = (0.01, 0.99),
) -> pd.DataFrame:
    """Run a binary TMLE-style template and return marginal effect estimates.

    The returned estimates are intended for reproducibility review of the
    analytic workflow. Confidence intervals should be estimated using the
    prespecified influence-curve/bootstrapping approach approved for the local
    analysis environment.
    """
    d = df[df["Group"].isin([reference_group, treated_group])].copy()
    if d.empty:
        raise ValueError("No observations in requested binary contrast.")

    d["Treatment"] = (d["Group"] == treated_group).astype(int)
    d["Outcome"] = d["asphyxia"].astype(int)

    X, _ = design_matrix(d)
    A = d["Treatment"].to_numpy(dtype=int)
    Y = d["Outcome"].to_numpy(dtype=float)

    g_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    g_model.fit(X, A)
    g = np.clip(g_model.predict_proba(X)[:, 1], clip[0], clip[1])

    q_features = X.copy()
    q_features.insert(0, "Treatment", A)
    q_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    q_model.fit(q_features, Y)

    q_obs = np.clip(q_model.predict(q_features), 1e-6, 1 - 1e-6)
    q1_features = X.copy(); q1_features.insert(0, "Treatment", 1)
    q0_features = X.copy(); q0_features.insert(0, "Treatment", 0)
    q1 = np.clip(q_model.predict(q1_features), 1e-6, 1 - 1e-6)
    q0 = np.clip(q_model.predict(q0_features), 1e-6, 1 - 1e-6)

    H = A / g - (1 - A) / (1 - g)
    fluct = sm.GLM(Y, H.reshape(-1, 1), family=sm.families.Binomial(), offset=_logit(q_obs)).fit()
    epsilon = float(fluct.params[0])

    q1_star = _expit(_logit(q1) + epsilon * (1 / g))
    q0_star = _expit(_logit(q0) - epsilon * (1 / (1 - g)))

    risk_treated = float(np.mean(q1_star))
    risk_reference = float(np.mean(q0_star))
    odds_treated = risk_treated / (1 - risk_treated)
    odds_reference = risk_reference / (1 - risk_reference)

    return pd.DataFrame([{
        "contrast": f"Group {treated_group} vs Group {reference_group}",
        "n": int(len(d)),
        "risk_treated": risk_treated,
        "risk_reference": risk_reference,
        "risk_difference": risk_treated - risk_reference,
        "risk_ratio": risk_treated / risk_reference if risk_reference > 0 else np.nan,
        "odds_ratio": odds_treated / odds_reference if odds_reference > 0 else np.nan,
        "epsilon": epsilon,
        "note": "Template estimate; use approved local CI procedure for final inference.",
    }])
