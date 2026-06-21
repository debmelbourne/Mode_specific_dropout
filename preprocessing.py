"""Preprocessing utilities for the mode-specific dropout reproducibility package.

These functions operate on a schema-compatible tabular analysis dataset. They do
not parse raw CTG files and do not implement real-time monitoring/deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


GROUP_LABELS = {
    0: "US Low Dropout",
    1: "US High Dropout",
    2: "ECG Low Dropout",
    3: "ECG High Dropout",
}


@dataclass(frozen=True)
class AnalysisConfig:
    """Analysis settings aligned with the manuscript."""

    dropout_threshold: float = 30.0
    ecg_minutes_threshold: float = 10.0
    propensity_clip_lower: float = 0.01
    propensity_clip_upper: float = 0.99


def _first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalise_mode(series: pd.Series) -> pd.Series:
    """Map mode labels to 0=US and 1=ECG."""
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float).map({0.0: 0, 1.0: 1, 2.0: 1, 3.0: 1}).astype("Int64")
    text = s.astype(str).str.strip().str.lower()
    return text.map({
        "us": 0,
        "ultrasound": 0,
        "doppler": 0,
        "external ultrasound": 0,
        "ecg": 1,
        "fse": 1,
        "scalp ecg": 1,
        "fetal scalp ecg": 1,
        "internal ecg": 1,
    }).astype("Int64")


def assign_mode_and_dropout_groups(
    df: pd.DataFrame,
    config: AnalysisConfig = AnalysisConfig(),
    dropout_col: str | None = None,
) -> pd.DataFrame:
    """Assign the four exposure groups using mode and dropout threshold.

    Preferred dropout input is `mode_specific_dropout_pct`, derived using the
    manuscript formula D_im. Legacy aliases are accepted only for already-derived
    local analysis tables; users must verify that these columns correspond to the
    prespecified mode-specific dropout definition.

    Preferred mode inputs are either:
    - `primary_mode_of_acquisition` with values US/ECG or 0/1; or
    - `ECG_minutes`, from which ECG is assigned when ECG_minutes >= 10.
    """
    out = df.copy()

    if dropout_col is None:
        dropout_col = _first_existing(out, [
            "mode_specific_dropout_pct",
            "D_im",
            "dropout_pct",
            "DROPOUT_ALLMODES_ALLCHANNELS",
            "Dropout",
        ])
    if dropout_col is None or dropout_col not in out.columns:
        raise KeyError(
            "No dropout column found. Expected mode_specific_dropout_pct or a documented local alias."
        )

    mode_col = _first_existing(out, ["primary_mode_of_acquisition", "mode_of_acquisition", "Rec"])
    ecg_minutes_col = _first_existing(out, ["ECG_minutes", "ecg_minutes", "ecg_minutes_in_final_hour"])

    if mode_col is not None:
        rec = _normalise_mode(out[mode_col])
        if rec.isna().any():
            raise ValueError(f"Could not map all values in {mode_col} to US/ECG.")
        rec = rec.astype(int)
    elif ecg_minutes_col is not None:
        rec = (pd.to_numeric(out[ecg_minutes_col], errors="coerce") >= config.ecg_minutes_threshold).astype(int)
    else:
        raise KeyError(
            "Need primary_mode_of_acquisition/mode_of_acquisition/Rec or ECG_minutes to assign acquisition mode."
        )

    out["Rec"] = rec.astype(int)
    out["mode_specific_dropout_pct_used"] = pd.to_numeric(out[dropout_col], errors="coerce")
    out["dropout_binary"] = (out["mode_specific_dropout_pct_used"] > config.dropout_threshold).astype(int)
    out["Group"] = out["Rec"] * 2 + out["dropout_binary"]
    out["Group_label"] = out["Group"].map(GROUP_LABELS)
    return out


def create_analysis_features(df: pd.DataFrame, config: AnalysisConfig = AnalysisConfig()) -> pd.DataFrame:
    """Create harmonised variables used in the causal analyses."""
    out = assign_mode_and_dropout_groups(df, config=config)

    outcome_col = _first_existing(out, ["Asphyxia_composite", "asphyxia", "Outcome"])
    if outcome_col is None:
        raise KeyError("No outcome column found. Expected Asphyxia_composite/asphyxia/Outcome.")
    out["asphyxia"] = pd.to_numeric(out[outcome_col], errors="coerce").fillna(0).astype(int)

    maternal_age = pd.to_numeric(out.get("Maternal_age"), errors="coerce")
    out["Maternal_age_group"] = pd.cut(
        maternal_age,
        bins=[0, 35, 40, 120],
        labels=["<35", "35-39", ">=40"],
        include_lowest=True,
        right=False,
    )

    # Labour-duration ratio used in the manuscript's adjustment set.
    stage1_col = _first_existing(out, ["Duration_of_labour_1", "stage1", "dur_stage_1"])
    stage2_col = _first_existing(out, ["Duration_of_labour_2", "stage2", "dur_stage_2"])
    if stage1_col and stage2_col:
        stage2 = pd.to_numeric(out[stage2_col], errors="coerce").replace(0, np.nan)
        ratio = pd.to_numeric(out[stage1_col], errors="coerce") / stage2
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        ratio = ratio.where((ratio > 0) & (ratio <= 100))
        out["log_ratio_of_stages"] = np.log1p(ratio)
        out["quantile_ratio_of_stages"] = pd.qcut(
            out["log_ratio_of_stages"], q=5, labels=False, duplicates="drop"
        ) + 1
    else:
        out["log_ratio_of_stages"] = np.nan
        out["quantile_ratio_of_stages"] = np.nan

    bmi_col = _first_existing(out, ["Maternal_BMI", "BMI", "log_BMI"])
    if bmi_col == "log_BMI":
        out["log_BMI"] = pd.to_numeric(out[bmi_col], errors="coerce")
    elif bmi_col:
        out["log_BMI"] = np.log1p(pd.to_numeric(out[bmi_col], errors="coerce"))
    else:
        out["log_BMI"] = np.nan

    if "Parity" in out.columns:
        out["parity_collapsed"] = np.minimum(pd.to_numeric(out["Parity"], errors="coerce"), 3)

    # Harmonised names used downstream. Original columns are preserved.
    rename_map = {
        "Labour_onset_svsi": "labour_onset",
        "labour_svsi": "labour_onset",
        "Fetal_presention": "fetal_presentation",
        "Fetal_presentation": "fetal_presentation",
        "Meconeum": "meconium",
        "OBS_CODE": "primary_obs",
        "MOB": "mob_category",
        "Sentinel_SAFC": "sentinel_events",
    }
    for original, new in rename_map.items():
        if original in out.columns and new not in out.columns:
            out[new] = out[original]

    return out


def default_covariates(df: pd.DataFrame) -> list[str]:
    """Return covariates available in the provided dataset."""
    candidates = [
        "log_BMI",
        "Gestation",
        "Baby_weight",
        "Maternal_age_group",
        "quantile_ratio_of_stages",
        "primary_obs",
        "fetal_presentation",
        "mob_category",
        "parity_collapsed",
        "labour_onset",
        "sentinel_events",
        "Baby_gender",
        "meconium",
        "Regional_anaesthetic",
        "Fetal_position",
        "Maternal_position",
    ]
    return [c for c in candidates if c in df.columns]


def design_matrix(
    df: pd.DataFrame,
    covariates: list[str] | None = None,
    scale: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Create a numeric design matrix for propensity/outcome models."""
    if covariates is None:
        covariates = default_covariates(df)
    if not covariates:
        raise ValueError("No covariates available for design matrix.")

    X = df[covariates].copy()
    for col in X.columns:
        if X[col].dtype.name == "category" or X[col].dtype == object:
            continue
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = pd.get_dummies(X, drop_first=True, dummy_na=False)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean(numeric_only=True)).astype(float)

    if scale and X.shape[1] > 0:
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
    return X.astype(float), list(X.columns)
