#!/usr/bin/env python
"""Run covariate balance and stabilized-weight diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from moa_dropout.preprocessing import AnalysisConfig, create_analysis_features, design_matrix
from moa_dropout.ipw import estimate_stabilized_ipw
from moa_dropout.diagnostics import max_pairwise_smd, weight_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to approved schema-compatible analysis CSV")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--clip-lower", type=float, default=0.01)
    parser.add_argument("--clip-upper", type=float, default=0.99)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    config = AnalysisConfig(propensity_clip_lower=args.clip_lower, propensity_clip_upper=args.clip_upper)
    features = create_analysis_features(df, config=config)
    X, _ = design_matrix(features)
    weighted = estimate_stabilized_ipw(features, X, config=config)

    pre = max_pairwise_smd(weighted, X, weights_col=None).rename(columns={"max_abs_smd": "pre_weighting_max_abs_smd"})
    post = max_pairwise_smd(weighted, X, weights_col="w_Group").rename(columns={"max_abs_smd": "post_weighting_max_abs_smd"})
    balance = pre.merge(post, on="covariate")
    balance["balanced_post_ipw"] = balance["post_weighting_max_abs_smd"] < 0.10

    weights = weight_diagnostics(weighted)

    balance.to_csv(outdir / "balance_diagnostics_aggregate.csv", index=False)
    weights.to_csv(outdir / "weight_diagnostics_aggregate.csv", index=False)

    print("Balance diagnostics")
    print(balance.to_string(index=False))
    print("\nWeight diagnostics")
    print(weights.to_string(index=False))


if __name__ == "__main__":
    main()
