#!/usr/bin/env python
"""Run primary IPW analysis on an approved schema-compatible dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from moa_dropout.preprocessing import AnalysisConfig, create_analysis_features
from moa_dropout.ipw import run_primary_ipw, likelihood_ratio_test_group
from moa_dropout.preprocessing import design_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to approved schema-compatible analysis CSV")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--dropout-threshold", type=float, default=30.0)
    parser.add_argument("--clip-lower", type=float, default=0.01)
    parser.add_argument("--clip-upper", type=float, default=0.99)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    config = AnalysisConfig(
        dropout_threshold=args.dropout_threshold,
        propensity_clip_lower=args.clip_lower,
        propensity_clip_upper=args.clip_upper,
    )
    features = create_analysis_features(df, config=config)
    weighted, or_table = run_primary_ipw(features, config=config)
    X, _ = design_matrix(features)
    lrt = likelihood_ratio_test_group(weighted, X)

    group_counts = weighted.groupby(["Group", "Group_label"]).agg(
        n=("asphyxia", "size"),
        events=("asphyxia", "sum"),
        event_rate=("asphyxia", "mean"),
    ).reset_index()

    # Aggregate outputs only. Do not export individual-level weights by default.
    group_counts.to_csv(outdir / "group_counts_aggregate.csv", index=False)
    or_table.to_csv(outdir / "primary_ipw_or_table.csv", index=False)
    pd.DataFrame([lrt]).to_csv(outdir / "mode_dropout_likelihood_ratio_test.csv", index=False)

    print("Group counts")
    print(group_counts.to_string(index=False))
    print("\nPrimary IPW odds ratios")
    print(or_table.to_string(index=False))
    print("\nLikelihood-ratio test")
    print(lrt)


if __name__ == "__main__":
    main()
