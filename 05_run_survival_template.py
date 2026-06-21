#!/usr/bin/env python
"""Run IPW Cox survival-analysis template on an approved dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from moa_dropout.preprocessing import AnalysisConfig, create_analysis_features, design_matrix
from moa_dropout.ipw import estimate_stabilized_ipw
from moa_dropout.survival import fit_ipw_cox


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to approved schema-compatible analysis CSV")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--time-col", default="Duration_stage_total")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    features = create_analysis_features(df)
    X, _ = design_matrix(features)
    weighted = estimate_stabilized_ipw(features, X, config=AnalysisConfig())
    cox_table = fit_ipw_cox(weighted, time_col=args.time_col)
    cox_table.to_csv(outdir / "ipw_cox_survival_template.csv", index=False)
    print(cox_table.to_string(index=False))


if __name__ == "__main__":
    main()
