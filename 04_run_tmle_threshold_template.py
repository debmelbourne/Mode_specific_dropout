#!/usr/bin/env python
"""Run TMLE threshold-sensitivity template on an approved dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from moa_dropout.preprocessing import AnalysisConfig, create_analysis_features
from moa_dropout.tmle_template import run_binary_tmle_template


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to approved schema-compatible analysis CSV")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--thresholds", nargs="*", type=float, default=[5, 10, 15, 20, 25, 30])
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.input)

    rows = []
    for threshold in args.thresholds:
        features = create_analysis_features(raw, config=AnalysisConfig(dropout_threshold=threshold))
        try:
            result = run_binary_tmle_template(features)
            row = result.iloc[0].to_dict()
            row["threshold"] = threshold
            rows.append(row)
        except Exception as exc:
            rows.append({"threshold": threshold, "error": str(exc)})

    out = pd.DataFrame(rows)
    out.to_csv(outdir / "tmle_threshold_template_estimates.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
