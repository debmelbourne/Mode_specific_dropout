#!/usr/bin/env python
"""Compute E-values, BH-FDR labels, and optional random-effects pooling.

This script does not contain manuscript result CSVs. Provide local contrast-level
and/or cohort-level estimate files matching the schemas in the data directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from moa_dropout.evalues import evalue_table, add_bh_fdr
from moa_dropout.meta_analysis import random_effects_pool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contrast-estimates", help="CSV with columns contrast,aOR,ci_low,ci_high,nominal_p")
    parser.add_argument("--cohort-estimates", help="CSV with columns cohort,aOR,ci_low,ci_high")
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.contrast_estimates:
        contrasts = pd.read_csv(args.contrast_estimates)
        ev = evalue_table(contrasts)
        if "nominal_p" in ev.columns and "bh_fdr_q" not in ev.columns:
            ev["bh_fdr_q"] = add_bh_fdr(ev["nominal_p"])
        ev.to_csv(outdir / "evalues_multiplicity.csv", index=False)
        print("E-values and multiplicity labels")
        print(ev.to_string(index=False))
    else:
        print("No contrast-estimates file supplied; skipping E-values.")

    if args.cohort_estimates:
        cohorts = pd.read_csv(args.cohort_estimates)
        pooled = random_effects_pool(cohorts)
        pd.DataFrame([pooled]).to_csv(outdir / "random_effects_pooling.csv", index=False)
        print("\nRandom-effects pooling")
        print(pooled)
    else:
        print("No cohort-estimates file supplied; skipping random-effects pooling.")


if __name__ == "__main__":
    main()
