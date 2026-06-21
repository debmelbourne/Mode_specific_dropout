#!/usr/bin/env python
"""External-validation random-effects pooling template."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from moa_dropout.meta_analysis import random_effects_pool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-estimates", required=True, help="CSV with columns cohort,aOR,ci_low,ci_high")
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    estimates = pd.read_csv(args.cohort_estimates)
    pooled = random_effects_pool(estimates)
    pd.DataFrame([pooled]).to_csv(outdir / "external_validation_random_effects.csv", index=False)
    print(pooled)


if __name__ == "__main__":
    main()
