# Mode-specific FHR signal dropout and perinatal asphyxia

## Non-proprietary reproducibility repository

This repository accompanies the manuscript:

**Mode-Specific Fetal Heart Rate Signal Dropout as a Marker of Perinatal Asphyxia: A Multicentre Causal Machine Learning Study**

The repository provides a non-proprietary statistical analysis framework for reviewing and rerunning the reported analytic workflow on an approved, schema-compatible dataset. It is not a clinical product, not a real-time monitoring system, and not a deployment repository.

## Scope

Included:

- analysis-dataset schema and variable dictionary;
- non-proprietary dropout-definition pseudocode;
- outcome-harmonisation notes;
- reusable Python modules for:
  - four-level mode-dropout exposure assignment;
  - multinomial propensity-score modelling;
  - stabilized inverse-probability weighting (IPW);
  - IPW logistic regression with HC3 robust standard errors;
  - post-IPW covariate-balance diagnostics;
  - stabilized-weight diagnostics;
  - E-values and Benjamini-Hochberg labels;
  - cohort-level random-effects pooling;
  - TMLE threshold-sensitivity template;
  - IPW Cox survival-analysis template.
- `requirements.txt`, `environment.yml`, and project metadata.

Not included:

- patient-level clinical data;
- raw CTG traces;
- individual-level derived outputs;
- synthetic patient data;
- result CSVs already reported in the manuscript or Supplementary Material;
- proprietary Philips/IntelliSpace extraction code;
- patent-sensitive feature-extraction, preprocessing, model-integration, deployment, or real-time decision-support code;
- trained deployment models;
- institutional identifiers or patient identifiers.

## Data

No patient-level data are included. Users with appropriate institutional approval may run the scripts on a local approved analysis dataset that matches `data/analysis_input_schema.csv` and `docs/data_dictionary.csv`.

The Czech CTU-UHB dataset is publicly available from its original source and should be accessed according to its own governance and licensing terms. Australian institutional data cannot be publicly released because of patient confidentiality and institutional governance requirements.

## Quick start

Install the package in editable mode:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run analyses requiring an approved schema-compatible individual-level dataset:

```bash
python scripts/01_run_primary_ipw.py --input /path/to/approved_analysis_dataset.csv --outdir outputs
python scripts/02_run_balance_weight_diagnostics.py --input /path/to/approved_analysis_dataset.csv --outdir outputs
python scripts/04_run_tmle_threshold_template.py --input /path/to/approved_analysis_dataset.csv --outdir outputs
python scripts/05_run_survival_template.py --input /path/to/approved_analysis_dataset.csv --outdir outputs
```

Run E-value or cohort-pooling utilities using locally prepared contrast-level or cohort-level estimates:

```bash
python scripts/03_run_evalues_and_meta.py \
  --contrast-estimates /path/to/contrast_estimates.csv \
  --cohort-estimates /path/to/cohort_estimates.csv \
  --outdir outputs
```

Input schemas are provided in:

- `data/analysis_input_schema.csv`
- `data/contrast_estimates_schema.csv`
- `data/cohort_estimates_schema.csv`

## Reproducibility boundary

The repository supports review of the statistical workflow. It does not reproduce the proprietary CTG extraction pipeline, raw signal parser, or real-time implementation. See `docs/reproducibility_boundary.md` and `NOTICE.md`.


Karmakar, D., Mendis, L., Keenan, E., Palaniswami, M., Makalic, E., & Brownfoot, F. (2026). Mode-specific FHR signal dropout and perinatal asphyxia: non-proprietary reproducibility package (V1.0.0). Zenodo. https://doi.org/10.5281/zenodo.20782840
