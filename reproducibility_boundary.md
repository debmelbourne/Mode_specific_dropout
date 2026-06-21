# Reproducibility boundary

This repository provides statistical workflow transparency for manuscript review.

## Included

- Variable dictionary and schema.
- Mode-specific dropout analytic definition.
- Exposure classification rules for the four mode-dropout groups.
- Statistical scripts/templates for IPW, TMLE, Cox survival analysis, balance diagnostics, E-values, multiplicity labels, and cohort-level pooling.
- Software-environment files.

## Excluded

- Patient-level institutional data.
- Raw CTG traces.
- Philips/IntelliSpace extraction code.
- Real-time signal processing implementation.
- Patent-sensitive feature extraction and preprocessing code.
- Model-integration and deployment code.
- Clinical decision-support software.
- Trained deployment models.
- Synthetic patient data.
- Result CSVs already reported in the manuscript and Supplementary Material.

## Rationale

This boundary supports reproducibility of the statistical workflow while protecting patient confidentiality, institutional-governance requirements, and pending intellectual property.
