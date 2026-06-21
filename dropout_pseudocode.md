# Mode-specific dropout pseudocode

For participant *i* and acquisition mode *m*:

```text
D_im = sum_t I(M_it = m and FHR_it = 0) / sum_t I(M_it = m)
```

where:

- `M_it` is the acquisition mode at sample `t`;
- `FHR_it = 0` denotes a raw zero-value sample in the backend hardware stream during active monitoring;
- display smoothing, interpolation, or visual bridging is not counted as valid signal recovery;
- mode-specific denominators are restricted to samples acquired in the relevant mode;
- simultaneous streams are not double-counted;
- when ECG is present and meets the prespecified duration rule, ECG samples are prioritised for ECG-mode dropout.

The four exposure groups are:

1. Ultrasound Low Dropout;
2. Ultrasound High Dropout;
3. ECG Low Dropout;
4. ECG High Dropout.

High dropout was prespecified as >30% signal loss in the primary analysis. Threshold-sensitivity analyses can be run using the TMLE/IPW threshold templates.

This file is a non-proprietary analytic definition. It is not a raw signal parser and does not implement a proprietary or real-time pipeline.
