# Historical results — preserved

The four CSVs in this directory are the original output snapshot and are preserved byte for byte. In particular, `mnl_parameter_estimates.csv` uses conventional row-independent standard errors; its zero p-values are historical floating-point display results, not exact zero probabilities.

For the independently checked respondent-clustered uncertainty, use [`../reference/mnl_inference/summary.json`](../reference/mnl_inference/summary.json) and the separately registered reproduction route in the root README. The Car constant's cluster normal-reference p-value is about0.231; the old conventional p-value about0.000348 must not be used as though it accounts for repeated responses. Point estimates, holdout observations and scenario outputs are separate from these inference assumptions. No commercial, causal fare-response or real-population validation claim follows.
