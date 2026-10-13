# Pricing & Customer Choice Strategy with Discrete Choice Modeling

Independent portfolio project using the **EPFL/Biogeme Swissmetro stated-preference choice dataset**.

## Objective

Use discrete-choice modeling to answer three decision questions:

1. How do travelers trade off **time and monetary cost** across Train, Swissmetro and Car?
2. Does a parsimonious multinomial-logit model generalize to **unseen respondents**?
3. How does modeled Swissmetro choice probability change under alternative fare scenarios?

## Data and sample

The raw Swissmetro file contains **10,728 observations**. The standard Biogeme sample restriction keeps trips with `PURPOSE` equal to 1 or 3 and valid choices, leaving:

- **6,768 usable choice observations**
- **752 respondents**
- alternatives: Train, Swissmetro, Car

The notebook also reproduces the standard availability and GA-travelcard cost treatment used in the benchmark specification.

## Model

The project estimates the standard four-parameter multinomial logit model:

- `ASC_TRAIN`
- `B_TIME`
- `B_COST`
- `ASC_CAR`

Swissmetro is the reference alternative for the alternative-specific constants.

Estimated parameters:

| Parameter | Estimate |
|---|---:|
| ASC_TRAIN | -0.7012 |
| B_TIME | -1.2779 |
| B_COST | -1.0838 |
| ASC_CAR | -0.1546 |

Time and cost coefficients are both negative, which is economically coherent.

## External implementation validation

The multinomial-logit likelihood is implemented directly and estimated with numerical maximum likelihood.

Final log likelihood:

**-5331.252007**

Official Biogeme benchmark:

**-5331.252000**

Absolute difference:

**0.000007**

SciPy reports a precision-loss warning near the optimum, but the benchmark match and coefficient agreement show that the implementation has reached the intended solution.

## Value of travel time

Because the generic time and cost variables use the same scaling, their coefficient ratio yields a model-implied time-money trade-off.

- Value of travel time: **70.74 CHF/hour**
- Willingness to pay for a 10-minute saving: **11.79 CHF**

These are model-implied values within this stated-preference experiment, not universal valuations.

## Respondent-level holdout validation

Swissmetro contains repeated choice tasks for each respondent. A row-level random split would leak the same respondent into both train and test data.

The project therefore splits by **respondent ID**:

- Train: 601 respondents / 5,409 observations
- Test: 151 respondents / 1,359 observations

A stronger baseline than equal-probability guessing is used: training-set alternative shares, adjusted for availability.

| Model | Test log-loss | Test accuracy |
|---|---:|---:|
| Training-share baseline | 0.8990 | 58.35% |
| Multinomial logit | **0.8794** | **65.86%** |

The MNL improves holdout log-loss by **2.18%** versus the training-share baseline.

The predictive improvement is modest; the project's main value is interpretable choice modeling and scenario analysis rather than claiming state-of-the-art prediction.

## Pricing sensitivity

For respondents without a GA travelcard, Swissmetro fare is varied from **0.60x to 1.40x** the base fare while holding other attributes fixed.

At the base fare:

- predicted Swissmetro share: **58.29%**

Approximate own-fare arc elasticity around the base:

**-0.45**

This indicates price-inelastic modeled demand around the base scenario.

Within the tested grid, the expected fare-revenue proxy is highest near:

- fare multiplier: **1.35x**
- predicted Swissmetro share: **49.18%**
- fare-revenue proxy index: **105.86**, where base = 100

This is **not** presented as an optimal commercial price. It is only the highest fare-revenue proxy observed within the tested scenario grid.

## Why the fare-revenue metric is a proxy

Swissmetro was hypothetical and the survey is stated preference. The scenario calculation does not include:

- realized market size
- operating cost
- capacity constraints
- competition
- implementation effects
- network effects
- demand dynamics

Therefore it is a sensitivity analysis, not a revenue forecast or profit optimization.

## Tech stack

- Python
- Pandas / NumPy
- SciPy numerical optimization
- Analytic NumPy/SciPy likelihood, information and respondent-cluster inference
- scikit-learn holdout evaluation
- Matplotlib
- maximum-likelihood estimation
- multinomial logit
- discrete-choice probability simulation

## Repository structure

```text
.
├── README.md
├── pricing_customer_choice_discrete_choice.ipynb
├── requirements.txt
├── src/
│   └── discrete_choice_pipeline.py
└── results/
    ├── mnl_parameter_estimates.csv
    ├── respondent_holdout_validation.csv
    ├── swissmetro_pricing_scenarios.csv
    └── project7_metrics.csv
```

## Inference and reproduction — verified 13 October 2026

The old five-column `results/mnl_parameter_estimates.csv` and original fitting source remain unchanged. The original CLI generates point estimates in a separate file and does not regenerate those historical inference columns. The new **optional** analytic inference/reproduction route supplies explicit conventional, row-sandwich and respondent-cluster uncertainty; it never writes inside committed `results/`.

The method and official source were registered before the new inference run: [protocol](docs/INFERENCE_REPRODUCTION_PROTOCOL.md), [exact input receipt](research_inference/INPUT_SOURCE.json), [complete result](reference/mnl_inference/summary.json). Thirteen substantive checks include a closed-form grouped covariance example, finite-difference gradient/information, availability, row/group-label permutation, malformed-input refusal and non-mutation. Recorded-data replay of the unchanged core pipeline matched historical point estimates, respondent-holdout metrics and fare scenarios within registered tolerances, with all six historical source/result hashes preserved. It uses the pinned official source bytes in the existing loader; the old HTTP acquisition route is not certified.

Repeated tasks from one respondent do not provide independent evidence. The new covariance sums the likelihood scores within each of the **752 respondents / 6,768 tasks**, then applies the explicitly registered finite-sample factor. This gives much wider uncertainty than the original row-independent table:

| Parameter | Estimate | Conventional SE | Respondent-cluster SE | Cluster normal-reference p |
|---|---:|---:|---:|---:|
| ASC_TRAIN | -0.701187 | 0.054874 | 0.183633 | 0.000134 |
| B_TIME | -1.277860 | 0.056883 | 0.237938 | 0.0000000785 |
| B_COST | -1.083791 | 0.051830 | 0.161312 | 0.0000000000183 |
| ASC_CAR | -0.154632 | 0.043235 | 0.129023 | 0.230726 |

The Car constant is no longer significant at a conventional 5% threshold in the respondent-cluster diagnostic; this adverse result is retained. The official Biogeme example's reported robust SE are **row-sandwich**, not this respondent-cluster calculation. New row-sandwich SE independently match that official table to its displayed precision. Clustering assumes respondents are independent and does not fix model specification, IIA, sampling bias or stated-versus-real behavior. The normal-reference p-values are asymptotic diagnostics, not guaranteed finite-sample coverage, causal fare effects or a commercial pricing recommendation.

The analytic optimizer reports a trust-region precision termination flag (`optimizer_success=false`), which is preserved. Its gradient infinity norm **2.10e-8** meets the **preregistered <=1e-6** stationarity criterion, the information is full-rank, and log likelihood matches the fixed official benchmark. No success flag or tolerance was changed after observing outcomes to conceal this diagnostic.

## Reproduce

For the verified inference and recorded-input replay, use a fresh environment and a **new nonexistent output directory**:

```bash
python -m pip install -r requirements-inference.txt
python -m unittest discover -s tests -v
python research_inference/run.py --output /tmp/swissmetro-inference
```

The runner fetches the immutable official repository input, verifies its SHA256, constructs the registered sample, produces `parameter_inference.csv` and `summary.json`, and refuses to overwrite an existing output directory or anything inside committed `results/`. Historical point/holdout/scenario replay runs in a temporary directory. The same dedicated [CI workflow](.github/workflows/inference.yml) executes this route. Full result replay uses the registered1e-8 float tolerance; integer identities, source/protocol/schema and historical file hashes are exact.

The original notebook, unpinned `requirements.txt` and `src/discrete_choice_pipeline.py` remain historical entry points. Use a disposable directory for them; they do not themselves implement the new clustered inference route or guarantee reproducing its results.

## Limitations

- Swissmetro is a **stated-preference** experiment involving a hypothetical transport mode.
- Multiple observations come from each respondent; the holdout correctly splits by respondent ID, but the original conventional inference CSV does not adjust for within-respondent clustering. The separate registered route now reports clustered uncertainty with its own assumptions.
- Standard multinomial logit assumes **independence of irrelevant alternatives (IIA)** and homogeneous generic time/cost coefficients.
- Value of time is model-implied within this survey context.
- Pricing scenarios are not commercial forecasts.
- The fare-revenue proxy excludes cost, capacity, competition and implementation effects.

## What this project demonstrates

- discrete-choice / choice modeling;
- multinomial-logit maximum-likelihood estimation;
- external benchmark replication;
- willingness-to-pay interpretation;
- respondent-level leakage control;
- probabilistic holdout evaluation;
- price elasticity;
- pricing-scenario analysis;
- careful separation of model output from commercial claims.

