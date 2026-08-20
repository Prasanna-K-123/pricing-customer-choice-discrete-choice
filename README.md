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
- Statsmodels numerical Hessian utilities
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

## Reproduce

1. Install dependencies from `requirements.txt`.
2. Run the notebook from top to bottom, or use the core script in `src/`.
3. The data are downloaded from EPFL at runtime.
4. Verify that the estimated full-sample log likelihood is within **0.05** of the official benchmark before interpreting downstream results.

## Limitations

- Swissmetro is a **stated-preference** experiment involving a hypothetical transport mode.
- Multiple observations come from each respondent; the holdout correctly splits by respondent ID, but the conventional Hessian standard errors shown in the notebook do not adjust for within-respondent clustering.
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
