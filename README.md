# Uplift Modelling on the Hillstrom Email Campaign

Which customers should receive a marketing email? This project builds and evaluates uplift models that estimate the *incremental* effect of an email on purchase conversion, rather than just who is likely to buy.

> **Headline result: no reliable targeting uplift was found.**
> The email campaign itself clearly works, but none of the models tested can be shown to rank customers better than random. Every validation Qini coefficient is negative, and no final-test Qini interval excludes zero. A LightGBM T-Learner was carried forward as the "least bad" candidate, not as a proven winner. This is a negative result, reported as such.

**Dataset:** the Hillstrom MineThatData E-Mail Analytics and Data Mining Challenge (K. Hillstrom, 2008): 64,000 customers who last purchased within twelve months, randomly assigned to a Men's Email, a Women's Email or No Email. Conversion is a purchase on the site in the following two weeks. The dataset is **not included in this repo**; see [Getting the data](#getting-the-data).

## Key findings

- **The campaign works.** Conversion is 1.25% (Men's Email), 0.88% (Women's Email) and 0.57% (No Email).
- **No reliable targeting uplift found.** Conversions are very sparse (about 0.5-1.3%), so individual-level uplift is hard to rank. Qini coefficients on validation were negative for every model evaluated, and all pairwise bootstrap intervals include zero.
- **LightGBM T-Learner was carried forward as the "least bad" model.** It had the best validation Qini/AUUC and the best mean Qini across 5 repeated out-of-fold runs. This is a consistency-based choice, not evidence that it enables useful targeting.
- **The simple baseline does as well.** On the final test split, the Logistic T-Learner has similar or higher point estimates than LightGBM when targeting the top 10-30% of customers, and probability metrics are essentially identical (see below). The extra complexity of LightGBM is not justified by these results.
- **A fixed "Always Men's Email" policy is a strong benchmark.** In the three-arm policy evaluation the model-based policies did not significantly beat it.

### Model selection (Men's Email vs No Email)

| Model              | Validation Qini       | Repeated OOF Qini (mean of 5) |
| ------------------ | --------------------- | ----------------------------- |
| LightGBM T-Learner | -0.037                | +0.016                        |
| X-Learner          | not run on validation | -0.014                        |
| Logistic T-Learner | -0.066                | -0.025                        |
| S-Learner          | -0.083                | -0.038                        |

The X-Learner was evaluated only in the repeated out-of-fold study (mean Qini -0.014: below LightGBM, above the Logistic T-Learner and S-Learner). It was not run on the validation split, so it is not directly comparable to the validation column.

### Three-arm policy value (validation split, doubly-robust estimate)

| Policy                     | Estimated conversion rate (95% CI) |
| -------------------------- | ---------------------------------- |
| Always No Email            | 0.59% (0.37 - 0.83)                |
| Always Women's Email       | 0.89% (0.61 - 1.17)                |
| Always Men's Email         | 1.24% (0.92 - 1.56)                |
| Logistic Regression policy | 1.18% (0.87 - 1.51)                |
| LightGBM policy            | 1.07% (0.78 - 1.38)                |

## Final test result

The test split is scored once, by `python main.py final-test`. The model is the LightGBM T-Learner, using the hyper-parameters chosen on validation and refit on train + validation, with the Logistic T-Learner as baseline. Results are written to `reports/final_test/`.

Test split: 8,523 customers, observed average effect 0.70 pp. Brackets show 95% stratified bootstrap intervals (2,000 replications).

| Metric                | LightGBM T-Learner     | Logistic T-Learner (baseline) |
| --------------------- | ---------------------- | ----------------------------- |
| Qini coefficient      | 0.026 (-0.104, 0.160)  | 0.013 (-0.118, 0.146)         |
| AUUC (random = 0.352) | 0.378 (0.147, 0.614)   | 0.365 (0.139, 0.599)          |
| Uplift@10%            | 1.40 pp (0.04, 2.90)   | 1.47 pp (0.10, 3.02)          |
| Uplift@20%            | 0.82 pp (-0.12, 1.78)  | 1.44 pp (0.32, 2.54)          |
| Uplift@30%            | 0.76 pp (0.05, 1.53)   | 1.33 pp (0.56, 2.20)          |

Probability metrics on the test split (per arm) are nearly identical:

| Model              | Treatment log loss | Control log loss | Treatment Brier | Control Brier |
| ------------------ | ------------------ | ---------------- | --------------- | ------------- |
| LightGBM T-Learner | 0.0668             | 0.0343           | 0.01248         | 0.00560       |
| Logistic T-Learner | 0.0666             | 0.0344           | 0.01248         | 0.00560       |

How to read this:

- **No evidence of beating random targeting.** Neither model's Qini interval excludes zero at this sample size. The random-targeting AUUC (0.352) lies inside both AUUC intervals.
- **The baseline matches or beats the "final" model on point estimates.** The Logistic T-Learner's Uplift@K is similar at 10% and higher at 20% and 30%, but the intervals overlap, so none of these differences is established by this table alone. LightGBM was pre-selected before the test split was scored, and that choice was not changed afterwards.
- **No "better" claim between models is made.** The intervals above are each model's own bootstrap and say nothing about the *difference* between models. The paired comparison is described next.

### Paired comparison

`src/paired_bootstrap.py` resamples the same test rows for both models in every replication and reports the distribution of the difference (LightGBM minus Logistic) for Qini, AUUC and Uplift@10/20/30%:

```
python -m src.paired_bootstrap     # after python main.py final-test
```

Output: `reports/final_test/paired_bootstrap_summary.csv`. Until this has been run and its results added here, no statement about one model being better than the other should be read into this README.

## Method

1. **Data and EDA** - quality checks, treatment/control balance, campaign summaries (`src/eda.py`, `src/data_preprocessing.py`). The two-arm data is split 60/20/20 into train (25,567), validation (8,523) and test (8,523).
2. **Models** - Logistic T-Learner, S-Learner, LightGBM T-Learner (tuned with 5-fold CV on log loss, separately for the treatment and control arm), and an X-Learner in the repeated out-of-fold study.
3. **Evaluation** - Qini curve and coefficient, AUUC, Uplift@K, probability metrics (log loss, Brier), tie-aware ranking for LightGBM, 2,000-replication stratified bootstrap, and 5 repeated out-of-fold runs.
4. **Feature experiments** - baseline, log-history, interactions and no-history-segment variants compared by CV log loss.
5. **Three-arm extension** - pairwise Qini curves per contrast and policy value with IPW and doubly-robust estimators.
6. **Final test evaluation** - one-time scoring of the held-out test split. The model was fixed before this step, and the script checks that no test rows appear in the train or validation data.

## Project structure

```
src/
  config.py, data_preprocessing.py, feature_engineering.py, eda.py
  evaluation.py, model_comparison.py, bootstrap_evaluation.py
  repeated_oof_evaluation.py, final_test_evaluation.py, paired_bootstrap.py
  models/   t_learner, s_learner, lightgbm_t_learner, three_arm_*
tests/      unit tests for the metrics and split safety
reports/    metrics, curves and figures
artifacts/  trained models (not committed)
data/       raw and processed data (not committed)
main.py     pipeline runner
```

## Getting the data

The raw CSV is not committed. The project now uses **scikit-uplift** to fetch the canonical Hillstrom dataset and cache it locally:

```powershell
pip install -r requirements.txt
python -m src.dataset
```

This calls `sklift.datasets.fetch_hillstrom(target_col="all")`, downloads the upstream file if it is not already cached under `data/raw/scikit_uplift/`, validates the expected feature/target schema, and writes the normalized dataset to `data/raw/hillstrom.csv`. The original challenge remains available at the [MineThatData challenge page](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html).

## How to run

Requires Python 3.11 or later.

```
pip install -r requirements.txt
python main.py --list               # show stages
python main.py all                  # reproduce model selection
python main.py final-test           # run ONCE, last
python -m src.paired_bootstrap      # paired LightGBM vs Logistic comparison (needs final-test outputs)
python -m pytest tests              # run unit tests
```

All random seeds are fixed (`RANDOM_STATE = 42` in `src/config.py`).

### Hyper-parameters

**LightGBM T-Learner.** Fixed for all runs: `objective="binary"`, `min_child_samples=50`, `reg_lambda=5.0`, `random_state=42`. The remaining parameters were chosen per arm by 5-fold CV on log loss over the grid `n_estimators` in {50, 100}, `learning_rate` in {0.03, 0.05}, `num_leaves` in {3, 7}, `max_depth` in {2, 3}, and are re-used for the final train + validation refit.

| Arm       | `n_estimators` | `learning_rate` | `num_leaves` | `max_depth` | Best CV log loss |
| --------- | -------------- | --------------- | ------------ | ----------- | ---------------- |
| Treatment | 50             | 0.03            | 7            | 3           | 0.0670           |
| Control   | 50             | 0.05            | 3            | 2           | 0.0345           |

For the control arm, `max_depth` 2 and 3 tie at `num_leaves=3` (they are equivalent), and the first grid entry is kept. Many grid points are within one CV standard deviation of the best, so these settings are not sharply preferred.

**Logistic T-Learner.** `LogisticRegression(solver="lbfgs", class_weight=None, max_iter=2000)` with `C=0.01` for the treatment arm and `C=0.1` for the control arm.

## Limitations

- **Sparse conversions.** Conversion rates are around 1%, so every uplift estimate is noisy and the results are exploratory.
- **Interval coverage.** Bootstrap intervals are conditional on the trained models and exclude refit uncertainty, so they understate the total uncertainty.
- **Model comparison.** The per-model intervals do not test differences between models. Use the paired bootstrap above, and do not read any LightGBM-vs-Logistic difference as significant without it.
- **Coarse LightGBM ranking.** 90% of LightGBM validation scores fall in tied groups.
- **Scope of the final test.** The final test covers Men's Email vs No Email only; the three-arm results are validation-split estimates.
- **No cost modelling.** Spend, email cost and profit are not modelled. A positive Qini does not by itself prove a campaign is profitable.

## Planned work

- Report the results of the paired bootstrap in this README once run.
- Refit-based (full pipeline) bootstrap intervals that include training uncertainty.
- Run the X-Learner on the validation split so it is comparable with the other models.
- Extend the held-out evaluation to the Women's Email arm and add cost/spend modelling.


## Methodology upgrades (research-methodology-upgrades branch)

The following extensions are implemented as separate stages so the original results remain reproducible and the held-out test set stays isolated.

### X-Learner validation

The X-Learner now has a dedicated train-fit / validation-evaluate stage. Its outcome models are tuned by inner CV on the training split; imputed treatment effects are learned with inner-CV Ridge models. Validation predictions and scikit-uplift metrics are written to:

- `reports/x_learner_validation_predictions.csv`
- `reports/x_learner_validation_metrics.csv`

Run: `python -m src.x_learner_validation` or `python main.py x-learner-validation`.

### Nested cross-validation and multiple comparisons

`src/nested_cv.py` performs five outer stratified folds, with inner tuning confined to each outer training fold, for Logistic T-Learner, S-Learner, X-Learner and LightGBM T-Learner. This estimates model-selection performance without using the same fold for tuning and evaluation. It reports scikit-uplift normalized Qini AUC and uplift AUC alongside arm-specific log loss.

`src/multiple_comparison.py` applies Holm's step-down correction to paired outer-fold Qini comparisons. The corrected p-values are exploratory because five outer folds provide limited degrees of freedom; report effect sizes and uncertainty, not just significance.

Run:
```powershell
python -m src.nested_cv
python -m src.multiple_comparison
```

Outputs are saved under `reports/nested_cv/`. Nested CV is computationally more expensive than the original validation workflow.

### Formal treatment-effect heterogeneity test

`src/heterogeneity_test.py` tests the existing two-arm experiment only: Men's Email vs No Email. It compares a constant-effect linear probability model with a treatment-by-feature interaction model using an HC3-robust score test under the restricted constant-effect model. The intercept and treatment indicator are retained as mandatory controls; covariates and interaction directions aliased with those controls are removed using pivoted QR. The report includes retained/aliased covariate counts, estimable interaction count and effective degrees of freedom. It stops without a p-value if the robust score covariance for estimable directions is deficient. Women's Email is part of the separate three-arm evaluation and is not tested using the two-arm development files. This is a global test of the estimable linear interaction basis, not a guarantee that individual CATE estimates are accurate.

Run: `python -m src.heterogeneity_test`.

Output: `reports/heterogeneity/global_interaction_tests.csv`.

### Full three-arm held-out test

The existing three-arm split already reserves a test partition. The new `src/three_arm_final_test.py` refits the Logistic and validation-selected LightGBM arm models on train + validation, then evaluates both email contrasts against No Email and the learned/fixed campaign policies on the untouched three-arm test set. It reports scikit-uplift Qini/AUUC metrics, IPW and doubly-robust policy values, and customer-bootstrap confidence intervals.

Run this only after the three-arm training stages have been completed:
```powershell
python main.py three-arm-data three-arm-t-learner three-arm-lightgbm
python -m src.three_arm_final_test
```

Output directory: `reports/three_arm/final_test/`. The three-arm test is a one-time evaluation: do not tune models or change the model-selection decision based on these test results.

### CI

GitHub Actions runs `python -m pytest -q` on pushes and pull requests using Python 3.11. The test workflow installs the full requirements file, including scikit-uplift and statsmodels. CI tests are synthetic/unit tests; the workflow does not download Hillstrom or run the computationally expensive training pipeline.

### Updated run order

```powershell
python -m pip install -r requirements.txt
python main.py fetch-data
python main.py all
python main.py x-learner-validation
python main.py nested-cv
python main.py multiple-comparison
python main.py heterogeneity
# Train three-arm models, then score the three-arm test exactly once:
python main.py three-arm-data three-arm-t-learner three-arm-lightgbm
python main.py three-arm-final-test
python -m pytest -q
```

**Important:** New nested-CV, X-Learner validation, heterogeneity and three-arm final-test results are generated by running the commands above; no new numeric findings are claimed in this README until those reports exist. The previously reported two-arm test metrics remain historical results from the earlier pipeline.
