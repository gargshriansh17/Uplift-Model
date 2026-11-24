# Uplift Modelling on the Hillstrom Email Campaign

Which customers should receive a marketing email? This project builds and
rigorously evaluates uplift models that estimate the *incremental* effect of an
email on purchase conversion, rather than just who is likely to buy.

**Dataset:** the Hillstrom MineThatData E-Mail Analytics and Data Mining
Challenge (K. Hillstrom, 2008). It contains 64,000 customers who last purchased
within twelve months and were randomly assigned to a Men's Email, a Women's
Email or No Email; conversion is a purchase on the site in the following two
weeks. Download it from the
[original challenge page](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)
and save it as `data/raw/hillstrom.csv`.

## Key findings

- **The campaign works.** Conversion is 1.25% (Men's Email), 0.88% (Women's
  Email) and 0.57% (No Email).
- **Targeting models did not reliably beat random targeting.** Conversions are
  very sparse (about 0.5-1.3%), so individual-level uplift is hard to rank.
  Qini coefficients on validation were negative for every model evaluated, and
  all pairwise bootstrap intervals include zero.
- **LightGBM T-Learner was selected as the final model** because it was the most
  consistent: best validation Qini/AUUC and best mean Qini across 5 repeated
  out-of-fold runs. This is a "least bad / most consistent" choice, not a proven
  winner.
- **A fixed "Always Men's Email" policy is a strong benchmark.** In the
  three-arm policy evaluation the model-based policies did not significantly
  beat it (table below).

### Model selection (Men's Email vs No Email)

| Model | Validation Qini | Repeated OOF Qini (mean of 5) |
|---|---|---|
| LightGBM T-Learner | -0.037 | +0.016 |
| X-Learner | not run on validation | -0.014 |
| Logistic T-Learner | -0.066 | -0.025 |
| S-Learner | -0.083 | -0.038 |

### Three-arm policy value (validation split, doubly-robust estimate)

| Policy | Estimated conversion rate (95% CI) |
|---|---|
| Always No Email | 0.59% (0.37 - 0.83) |
| Always Women's Email | 0.89% (0.61 - 1.17) |
| Always Men's Email | 1.24% (0.92 - 1.56) |
| Logistic Regression policy | 1.18% (0.87 - 1.51) |
| LightGBM policy | 1.07% (0.78 - 1.38) |

## Final test result

The test split is scored once, by `python main.py final-test`. The model is the
LightGBM T-Learner, using the hyper-parameters chosen on validation and refit on
train + validation, with the Logistic T-Learner as baseline. Results are written
to `reports/final_test/`.

Test split: 8,523 customers, observed average effect 0.70 pp. Brackets show
95% bootstrap intervals.

| Metric | LightGBM T-Learner (final) | Logistic T-Learner (baseline) |
|---|---|---|
| Qini coefficient | 0.026 (-0.104, 0.160) | 0.013 (-0.118, 0.146) |
| AUUC (random = 0.352) | 0.378 | 0.365 |
| Uplift@10% | 1.40 pp (0.04, 2.90) | 1.47 pp (0.10, 3.02) |
| Uplift@20% | 0.82 pp (-0.12, 1.78) | 1.44 pp (0.32, 2.54) |
| Uplift@30% | 0.76 pp (0.05, 1.53) | 1.33 pp (0.56, 2.20) |

Neither model's Qini interval excludes zero, so there is no evidence that either
beats random targeting at this sample size. LightGBM was the pre-selected final
model, but the Logistic baseline was as good or better when targeting the top
20-30% of customers. The two models were not compared with a paired test, so the
differences should not be read as significant. Conversions are sparse (about
1%), which limits the precision of individual-level uplift estimates.

## Method

1. **Data and EDA** - quality checks, treatment/control balance, campaign
   summaries (`src/eda.py`, `src/data_preprocessing.py`). The two-arm data is
   split 60/20/20 into train (25,567), validation (8,523) and test (8,523).
2. **Models** - Logistic T-Learner, S-Learner, LightGBM T-Learner (tuned with
   5-fold CV on log loss), and an X-Learner in the repeated out-of-fold study.
3. **Evaluation** - Qini curve and coefficient, AUUC, Uplift@K, probability
   metrics (log loss, Brier), tie-aware ranking for LightGBM, 2,000-replication
   stratified bootstrap, and 5 repeated out-of-fold runs.
4. **Feature experiments** - baseline, log-history, interactions and
   no-history-segment variants compared by CV log loss.
5. **Three-arm extension** - pairwise Qini curves per contrast and policy value
   with IPW and doubly-robust estimators.
6. **Final test evaluation** - one-time scoring of the held-out test split. The
   model was fixed before this step, and the script checks that no test rows
   appear in the train or validation data.

## Project structure

```
src/
  config.py, data_preprocessing.py, feature_engineering.py, eda.py
  evaluation.py, model_comparison.py, bootstrap_evaluation.py
  repeated_oof_evaluation.py, final_test_evaluation.py
  models/   t_learner, s_learner, lightgbm_t_learner, three_arm_*
tests/      unit tests for the metrics and split safety
reports/    metrics, curves and figures
artifacts/  trained models (not committed)
data/       raw and processed data (not committed)
main.py     pipeline runner
```

## How to run

Requires Python 3.11 or later.

```bash
pip install -r requirements.txt
python main.py --list          # show stages
python main.py all             # reproduce model selection
python main.py final-test      # run ONCE, last
python -m pytest tests         # run unit tests
```

All random seeds are fixed (`RANDOM_STATE = 42` in `src/config.py`).

## Limitations

- Sparse conversions make every uplift estimate noisy; results are exploratory.
- Bootstrap intervals are conditional on the trained models (no refit
  uncertainty), and the final comparison between models is not a paired test.
- A positive Qini does not by itself prove a campaign is profitable; spend and
  costs are not modelled.
- 90% of LightGBM validation scores fall in tied groups, so its ranking is
  coarse.
- The final test evaluation covers Men's Email vs No Email only; the three-arm
  results are validation-split estimates.