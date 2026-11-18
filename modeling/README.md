# Modelling: candidate classifiers, comparison and Bayesian tuning

This stage takes the **pairs table** from feature engineering and trains classifiers that score
how likely each candidate road segment (arc) is the true one for a GPS point.
The probabilities are then used for Top-1 matching and as HMM emission scores.

## Input: the pairs table (agreed with feature engineering)

`data/processed/pairs.csv` (or `.parquet`), one row per (GPS point, candidate arc):

| column | meaning |
|---|---|
| `trip_id` | trip folder name, e.g. `00000048` (string). Used for GroupKFold |
| `partition` | `train` or `student_test` only. Evaluator trips must never be in this file |
| `point_idx` | 0-based row index of the GPS point in the trip's `.track` |
| `arc_id` | candidate arc (0-based row index in the trip's `.arcs`) |
| `label` | 1 if `arc_id` is the reference arc for this point, else 0 |
| `f_*` | features. Every column starting with `f_` is used automatically |

Points whose reference arc is missing from the candidate set must still be included (all labels 0),
so the candidate miss rate can be measured.

Route continuity needs the raw trips at `data/raw/{train,student_test}/<trip>/<trip>.arcs`
(change it with `--data-root`).

## Install

```bash
pip install -r modeling/requirements.txt
```

## Run (from the repo root)

```bash
# 0. (optional) test the pipeline before the real features are ready. Fake data, don't report it
python modeling/make_fake_pairs.py
python modeling/run_experiments.py --pairs data/processed/pairs_fake.csv --data-root none

# 1. compare all models (logreg, gnb, knn, rf, hgb, mlp, + lightgbm/xgboost if installed)
python modeling/run_experiments.py
python modeling/run_experiments.py --models rf,hgb      # subset

# 2. charts + report comparison table
python modeling/plot_results.py

# 3. Bayesian optimisation of the chosen model (TPE sampler, GroupKFold objective)
python modeling/tune_optuna.py --model hgb --trials 40 --objective macro_f1
python modeling/plot_results.py                          # now includes <model>_tuned
```

Delete `results/modeling/` and `artifacts/models/` after testing with fake data, so fake numbers never mix with real ones.

## Method

- **Cross-validation:** `GroupKFold(5)` grouped by `trip_id` on the 80 training trips, so no trip is in both the training and validation folds.
- **Student test:** the model is refit on all training trips and evaluated once on the 10 student-test trips.
- **Tuning:** Optuna TPE (Bayesian), objective = mean CV score on training trips only, with median pruning across folds. The student-test trips are not used during tuning.
- **Seeds:** fixed at 42 (`--seed`).

## Metrics (`metrics.py`)

| metric | level | definition |
|---|---|---|
| Macro F1 / precision / recall | candidate row | binary label at threshold 0.5, averaged over both classes |
| Log loss | candidate row | on P(label=1) |
| Top-1 accuracy (`top1_all`) | GPS point | the highest-probability candidate is the reference arc; points with no correct candidate count as wrong |
| Top-1 covered (`top1_covered`) | GPS point | same, but only over points whose reference arc was among the candidates |
| Candidate miss rate | GPS point | share of points with no correct candidate |
| Route continuity | trip | share of consecutive predicted arc changes where arc a's end node = arc b's start node |

## Outputs

| path | content |
|---|---|
| `results/modeling/results.csv` | one row per run: config, seed, CV mean±std, train and student-test metrics, timings, confusion matrix |
| `results/modeling/comparison_table.csv` | the report's comparison table (latest run per model) |
| `results/modeling/cv_folds/<model>.csv` | per-fold CV scores |
| `results/modeling/predictions/<model>_student_test.csv` | probability for every candidate → input for the HMM/Viterbi decoder |
| `results/modeling/plots/` | CV F1, student-test Top-1/F1, log loss, inference time, overfitting check, confusion matrices, training histories (MLP and HistGB only, the models that actually produce them) |
| `results/modeling/optuna/` | trial log CSV, SQLite study, best params, optimisation history plot |
| `artifacts/models/<model>.joblib` | `{"model", "features", ...}`, which the API loads |

## Files

- `common.py`: paths, schema validation and the leakage guard (rejects any partition other than train/student_test)
- `metrics.py`: all metrics
- `models.py`: model definitions and default hyperparameters (`build_model`)
- `experiment.py`: CV → refit → student-test evaluation → saving and logging
- `run_experiments.py`: model comparison
- `tune_optuna.py`: Bayesian hyperparameter optimisation
- `plot_results.py`: comparison charts and table
- `make_fake_pairs.py`: fake data for testing only
