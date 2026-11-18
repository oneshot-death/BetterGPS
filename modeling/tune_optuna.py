"""Bayesian hyperparameter optimisation (Optuna TPE) for the chosen model.

Objective: mean GroupKFold (by trip) score on the 80 training trips only.
The student-test partition is used ONLY once, after tuning, to compare tuned vs untuned.

Usage (from the repo root):
    python modeling/tune_optuna.py --model hgb --trials 40
    python modeling/tune_optuna.py --model lightgbm --trials 50 --objective top1_all

Outputs: results/modeling/optuna/<model>_trials.csv, <model>.db (full study),
<model>_history.png, <model>_best_params.json; a "<model>_tuned" row in results.csv
and artifacts/models/<model>_tuned.joblib.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402
import optuna  # noqa: E402

from common import DEFAULT_DATA_ROOT, DEFAULT_PAIRS, RESULTS_DIR, SEED, load_pairs, split_partitions  # noqa: E402
from experiment import cross_validate, fit_and_evaluate  # noqa: E402
from models import MODEL_SPECS, build_model  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MLP_LAYERS = {"64": (64,), "128": (128,), "64-32": (64, 32), "128-64": (128, 64),
              "128-64-32": (128, 64, 32)}


def search_space(name, trial):
    """Hyperparameter ranges per model. Log scales for rates/regularisation."""
    if name == "hgb":
        return dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 300, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
            max_features=trial.suggest_float("max_features", 0.5, 1.0),
        )
    if name == "rf":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 6, 40, log=True),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            max_features=trial.suggest_float("max_features", 0.3, 1.0),
        )
    if name == "lightgbm":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 200, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
    if name == "xgboost":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
    if name == "mlp":
        return dict(
            hidden_layer_sizes=MLP_LAYERS[trial.suggest_categorical("hidden_layer_sizes", list(MLP_LAYERS))],
            alpha=trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            learning_rate_init=trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        )
    if name == "knn":
        return dict(n_neighbors=trial.suggest_int("n_neighbors", 3, 101, log=True),
                    weights=trial.suggest_categorical("weights", ["uniform", "distance"]))
    if name == "logreg":
        return dict(C=trial.suggest_float("C", 1e-3, 100.0, log=True))
    raise ValueError(f"no search space defined for '{name}'")


def params_from_trial_values(name, values):
    """Turn stored trial params back into estimator kwargs."""
    p = dict(values)
    if name == "mlp":
        p["hidden_layer_sizes"] = MLP_LAYERS[p["hidden_layer_sizes"]]
    if name == "lightgbm":
        p["subsample_freq"] = 1
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=f"one of: {', '.join(MODEL_SPECS)}")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--objective", default="macro_f1", choices=["macro_f1", "top1_all"],
                    help="CV metric to maximise")
    ap.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    name = args.model

    df, feats = load_pairs(args.pairs)
    train, test = split_partitions(df)
    data_root = None if args.data_root.lower() == "none" else args.data_root
    max_rows = MODEL_SPECS[name]["max_train_rows"]

    out = RESULTS_DIR / "optuna"
    out.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        model = build_model(name, seed=args.seed, **search_space(name, trial))

        def on_fold(k, scores):          # report running mean so bad trials get pruned early
            trial.report(scores[args.objective].mean(), k)
            if trial.should_prune():
                raise optuna.TrialPruned()

        cv = cross_validate(model, train, feats, args.folds, max_rows, args.seed, on_fold)
        trial.set_user_attr("cv_macro_f1", float(cv["macro_f1"].mean()))
        trial.set_user_attr("cv_top1_all", float(cv["top1_all"].mean()))
        trial.set_user_attr("cv_log_loss", float(cv["log_loss"].mean()))
        trial.set_user_attr("cv_std", float(cv[args.objective].std()))
        return float(cv[args.objective].mean())

    study = optuna.create_study(
        study_name=f"{name}_{args.objective}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),        # Bayesian (Tree-structured Parzen Estimator)
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        storage=f"sqlite:///{out / f'{name}.db'}",
        load_if_exists=True,
    )
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    tune_time = time.perf_counter() - t0

    trials = study.trials_dataframe()
    trials.to_csv(out / f"{name}_trials.csv", index=False)
    best_params = params_from_trial_values(name, study.best_params)
    with open(out / f"{name}_best_params.json", "w") as f:
        json.dump({"objective": args.objective, "best_cv_value": study.best_value,
                   "best_params": study.best_params, "n_trials": len(study.trials),
                   "tuning_time_s": tune_time}, f, indent=2)
    plot_history(study, name, args.objective, out / f"{name}_history.png")
    print(f"\nbest CV {args.objective}: {study.best_value:.4f} with {study.best_params}")
    print(f"tuning took {tune_time:.0f}s over {len(study.trials)} trials")

    # Refit with the best configuration and evaluate once on student_test
    tuned = build_model(name, seed=args.seed, **best_params)
    fit_and_evaluate(name, tuned, train, test, feats, data_root, n_folds=args.folds,
                     max_rows=max_rows, seed=args.seed, tag=f"{name}_tuned",
                     extra={"tuning_trials": len(study.trials), "tuning_time_s": tune_time,
                            "tuning_objective": args.objective})
    print("Compare against the untuned row for the same model in results/modeling/results.csv "
          "(or run plot_results.py).")


def plot_history(study, name, objective, path):
    done = [t for t in study.trials if t.value is not None]
    if not done:
        return
    xs = [t.number for t in done]
    ys = [t.value for t in done]
    best = [max(ys[: i + 1]) for i in range(len(ys))]
    pruned = [t.number for t in study.trials if t.state.name == "PRUNED"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.scatter(xs, ys, s=18, label="completed trial")
    ax.plot(xs, best, color="tab:red", label="best so far")
    if pruned:
        ax.scatter(pruned, [min(ys)] * len(pruned), marker="x", color="grey", s=18,
                   label="pruned trial")
    ax.set_xlabel("trial")
    ax.set_ylabel(f"CV {objective}")
    ax.set_title(f"Bayesian optimisation history: {name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
