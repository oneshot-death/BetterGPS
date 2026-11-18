"""Shared experiment logic: group-aware CV, final fit, student-test evaluation, logging.

Used by run_experiments.py (model comparison) and tune_optuna.py (tuned model), so both
are trained and scored with exactly the same procedure and data partitions.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from common import MODEL_DIR, RESULTS_DIR, SEED
from metrics import evaluate
from models import MODEL_SPECS, describe, final_estimator

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def subsample(df, max_rows, seed=SEED):
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed)


def cross_validate(model, train, feats, n_folds=5, max_rows=None, seed=SEED, on_fold=None):
    """GroupKFold by trip_id on the training partition only.

    `on_fold(fold_idx, scores_so_far)` lets Optuna report intermediate values for pruning.
    Returns a dataframe with one row of metrics per fold.
    """
    gkf = GroupKFold(n_splits=n_folds)
    rows = []
    for k, (tr_idx, va_idx) in enumerate(gkf.split(train, groups=train["trip_id"])):
        tr = subsample(train.iloc[tr_idx], max_rows, seed)
        va = train.iloc[va_idx]
        m = clone(model)
        m.fit(tr[feats].to_numpy(), tr["label"].to_numpy())
        proba = m.predict_proba(va[feats].to_numpy())[:, 1]
        scores, _, _ = evaluate(va, proba)            # continuity not needed inside CV
        scores["fold"] = k
        scores["n_val_trips"] = va["trip_id"].nunique()
        rows.append(scores)
        if on_fold is not None:
            on_fold(k, pd.DataFrame(rows))
    return pd.DataFrame(rows)


def fit_and_evaluate(name, model, train, test, feats, data_root, n_folds=5,
                     max_rows=None, seed=SEED, tag=None, extra=None, cv=None):
    """CV on train -> refit on all train -> evaluate on student_test -> save everything.

    Pass a precomputed `cv` dataframe to skip re-running cross-validation.
    Returns the summary row that is appended to results.csv.
    """
    tag = tag or name
    out_dir = RESULTS_DIR
    for sub in ["cv_folds", "predictions", "plots"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {tag}: {describe(model)}")
    if cv is None:
        t0 = time.perf_counter()
        cv = cross_validate(model, train, feats, n_folds, max_rows, seed)
        print(f"  CV done in {time.perf_counter() - t0:.1f}s")
    cv.to_csv(out_dir / "cv_folds" / f"{tag}.csv", index=False)

    # Final fit on all 80 training trips
    tr = subsample(train, max_rows, seed)
    final = clone(model)
    t0 = time.perf_counter()
    final.fit(tr[feats].to_numpy(), tr["label"].to_numpy())
    train_time = time.perf_counter() - t0

    # Training-set score (to diagnose over/underfitting against CV and student-test)
    # (computed on up to 20 whole training trips to keep slow models like KNN fast)
    trips = pd.Series(train["trip_id"].unique()).sample(
        n=min(20, train["trip_id"].nunique()), random_state=seed)
    tr_eval = train[train["trip_id"].isin(trips)]
    train_scores, _, _ = evaluate(tr_eval, final.predict_proba(tr_eval[feats].to_numpy())[:, 1])

    # Student-test evaluation
    t0 = time.perf_counter()
    proba = final.predict_proba(test[feats].to_numpy())[:, 1]
    infer_time = time.perf_counter() - t0
    scores, best, cm = evaluate(test, proba, data_root=data_root)

    # Save artifacts
    joblib.dump({"model": final, "features": feats, "name": name, "tag": tag},
                MODEL_DIR / f"{tag}.joblib", compress=3)
    pred = test[["trip_id", "point_idx", "arc_id", "label"]].copy()
    pred["proba"] = proba                       # full candidate probabilities -> HMM emissions
    pred.to_csv(out_dir / "predictions" / f"{tag}_student_test.csv", index=False)
    plot_confusion(cm, tag, out_dir / "plots" / f"cm_{tag}.png")
    plot_training_history(final, tag, out_dir / "plots" / f"history_{tag}.png")

    row = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "model": tag,
        "base_model": name,
        "family": MODEL_SPECS[name]["family"],
        "config": describe(model),
        "seed": seed,
        "n_features": len(feats),
        "n_train_rows": len(tr),
        "cv_folds": n_folds,
        "cv_macro_f1_mean": cv["macro_f1"].mean(), "cv_macro_f1_std": cv["macro_f1"].std(),
        "cv_top1_mean": cv["top1_all"].mean(), "cv_top1_std": cv["top1_all"].std(),
        "cv_log_loss_mean": cv["log_loss"].mean(), "cv_log_loss_std": cv["log_loss"].std(),
        "train_macro_f1": train_scores["macro_f1"], "train_top1": train_scores["top1_all"],
        **{f"test_{k}": v for k, v in scores.items()},
        "train_time_s": train_time,
        "infer_time_s": infer_time,
        "infer_ms_per_1k_rows": 1000 * infer_time / len(test) * 1000,
        "confusion_matrix": json.dumps(cm.tolist()),
        **(extra or {}),
    }
    append_result(row)
    print(f"  CV macroF1 {row['cv_macro_f1_mean']:.4f}±{row['cv_macro_f1_std']:.4f} | "
          f"CV top1 {row['cv_top1_mean']:.4f} | test top1 {scores['top1_all']:.4f} | "
          f"test macroF1 {scores['macro_f1']:.4f} | test logloss {scores['log_loss']:.4f} | "
          f"continuity {scores['route_continuity']:.4f}")
    return row


def append_result(row):
    path = RESULTS_DIR / "results.csv"
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)


def plot_confusion(cm, tag, path):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["not route arc", "route arc"])
    ax.set_yticks([0, 1], ["not route arc", "route arc"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{tag} (student test)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_history(model, tag, path):
    """Only for models that genuinely produce epoch/iteration histories (MLP, HistGB)."""
    est = final_estimator(model)
    curves = {}
    if hasattr(est, "loss_curve_"):                                   # MLPClassifier
        curves["training loss"] = est.loss_curve_
        if getattr(est, "validation_scores_", None) is not None:
            curves["validation accuracy (right axis)"] = est.validation_scores_
    elif getattr(est, "train_score_", None) is not None and len(est.train_score_):  # HistGB
        curves["training log loss"] = -np.asarray(est.train_score_)
        if getattr(est, "validation_score_", None) is not None and len(est.validation_score_):
            curves["validation log loss"] = -np.asarray(est.validation_score_)
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax2 = None
    for label, values in curves.items():
        if "right axis" in label:
            ax2 = ax.twinx()
            ax2.plot(values, color="tab:orange", label=label)
            ax2.set_ylabel("accuracy")
        else:
            ax.plot(values, label=label)
    ax.set_xlabel("epoch" if hasattr(est, "loss_curve_") else "boosting iteration")
    ax.set_ylabel("loss")
    ax.set_title(f"Training history: {tag}")
    lines = ax.get_legend_handles_labels()
    if ax2 is not None:
        l2 = ax2.get_legend_handles_labels()
        lines = (lines[0] + l2[0], lines[1] + l2[1])
    ax.legend(*lines, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
