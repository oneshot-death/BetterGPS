"""Evaluation metrics for candidate classification and route matching.

Two levels:
  * Row level (one row = one GPS point x one candidate arc): binary classification
    -> precision, recall, macro F1, log loss, confusion matrix.
  * Observation level (one GPS point): pick the candidate with the highest predicted
    probability -> Top-1 accuracy, candidate miss rate, route continuity.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)

GROUP = ["trip_id", "point_idx"]


def top1_predictions(df, proba):
    """For every GPS point return the highest-scoring candidate.

    Returns a dataframe with trip_id, point_idx, arc_id (predicted), proba, label
    (label = 1 if the predicted candidate is the reference arc) and has_true
    (whether the reference arc was among the candidates at all).
    """
    d = df[["trip_id", "partition", "point_idx", "arc_id", "label"]].copy()
    d["proba"] = np.asarray(proba)
    best = d.loc[d.groupby(GROUP, sort=True)["proba"].idxmax()].copy()
    has_true = d.groupby(GROUP, sort=True)["label"].max().to_numpy()
    best["has_true"] = has_true
    return best.reset_index(drop=True)


def top1_accuracy(best):
    """Share of GPS points whose top candidate is the reference arc.

    `all`     : every point counts (points with no correct candidate count as errors)
    `covered` : only points where the reference arc was in the candidate set
                (isolates classifier quality from candidate-generation misses)
    """
    covered = best[best["has_true"] == 1]
    return {
        "top1_all": float(best["label"].mean()),
        "top1_covered": float(covered["label"].mean()) if len(covered) else float("nan"),
    }


def candidate_miss_rate(df):
    """Share of GPS points whose reference arc is absent from the candidate set."""
    has_true = df.groupby(GROUP)["label"].max()
    return float(1.0 - has_true.mean())


def load_arcs(data_root, partition, trip_id):
    path = Path(data_root) / partition / trip_id / f"{trip_id}.arcs"
    if not path.exists():
        return None
    return np.loadtxt(path, dtype=np.int64, ndmin=2)


def route_continuity(best, data_root):
    """Share of consecutive predicted arc changes that are physically connected.

    Predicted arcs are ordered by point_idx; repeated arcs are collapsed. A change
    a -> b is continuous if arc a ends at the node where arc b starts.
    Returns NaN if the raw .arcs files cannot be found.
    """
    ok, total, missing = 0, 0, 0
    for (partition, trip), g in best.groupby(["partition", "trip_id"]):
        arcs = load_arcs(data_root, partition, trip)
        if arcs is None:
            missing += 1
            continue
        seq = g.sort_values("point_idx")["arc_id"].to_numpy()
        seq = seq[np.r_[True, seq[1:] != seq[:-1]]]          # collapse repeats
        if len(seq) < 2:
            continue
        a, b = seq[:-1], seq[1:]
        ok += int(np.sum(arcs[a, 1] == arcs[b, 0]))
        total += len(a)
    if missing:
        warnings.warn(f"route continuity: .arcs not found for {missing} trip(s) under {data_root}")
    return ok / total if total else float("nan")


def evaluate(df, proba, threshold=0.5, data_root=None):
    """Full metric set for one partition. `proba` = P(label=1) for each row of df."""
    y = df["label"].to_numpy()
    proba = np.clip(np.asarray(proba, dtype=float), 1e-7, 1 - 1e-7)
    pred = (proba >= threshold).astype(int)
    best = top1_predictions(df, proba)
    out = {
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        "f1_pos": f1_score(y, pred, pos_label=1, zero_division=0),
        "log_loss": log_loss(y, proba, labels=[0, 1]),
        "candidate_miss_rate": candidate_miss_rate(df),
        **top1_accuracy(best),
        "route_continuity": route_continuity(best, data_root) if data_root else float("nan"),
    }
    return out, best, confusion_matrix(y, pred, labels=[0, 1])
