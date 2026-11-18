"""Route reconstruction experiment: classifier-only vs classifier + HMM/Viterbi.

Steps
 1. Out-of-fold (OOF) classifier probabilities for the 80 training trips (GroupKFold by trip),
    so decoder parameters are tuned on predictions the model did not train on.
 2. Grid search of decoder parameters (beta, emission weight) on a subset of training trips,
    maximising observation-level Top-1 accuracy.
 3. Decode the 10 student-test trips with the chosen parameters and compare against
    classifier-only predictions: Top-1, route continuity, route precision/recall/F1.

Usage (from the repo root, after modeling/tune_optuna.py has produced the tuned model):
    python hmm/run_hmm.py --model-tag xgboost_tuned
    python hmm/run_hmm.py --model-tag xgboost --tune-trips 10      # faster

Outputs: results/hmm/
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "modeling"))

from common import DEFAULT_DATA_ROOT, DEFAULT_PAIRS, MODEL_DIR, RESULTS_DIR, SEED, load_pairs, split_partitions  # noqa: E402
from viterbi import classifier_only, full_route, load_trip, prepare_trip, viterbi  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PRED_DIR = RESULTS_DIR / "predictions"
OUT = RESULTS_DIR.parent / "hmm"


# ----------------------------------------------------------------------------- step 1
def oof_predictions(tag, pairs_path, folds, seed):
    """Out-of-fold P(label=1) for every training row, same GroupKFold as model CV."""
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold

    path = PRED_DIR / f"{tag}_train_oof.csv"
    if path.exists():
        print(f"using cached OOF predictions {path}")
        return pd.read_csv(path, dtype={"trip_id": str})
    art = joblib.load(MODEL_DIR / f"{tag}.joblib")
    df, _ = load_pairs(pairs_path)
    train, _ = split_partitions(df)
    feats = art["features"]
    proba = np.zeros(len(train))
    t0 = time.perf_counter()
    for k, (tr, va) in enumerate(GroupKFold(n_splits=folds).split(train, groups=train["trip_id"])):
        m = clone(art["model"])
        m.fit(train.iloc[tr][feats].to_numpy(), train.iloc[tr]["label"].to_numpy())
        proba[va] = m.predict_proba(train.iloc[va][feats].to_numpy())[:, 1]
        print(f"  OOF fold {k + 1}/{folds} done")
    print(f"OOF predictions in {time.perf_counter() - t0:.0f}s")
    out = train[["trip_id", "point_idx", "arc_id", "label"]].copy()
    out["proba"] = proba
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


# ----------------------------------------------------------------------------- metrics
def continuity(arc_seq, arcs):
    seq = arc_seq[np.r_[True, arc_seq[1:] != arc_seq[:-1]]]
    if len(seq) < 2:
        return np.nan, 0, 0
    ok = int(np.sum(arcs[seq[:-1], 1] == arcs[seq[1:], 0]))
    return ok / (len(seq) - 1), ok, len(seq) - 1


def route_prf(pred_arcs, ref_arcs, arc_len):
    """Length-weighted overlap between the set of predicted arcs and the reference route."""
    p, r = set(map(int, pred_arcs)), set(map(int, ref_arcs))
    inter = sum(arc_len[a] for a in p & r)
    prec = inter / max(sum(arc_len[a] for a in p), 1e-9)
    rec = inter / max(sum(arc_len[a] for a in r), 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return prec, rec, f1


def score(lat, choice, ref_route):
    arcs_chosen = np.array([lat.cand_arcs[t][k] for t, k in enumerate(choice)])
    correct = np.array([lat.labels[t][k] for t, k in enumerate(choice)]) if lat.labels else None
    cont, ok, n = continuity(arcs_chosen, lat.arcs)
    prec, rec, f1 = route_prf(arcs_chosen, ref_route, lat.arc_len) if ref_route is not None else (np.nan,) * 3
    return {
        "n_points": len(choice),
        "n_correct": int(correct.sum()) if correct is not None else np.nan,
        "top1": float(correct.mean()) if correct is not None else np.nan,
        "continuity": cont, "cont_ok": ok, "cont_n": n,
        "route_precision": prec, "route_recall": rec, "route_f1": f1,
    }, arcs_chosen


def full_route_scores(lat, route, ref_route):
    """Continuity and overlap of the reconstructed connected route (breaks = None)."""
    ok = n = 0
    for x, y in zip(route[:-1], route[1:]):
        if x is None or y is None:
            continue
        n += 1
        ok += int(lat.arcs[x, 1] == lat.arcs[y, 0])
    arcs_only = [a for a in route if a is not None]
    prec, rec, f1 = route_prf(arcs_only, ref_route, lat.arc_len) if ref_route is not None else (np.nan,) * 3
    return {"full_cont_ok": ok, "full_cont_n": n, "full_continuity": ok / n if n else np.nan,
            "full_breaks": sum(a is None for a in route),
            "full_route_precision": prec, "full_route_recall": rec, "full_route_f1": f1}


def build_lattices(preds, partition, trips, data_root, max_gap_s):
    lats, refs = {}, {}
    for i, trip in enumerate(trips):
        t0 = time.perf_counter()
        track, nodes, arcs = load_trip(data_root, partition, trip)
        lats[trip] = prepare_trip(preds[preds["trip_id"] == trip], track, nodes, arcs, trip,
                                  max_gap_s=max_gap_s)
        route_file = Path(data_root) / partition / trip / f"{trip}.route"
        refs[trip] = (np.loadtxt(route_file, dtype=np.int64, ndmin=1)
                      if route_file.exists() and route_file.stat().st_size else None)
        print(f"  [{i + 1}/{len(trips)}] {partition} {trip}: {len(lats[trip].point_idx)} points, "
              f"lattice built in {time.perf_counter() - t0:.1f}s")
    return lats, refs


def pooled(rows):
    df = pd.DataFrame(rows)
    return {
        "top1": df["n_correct"].sum() / df["n_points"].sum(),
        "continuity": df["cont_ok"].sum() / max(df["cont_n"].sum(), 1),
        "route_f1_mean": df["route_f1"].mean(),
    }


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", default="xgboost_tuned",
                    help="name of artifacts/models/<tag>.joblib and its student-test predictions")
    ap.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--tune-trips", type=int, default=20, help="training trips used for the decoder grid search")
    ap.add_argument("--betas", default="2,5,10,20,50")
    ap.add_argument("--weights", default="0.5,1,2")
    ap.add_argument("--max-gap", type=float, default=30.0, help="seconds; larger gaps split the sequence")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.model_tag

    # 1. OOF predictions on training trips
    oof = oof_predictions(tag, args.pairs, args.folds, args.seed)
    rng = np.random.default_rng(args.seed)
    all_train = sorted(oof["trip_id"].unique())
    tune_trips = sorted(rng.choice(all_train, size=min(args.tune_trips, len(all_train)), replace=False))
    print(f"\nbuilding lattices for {len(tune_trips)} training trips (decoder tuning)")
    tr_lats, tr_refs = build_lattices(oof, "train", tune_trips, args.data_root, args.max_gap)

    # 2. Grid search of decoder parameters on training trips
    betas = [float(b) for b in args.betas.split(",")]
    weights = [float(w) for w in args.weights.split(",")]
    grid = []
    base = pooled([score(tr_lats[t], classifier_only(tr_lats[t]), tr_refs[t])[0] for t in tune_trips])
    grid.append({"decoder": "classifier_only", "beta": np.nan, "emission_weight": np.nan, **base})
    for beta, w in itertools.product(betas, weights):
        rows = [score(tr_lats[t], viterbi(tr_lats[t], beta, w)[0], tr_refs[t])[0] for t in tune_trips]
        grid.append({"decoder": "hmm", "beta": beta, "emission_weight": w, **pooled(rows)})
        print(f"  beta={beta:<5} w={w:<4} train top1={grid[-1]['top1']:.4f} "
              f"continuity={grid[-1]['continuity']:.4f}")
    grid = pd.DataFrame(grid)
    grid.to_csv(OUT / "decoder_grid_train.csv", index=False)
    best = grid[grid["decoder"] == "hmm"].sort_values(["top1", "continuity"], ascending=False).iloc[0]
    beta, w = float(best["beta"]), float(best["emission_weight"])
    print(f"\nbest decoder params on training trips: beta={beta}, emission_weight={w} "
          f"(top1 {best['top1']:.4f} vs classifier-only {base['top1']:.4f})")

    # 3. Student-test comparison
    test_preds = pd.read_csv(PRED_DIR / f"{tag}_student_test.csv", dtype={"trip_id": str})
    test_trips = sorted(test_preds["trip_id"].unique())
    print(f"\nbuilding lattices for {len(test_trips)} student-test trips")
    te_lats, te_refs = build_lattices(test_preds, "student_test", test_trips, args.data_root, args.max_gap)

    per_trip, decoded, routes = [], [], []
    for trip in test_trips:
        lat = te_lats[trip]
        t0 = time.perf_counter()
        choice_h, seg = viterbi(lat, beta, w)
        dec_time = time.perf_counter() - t0
        s_c, arcs_c = score(lat, classifier_only(lat), te_refs[trip])
        s_h, arcs_h = score(lat, choice_h, te_refs[trip])
        route = full_route(lat, arcs_h, seg)
        s_f = full_route_scores(lat, route, te_refs[trip])
        routes.append(pd.DataFrame({"trip_id": trip, "order": np.arange(len(route)),
                                    "arc_id": [-1 if a is None else a for a in route]}))
        per_trip.append({"trip_id": trip, "n_points": s_c["n_points"],
                         **{f"clf_{k}": v for k, v in s_c.items() if k != "n_points"},
                         **{f"hmm_{k}": v for k, v in s_h.items() if k != "n_points"},
                         **{f"hmm_{k}": v for k, v in s_f.items()},
                         "hmm_segments": int(seg.max() + 1),
                         "hmm_gap_splits": int(lat.gap_break.sum()),
                         "viterbi_time_s": dec_time})
        decoded.append(pd.DataFrame({"trip_id": trip, "point_idx": lat.point_idx,
                                     "arc_classifier": arcs_c, "arc_hmm": arcs_h, "segment": seg}))
    per_trip = pd.DataFrame(per_trip)
    per_trip.to_csv(OUT / "student_test_per_trip.csv", index=False)
    pd.concat(decoded).to_csv(OUT / "student_test_decoded.csv", index=False)
    pd.concat(routes).to_csv(OUT / "student_test_full_routes.csv", index=False)   # -1 = break

    def pool(prefix):
        return {
            "top1": per_trip[f"{prefix}_n_correct"].sum() / per_trip["n_points"].sum(),
            "route_continuity": per_trip[f"{prefix}_cont_ok"].sum() / per_trip[f"{prefix}_cont_n"].sum(),
            "route_precision_mean": per_trip[f"{prefix}_route_precision"].mean(),
            "route_recall_mean": per_trip[f"{prefix}_route_recall"].mean(),
            "route_f1_mean": per_trip[f"{prefix}_route_f1"].mean(),
        }

    full = {
        "route_continuity": per_trip["hmm_full_cont_ok"].sum() / per_trip["hmm_full_cont_n"].sum(),
        "route_precision_mean": per_trip["hmm_full_route_precision"].mean(),
        "route_recall_mean": per_trip["hmm_full_route_recall"].mean(),
        "route_f1_mean": per_trip["hmm_full_route_f1"].mean(),
        "breaks_total": int(per_trip["hmm_full_breaks"].sum()),
    }
    summary = {"model_tag": tag, "beta": beta, "emission_weight": w, "max_gap_s": args.max_gap,
               "tune_trips": list(map(str, tune_trips)),
               "student_test_classifier_only": pool("clf"), "student_test_hmm": pool("hmm"),
               "student_test_hmm_full_route": full}
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(OUT / "hmm_config.json", "w") as f:       # what the API needs at inference time
        json.dump({"beta": beta, "emission_weight": w, "max_gap_s": args.max_gap,
                   "route_factor": 2.0, "route_slack_m": 300.0, "backtrack_tol_m": 15.0,
                   "model_tag": tag}, f, indent=2)
    plot_comparison(per_trip, OUT / "classifier_vs_hmm.png")

    c, h = summary["student_test_classifier_only"], summary["student_test_hmm"]
    print("\nSTUDENT TEST           classifier-only    classifier+HMM")
    for k in c:
        print(f"  {k:22s} {c[k]:>12.4f}    {h[k]:>12.4f}")
    print("\n  reconstructed connected route (HMM + shortest-path fill-in):")
    for k, v in full.items():
        print(f"  {k:22s} {v:>12.4f}" if isinstance(v, float) else f"  {k:22s} {v:>12}")
    print(f"\noutputs -> {OUT}")


def plot_comparison(pt, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(len(pt))
    wdt = 0.38
    for ax, (metric, title) in zip(axes, [("top1", "Top-1 accuracy"),
                                          ("continuity", "Route continuity"),
                                          ("route_f1", "Route F1 (length-weighted)")]):
        ax.bar(x - wdt / 2, pt[f"clf_{metric}"], wdt, label="classifier only")
        ax.bar(x + wdt / 2, pt[f"hmm_{metric}"], wdt, label="classifier + HMM")
        ax.set_xticks(x, pt["trip_id"], rotation=45, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    fig.suptitle("Student-test trips: classifier-only vs HMM/Viterbi decoding")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
