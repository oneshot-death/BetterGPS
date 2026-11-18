"""Train and compare all candidate models.

Usage (from the repo root):
    python modeling/run_experiments.py                         # all models, data/processed/pairs.csv
    python modeling/run_experiments.py --models logreg,rf,hgb  # a subset
    python modeling/run_experiments.py --pairs data/processed/pairs_fake.csv --data-root none

Outputs go to results/modeling/ (results.csv, cv_folds/, predictions/, plots/)
and trained models to artifacts/models/.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DEFAULT_DATA_ROOT, DEFAULT_PAIRS, SEED, load_pairs, split_partitions  # noqa: E402
from experiment import fit_and_evaluate  # noqa: E402
from models import MODEL_SPECS, build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                    help="folder with train/ and student_test/ raw trips (for route continuity); 'none' to skip")
    ap.add_argument("--models", default=",".join(MODEL_SPECS),
                    help=f"comma-separated subset of: {', '.join(MODEL_SPECS)}")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    df, feats = load_pairs(args.pairs)
    train, test = split_partitions(df)
    data_root = None if args.data_root.lower() == "none" else args.data_root
    print(f"features ({len(feats)}): {feats}")
    print(f"train: {train.trip_id.nunique()} trips, {len(train):,} rows | "
          f"student_test: {test.trip_id.nunique()} trips, {len(test):,} rows")

    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        if name not in MODEL_SPECS:
            print(f"skipping unknown model '{name}'")
            continue
        model = build_model(name, seed=args.seed)
        fit_and_evaluate(name, model, train, test, feats, data_root,
                         n_folds=args.folds, max_rows=MODEL_SPECS[name]["max_train_rows"],
                         seed=args.seed)

    print("\nDone. Run `python modeling/plot_results.py` for comparison charts.")


if __name__ == "__main__":
    main()
