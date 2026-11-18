"""Shared paths, constants and data loading for the modelling stage (Person C).

The modelling code expects ONE table of (GPS point, candidate arc) pairs produced by
the feature-engineering stage (Person B). Agreed schema:

    trip_id    : trip folder name, e.g. "00000048"  (used for GroupKFold)
    partition  : "train" or "student_test"
    point_idx  : 0-based index of the GPS point inside the trip's .track file
    arc_id     : candidate arc id (0-based row index in that trip's .arcs file)
    label      : 1 if arc_id is the reference arc for this point, else 0
    f_*        : feature columns - every column starting with "f_" is used as a feature

Points whose reference arc is NOT among the candidates should still appear (all labels 0):
they are needed to compute the candidate miss rate.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]            # repo root
DEFAULT_PAIRS = ROOT / "data" / "processed" / "pairs.csv"
DEFAULT_DATA_ROOT = ROOT / "data" / "raw"             # contains train/ and student_test/ trip folders
RESULTS_DIR = ROOT / "results" / "modeling"
MODEL_DIR = ROOT / "artifacts" / "models"

KEY_COLS = ["trip_id", "partition", "point_idx", "arc_id", "label"]
ALLOWED_PARTITIONS = {"train", "student_test"}
SEED = 42


def load_pairs(path):
    """Load the pairs table, validate it, and return (dataframe, feature_column_list)."""
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, dtype={"trip_id": str})
    df["trip_id"] = df["trip_id"].astype(str)

    missing = [c for c in KEY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"pairs file is missing required columns: {missing}")

    bad = set(df["partition"].unique()) - ALLOWED_PARTITIONS
    if bad:
        # Leakage guard: evaluator-testing trips must never reach model development.
        raise ValueError(f"unexpected partitions {bad}; only {ALLOWED_PARTITIONS} are allowed here")

    feats = [c for c in df.columns if c.startswith("f_")]
    if not feats:
        raise ValueError("no feature columns found (feature columns must start with 'f_')")

    df = df.sort_values(["partition", "trip_id", "point_idx", "arc_id"]).reset_index(drop=True)
    return df, feats


def split_partitions(df):
    train = df[df["partition"] == "train"].reset_index(drop=True)
    test = df[df["partition"] == "student_test"].reset_index(drop=True)
    overlap = set(train["trip_id"]) & set(test["trip_id"])
    if overlap:
        raise ValueError(f"trips appear in both train and student_test: {sorted(overlap)[:5]}")
    return train, test
