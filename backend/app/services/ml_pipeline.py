
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Locate the BetterGPS repository root.
ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse Person B's existing geometry and candidate-generation functions.
from features.build_pairs import (
    pair_dist,
    candidates,
    bearing,
    angdiff,
)

MAX_GAP_S = 5.0


def build_inference_features(track, nodes, arcs):
    """
    Generate candidate road arcs and the same 14 features
    used during model training.

    This function does NOT read a .route file or generate labels.
    """

    track = np.asarray(track, dtype=float)
    nodes = np.asarray(nodes, dtype=float)
    arcs = np.asarray(arcs, dtype=int)

    if track.ndim != 2 or track.shape[1] != 3:
        raise ValueError("Track must contain longitude, latitude and timestamp.")

    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("Nodes must contain longitude and latitude.")

    if arcs.ndim != 2 or arcs.shape[1] != 2:
        raise ValueError("Arcs must contain from_node and to_node.")

    if len(track) == 0 or len(nodes) == 0 or len(arcs) == 0:
        raise ValueError("Track, nodes and arcs cannot be empty.")

    if not np.isfinite(track).all() or not np.isfinite(nodes).all():
        raise ValueError("Coordinates and timestamps must be finite.")

    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError("An arc references a node that does not exist.")

    # Use the same local coordinate conversion as Person B.
    lat0 = np.deg2rad(nodes[:, 1].mean())
    k = np.array([111320 * np.cos(lat0), 110540.0])

    N = nodes * k
    P = track[:, :2] * k

    A = N[arcs[:, 0]]
    B = N[arcs[:, 1]]

    # Generate candidate arcs without reading reference routes.
    c = candidates(P, A, B, radius=50.0, max_cands=10)

    if c.empty:
        raise ValueError("No candidate road arcs were generated.")

    rows_pt = c["p"].to_numpy()
    rows_arc = c["a"].to_numpy()
    dist = c["d"].to_numpy()
    rank = c["rank"].to_numpy()
    proj = c["t"].to_numpy()

    # Calculate the same features used during training.
    df = pd.DataFrame({
        "point_idx": rows_pt,
        "arc_id": rows_arc,
        "f_dist_m": dist,
        "f_rank": rank,
    })

    best_pp = (
        pd.Series(dist)
        .groupby(rows_pt)
        .transform("min")
        .to_numpy()
    )

    arc_vec = B - A
    arc_len = np.linalg.norm(arc_vec, axis=1)
    arc_brg = bearing(arc_vec)

    ts = track[:, 2]
    n = len(P)
    idx = np.arange(n)

    gap_prev = np.r_[0.0, np.diff(ts)]
    gap_next = np.r_[np.diff(ts), 0.0]

    prv = np.where(
        (idx > 0) & (gap_prev <= MAX_GAP_S),
        idx - 1,
        idx,
    )

    nxt = np.where(
        (idx < n - 1) & (gap_next <= MAX_GAP_S),
        idx + 1,
        idx,
    )

    mv = P[nxt] - P[prv]
    step = np.linalg.norm(mv, axis=1)
    hdg = bearing(mv)

    dt = np.maximum(ts[nxt] - ts[prv], 1e-9)
    speed = np.where(nxt != prv, step / dt, 0.0)
    moving = step > 2.0

    df["f_gap_prev_s"] = np.minimum(gap_prev, 300.0)[rows_pt]
    df["f_gap_next_s"] = np.minimum(gap_next, 300.0)[rows_pt]

    df["f_near_long_gap"] = (
        (gap_prev > MAX_GAP_S) |
        (gap_next > MAX_GAP_S)
    )[rows_pt].astype(int)

    df["f_dist_ratio"] = dist / (best_pp + 1.0)
    df["f_dist_gap_m"] = dist - best_pp

    df["f_bearing_diff"] = np.where(
        moving[rows_pt],
        angdiff(hdg[rows_pt], arc_brg[rows_arc]),
        90.0,
    )

    df["f_arc_len_m"] = arc_len[rows_arc]
    df["f_speed_mps"] = speed[rows_pt]
    df["f_proj_frac"] = proj

    df["f_n_cands"] = (
        pd.Series(rows_pt)
        .map(pd.Series(rows_pt).value_counts())
        .to_numpy()
    )

    df["f_dist_prev_m"] = pair_dist(
        P[prv[rows_pt]],
        A[rows_arc],
        B[rows_arc],
    )[0]

    df["f_dist_next_m"] = pair_dist(
        P[nxt[rows_pt]],
        A[rows_arc],
        B[rows_arc],
    )[0]

    return df
