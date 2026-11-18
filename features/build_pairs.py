"""Build data/processed/pairs.csv: one row per (GPS point, candidate arc).

    python features/build_pairs.py --quick   # f_dist_m + f_rank only
    python features/build_pairs.py           # full feature set

Raw trip layout: data/raw/{train,student_test}/<trip>/<trip>.{track,arcs,nodes,route}
  .track  : lon, lat, timestamp(s)         one row per GPS point
  .arcs   : from_node, to_node             directed arc; arc_id = 0-based row
  .nodes  : lon, lat                       node id = 0-based row
  .route  : arc_ids of the true route, in travel order

Assumptions (documented, adjustable by flags):
  * Candidates: arcs within --radius metres of the point (nearest --max-cands kept; if none, the
    radius is widened so every point has >=1 candidate).
  * Reference arc per point: monotone alignment of the points to the route arcs
    (non-decreasing route position minimising total point-to-arc distance, Viterbi-style).
    label=1 only for that arc, and only if it is among the point's candidates.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARTS = ["train", "student_test"]
MAX_GAP_S = 5.0                            # neighbours further apart in time are not used          # evaluator trips are never read


def pair_dist(P, A, B):
    """Row-wise distance and projection fraction of points P to segments A->B (metres)."""
    AB = B - A
    t = np.clip(((P - A) * AB).sum(1) / np.maximum((AB ** 2).sum(1), 1e-9), 0, 1)
    return np.linalg.norm(P - (A + t[:, None] * AB), axis=1), t


def candidates(P, A, B, radius, max_cands, spacing=10.0):
    """Spatial candidate search. Returns (point_idx, arc_id, dist, proj_frac), sorted by distance per point."""
    from scipy.spatial import cKDTree
    L = np.linalg.norm(B - A, axis=1)
    ns = np.maximum(np.ceil(L / spacing).astype(int), 1) + 1          # samples per arc incl. both ends
    owner = np.repeat(np.arange(len(A)), ns)
    off = np.arange(len(owner)) - np.repeat(np.cumsum(ns) - ns, ns)
    frac = off / np.repeat(ns - 1, ns)
    S = A[owner] + frac[:, None] * (B - A)[owner]
    tree = cKDTree(S)
    out_p, out_a = [], []
    todo = np.arange(len(P)); r = radius
    found = np.zeros(len(P), bool)
    while len(todo) and r <= 2000:
        lists = tree.query_ball_point(P[todo], r + spacing / 2)
        for pi, lst in zip(todo, lists):
            if lst:
                arcs_ = np.unique(owner[lst]); out_p.append(np.full(len(arcs_), pi)); out_a.append(arcs_); found[pi] = True
        todo = np.flatnonzero(~found); r *= 2          # points with nothing nearby: widen until >=1 candidate
    pi = np.concatenate(out_p); ai = np.concatenate(out_a)
    d, t = pair_dist(P[pi], A[ai], B[ai])
    df = pd.DataFrame({"p": pi, "a": ai, "d": d, "t": t}).sort_values(["p", "d", "a"], kind="stable")
    df["rank"] = df.groupby("p").cumcount()
    df = df[(df["rank"] < max_cands) & ((df["d"] <= radius) | (df["rank"] == 0))]
    return df


def bearing(d):
    return np.degrees(np.arctan2(d[..., 0], d[..., 1])) % 360     # x=east, y=north


def angdiff(a, b):
    d = np.abs(a - b) % 360
    return np.minimum(d, 360 - d)


def align_route(D_route):
    """D_route (n_pts, n_route): distance to each route arc. Returns route position per point."""
    n, m = D_route.shape
    dp = np.empty((n, m)); bp = np.empty((n, m), int)
    dp[0] = D_route[0]
    for i in range(1, n):
        prev = dp[i - 1]
        cm = np.minimum.accumulate(prev)
        idx = np.arange(m)
        arg = np.maximum.accumulate(np.where(prev <= cm, idx, 0))   # argmin of prefix
        dp[i] = D_route[i] + cm
        bp[i] = arg
    pos = np.empty(n, int); pos[-1] = int(np.argmin(dp[-1]))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = bp[i, pos[i]]
    return pos


def build_trip(part, trip, quick, radius, max_cands):
    d = ROOT / "data" / "raw" / part / trip / trip
    arcs = np.loadtxt(f"{d}.arcs", dtype=int, ndmin=2)
    nodes = np.loadtxt(f"{d}.nodes", ndmin=2)
    route = np.loadtxt(f"{d}.route", dtype=int, ndmin=1)
    trk = np.loadtxt(f"{d}.track", ndmin=2)

    lat0 = np.deg2rad(nodes[:, 1].mean())
    k = np.array([111320 * np.cos(lat0), 110540.0])
    N = nodes * k
    P = trk[:, :2] * k
    A, B = N[arcs[:, 0]], N[arcs[:, 1]]

    # reference arc: monotone alignment of the points to the route arcs
    Dr = np.empty((len(P), len(route)), dtype=np.float32)
    for j, r in enumerate(route):
        Dr[:, j] = pair_dist(P, np.broadcast_to(A[r], P.shape), np.broadcast_to(B[r], P.shape))[0]
    ref = route[align_route(Dr)]
    n = len(P)

    c = candidates(P, A, B, radius, max_cands)
    rows_pt = c["p"].values; rows_arc = c["a"].values
    dist = c["d"].values; rank = c["rank"].values; proj = c["t"].values

    df = pd.DataFrame({
        "trip_id": trip, "partition": part,
        "point_idx": rows_pt, "arc_id": rows_arc,
        "label": (rows_arc == ref[rows_pt]).astype(int),
        "f_dist_m": dist, "f_rank": rank,
    })
    if quick:
        return df

    best_pp = pd.Series(dist).groupby(rows_pt).transform("min").values
    arc_vec = B - A
    arc_len = np.linalg.norm(arc_vec, axis=1)
    arc_brg = bearing(arc_vec)

    # GPS gaps (Person A EDA: timestamp diff > 1 s; some are hours long). Neighbour-based features
    # (heading, speed, neighbour distances) never straddle a gap longer than MAX_GAP_S: a side whose
    # neighbour is across a long gap falls back to the point itself (no interpolation).
    ts = trk[:, 2]
    idx = np.arange(n)
    gap_prev = np.r_[0.0, np.diff(ts)]
    gap_next = np.r_[np.diff(ts), 0.0]
    prv = np.where((idx > 0) & (gap_prev <= MAX_GAP_S), idx - 1, idx)
    nxt = np.where((idx < n - 1) & (gap_next <= MAX_GAP_S), idx + 1, idx)
    mv = P[nxt] - P[prv]
    step = np.linalg.norm(mv, axis=1)
    hdg = bearing(mv)
    dt = np.maximum(ts[nxt] - ts[prv], 1e-9)
    speed = np.where(nxt != prv, step / dt, 0.0)
    moving = step > 2.0                                       # heading unreliable when ~stationary

    df["f_gap_prev_s"] = np.minimum(gap_prev, 300.0)[rows_pt]
    df["f_gap_next_s"] = np.minimum(gap_next, 300.0)[rows_pt]
    df["f_near_long_gap"] = ((gap_prev > MAX_GAP_S) | (gap_next > MAX_GAP_S))[rows_pt].astype(int)
    df["f_dist_ratio"] = dist / (best_pp + 1.0)
    df["f_dist_gap_m"] = dist - best_pp
    df["f_bearing_diff"] = np.where(moving[rows_pt], angdiff(hdg[rows_pt], arc_brg[rows_arc]), 90.0)
    df["f_arc_len_m"] = arc_len[rows_arc]
    df["f_speed_mps"] = speed[rows_pt]
    df["f_proj_frac"] = proj
    df["f_n_cands"] = pd.Series(rows_pt).map(pd.Series(rows_pt).value_counts()).values
    df["f_dist_prev_m"] = pair_dist(P[prv[rows_pt]], A[rows_arc], B[rows_arc])[0]   # neighbours' distance to same arc
    df["f_dist_next_m"] = pair_dist(P[nxt[rows_pt]], A[rows_arc], B[rows_arc])[0]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--radius", type=float, default=50.0)
    ap.add_argument("--max-cands", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "pairs.csv"))
    a = ap.parse_args()

    parts = []
    for part in PARTS:
        for tdir in sorted((ROOT / "data" / "raw" / part).iterdir()):
            if tdir.is_dir():
                parts.append(build_trip(part, tdir.name, a.quick, a.radius, a.max_cands))
    df = pd.concat(parts, ignore_index=True)
    df["trip_id"] = df["trip_id"].astype(str)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False, float_format="%.2f")

    g = df.groupby(["trip_id", "point_idx"])["label"].sum()
    pts = len(g)
    print(f"{len(df):,} rows, {pts:,} points, {df.trip_id.nunique()} trips -> {a.out}")
    print(f"max labels per point: {g.max()}, candidate miss rate: {(g == 0).mean():.4f}")
    print(df.groupby('partition').trip_id.nunique().to_dict())


if __name__ == "__main__":
    main()
