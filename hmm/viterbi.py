"""HMM map matching with Viterbi decoding (after Newson & Krumm, 2009).

Hidden states  : at each GPS observation, its candidate arcs (from the pairs table).
Emissions      : the classifier's probability that the candidate is the true arc,
                 log P(z_t | x_t) = w * log(p_classifier).
Transitions    : a vehicle normally drives a road distance close to the straight-line
                 distance between consecutive GPS fixes, so
                 log P(x_t -> x_t+1) = -| route_dist - great_circle_dist | / beta,
                 where route_dist is the shortest driving distance on the directed road
                 network from the projection of z_t on x_t to the projection of z_t+1 on x_t+1.
                 Unreachable pairs (no path within the search limit) get probability 0.
Gaps / breaks  : the sequence is split into independent segments when (a) the time gap
                 between fixes exceeds `max_gap_s`, or (b) no candidate at t+1 is reachable
                 from any candidate at t (an "HMM break"). Each segment is decoded separately.

The expensive part (shortest paths) does not depend on (w, beta), so `prepare_trip` runs it
once and `viterbi` can then be re-run cheaply for many decoder parameter settings.
"""
import heapq
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EARTH_R = 6_371_000.0


# ----------------------------------------------------------------------------- geometry
def load_trip(data_root, partition, trip_id):
    base = Path(data_root) / partition / trip_id / trip_id
    track = np.loadtxt(f"{base}.track", ndmin=2)            # lon, lat, t
    nodes = np.loadtxt(f"{base}.nodes", ndmin=2)            # lon, lat
    arcs = np.loadtxt(f"{base}.arcs", dtype=np.int64, ndmin=2)  # from, to
    return track, nodes, arcs


def to_xy(lon, lat, lon0, lat0):
    """Local equirectangular projection to metres. Error is negligible at city scale
    (a trip spans a few km), and it lets us use plain vector geometry."""
    x = np.radians(lon - lon0) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_R
    return np.column_stack([x, y])


class RoadGraph:
    """Directed road network with arc lengths and a cached, distance-bounded Dijkstra."""

    def __init__(self, nodes_xy, arcs):
        self.xy = nodes_xy
        self.arcs = arcs
        self.arc_len = np.linalg.norm(nodes_xy[arcs[:, 1]] - nodes_xy[arcs[:, 0]], axis=1)
        self.out = [[] for _ in range(len(nodes_xy))]
        for arc_id, ((u, v), length) in enumerate(zip(arcs, self.arc_len)):
            self.out[u].append((v, length, arc_id))
        self._cache = {}                     # source node -> (limit, {node: dist}, {node: arc in})

    def dist_from(self, src, limit):
        """Shortest road distances from `src` to every node within `limit` metres."""
        cached = self._cache.get(src)
        if cached is not None and cached[0] >= limit:
            return cached[1]
        limit = max(limit, 2 * cached[0]) if cached else limit   # grow geometrically
        dist = {src: 0.0}
        prev = {}                            # node -> arc used to reach it (for path recovery)
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf):
                continue
            for v, w, arc_id in self.out[u]:
                nd = d + w
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    prev[v] = arc_id
                    heapq.heappush(heap, (nd, v))
        self._cache[src] = (limit, dist, prev)
        return dist

    def path(self, src, dst, limit=3000.0):
        """Arcs of the shortest path src -> dst (empty if src == dst, None if unreachable)."""
        if src == dst:
            return []
        dist = self.dist_from(src, self._cache[src][0] if src in self._cache else limit)
        if dst not in dist:
            dist = self.dist_from(src, limit)
            if dst not in dist:
                return None
        prev = self._cache[src][2]
        out, node = [], dst
        while node != src:
            arc_id = prev[node]
            out.append(arc_id)
            node = int(self.arcs[arc_id, 0])
        return out[::-1]

    def project(self, arc_ids, pt):
        """Fraction (0..1) along each arc of the closest point to `pt`."""
        a = self.xy[self.arcs[arc_ids, 0]]
        b = self.xy[self.arcs[arc_ids, 1]]
        ab = b - a
        denom = np.maximum((ab ** 2).sum(1), 1e-9)
        return np.clip(((pt - a) * ab).sum(1) / denom, 0.0, 1.0)


# ----------------------------------------------------------------------------- preparation
@dataclass
class TripLattice:
    trip_id: str
    point_idx: np.ndarray                 # GPS point index for each step
    cand_arcs: list                       # list of int arrays, candidates per step
    cand_logp: list                       # list of arrays, log classifier probability
    route_dist: list                      # list (len T-1) of (k_t x k_t+1) arrays, inf = unreachable
    gc_dist: np.ndarray                   # (T-1,) straight-line distance between fixes
    gap_break: np.ndarray                 # (T-1,) bool, True = time gap too large
    labels: list = field(default_factory=list)   # per step, label of each candidate (if known)
    arcs: np.ndarray = None               # (n_arcs, 2) from/to node, for continuity checks
    arc_len: np.ndarray = None            # (n_arcs,) metres
    graph: "RoadGraph" = None             # kept to fill in the full connected route


def prepare_trip(cands, track, nodes, arcs, trip_id, max_gap_s=30.0,
                 route_factor=2.0, route_slack_m=300.0, backtrack_tol_m=15.0):
    """Build the candidate lattice and all transition road distances for one trip.

    cands: dataframe rows of ONE trip with point_idx, arc_id, proba (and optionally label).
    """
    lon0, lat0 = track[:, 0].mean(), track[:, 1].mean()
    g = RoadGraph(to_xy(nodes[:, 0], nodes[:, 1], lon0, lat0), arcs)
    gps = to_xy(track[:, 0], track[:, 1], lon0, lat0)
    times = track[:, 2]

    cands = cands.sort_values(["point_idx", "arc_id"])
    groups = list(cands.groupby("point_idx", sort=True))
    pidx = np.array([p for p, _ in groups])
    cand_arcs = [grp["arc_id"].to_numpy(np.int64) for _, grp in groups]
    cand_logp = [np.log(np.clip(grp["proba"].to_numpy(float), 1e-6, 1.0)) for _, grp in groups]
    labels = [grp["label"].to_numpy(int) for _, grp in groups] if "label" in cands else []
    frac = [g.project(a, gps[p]) for a, p in zip(cand_arcs, pidx)]

    T = len(pidx)
    gc = np.linalg.norm(gps[pidx[1:]] - gps[pidx[:-1]], axis=1) if T > 1 else np.zeros(0)
    gap = (times[pidx[1:]] - times[pidx[:-1]]) > max_gap_s if T > 1 else np.zeros(0, bool)

    route = []
    for t in range(T - 1):
        A, B = cand_arcs[t], cand_arcs[t + 1]
        R = np.full((len(A), len(B)), np.inf)
        if not gap[t]:
            limit = route_factor * gc[t] + route_slack_m
            for i, a in enumerate(A):
                la, fa = g.arc_len[a], frac[t][i]
                rest_a = (1.0 - fa) * la                       # distance to the end of arc a
                reach = None
                for j, b in enumerate(B):
                    fb = frac[t + 1][j]
                    if a == b:                                 # stayed on the same arc
                        d = (fb - fa) * la
                        if d >= -backtrack_tol_m:              # small backwards = GPS noise
                            R[i, j] = abs(d)
                            continue
                    if reach is None:
                        reach = g.dist_from(int(arcs[a, 1]), limit)
                    sp = reach.get(int(arcs[b, 0]))
                    if sp is not None:
                        R[i, j] = rest_a + sp + fb * g.arc_len[b]
        route.append(R)

    return TripLattice(trip_id, pidx, cand_arcs, cand_logp, route, gc, gap, labels, arcs, g.arc_len, g)


# ----------------------------------------------------------------------------- decoding
def viterbi(lat: TripLattice, beta=10.0, emission_weight=1.0):
    """Decode the most likely candidate index per step.

    Returns (choice, segment_id): choice[t] = index into lat.cand_arcs[t];
    segment_id[t] increments at every time-gap split or HMM break.
    """
    T = len(lat.point_idx)
    choice = np.zeros(T, dtype=int)
    seg = np.zeros(T, dtype=int)
    if T == 0:
        return choice, seg

    def backtrack(t_end, back, delta_end, t_start):
        k = int(np.argmax(delta_end))
        for t in range(t_end, t_start - 1, -1):
            choice[t] = k
            if t > t_start:
                k = back[t][k]

    start = 0
    seg_no = 0
    delta = emission_weight * lat.cand_logp[0]
    back = [None] * T
    for t in range(1, T):
        trans = -np.abs(lat.route_dist[t - 1] - lat.gc_dist[t - 1]) / beta   # -inf if unreachable
        scores = delta[:, None] + trans
        best_prev = scores.max(axis=0)
        if lat.gap_break[t - 1] or not np.isfinite(best_prev).any():
            backtrack(t - 1, back, delta, start)          # close the current segment
            seg_no += 1
            start = t
            delta = emission_weight * lat.cand_logp[t]
        else:
            back[t] = scores.argmax(axis=0)
            delta = best_prev + emission_weight * lat.cand_logp[t]
            # candidates unreachable from every previous state keep -inf and cannot be chosen
        seg[t] = seg_no
    backtrack(T - 1, back, delta, start)
    return choice, seg


def full_route(lat: TripLattice, arc_seq, segment):
    """Turn per-point arcs into a connected arc sequence by inserting the shortest path
    between consecutive distinct arcs of the same segment. Returns a list of arc ids
    (None marks a break between segments or an unreachable pair)."""
    out = []
    prev_arc, prev_seg = None, None
    for a, s in zip(arc_seq, segment):
        a = int(a)
        if a == prev_arc:
            continue
        if prev_arc is not None:
            if s != prev_seg:
                out.append(None)
            else:
                link = lat.graph.path(int(lat.arcs[prev_arc, 1]), int(lat.arcs[a, 0]))
                if link is None:
                    out.append(None)
                else:
                    out.extend(link)
        out.append(a)
        prev_arc, prev_seg = a, s
    return out


def classifier_only(lat: TripLattice):
    return np.array([int(np.argmax(lp)) for lp in lat.cand_logp])
