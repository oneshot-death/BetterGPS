"""Generate a FAKE pairs table in the agreed schema, ONLY to test the modelling code
before the real feature table from the feature-engineering stage is ready.

Never report results from this file.

    python modeling/make_fake_pairs.py   ->  data/processed/pairs_fake.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import ROOT  # noqa: E402

rng = np.random.default_rng(0)
rows = []
for partition, n_trips in [("train", 20), ("student_test", 4)]:
    for t in range(n_trips):
        trip = f"fake{partition[:2]}{t:04d}"
        for p in range(int(rng.integers(150, 300))):
            k = int(rng.integers(3, 9))                         # candidates for this point
            dist = np.sort(np.abs(rng.normal(0, 25, k)))        # metres to each candidate
            true = int(rng.choice(k, p=np.r_[0.7, np.full(k - 1, 0.3 / (k - 1))]))
            if rng.random() < 0.05:
                true = -1                                       # reference arc not among candidates
            for c in range(k):
                is_true = int(c == true)
                rows.append({
                    "trip_id": trip, "partition": partition, "point_idx": p,
                    "arc_id": int(rng.integers(0, 5000)), "label": is_true,
                    "f_dist_m": dist[c],
                    "f_rank": c,
                    "f_dist_ratio": dist[c] / (dist[0] + 1),
                    "f_bearing_diff": abs(rng.normal(10 if is_true else 60, 30)) % 180,
                    "f_arc_len_m": rng.gamma(2, 50),
                    "f_speed_mps": abs(rng.normal(10, 4)),
                })

out = ROOT / "data" / "processed" / "pairs_fake.csv"
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(f"wrote {len(rows):,} fake rows -> {out}")
