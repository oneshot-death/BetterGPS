# HMM / Viterbi route reconstruction

Turns the classifier's per-candidate probabilities into a coherent route on the road network.

## Model

| | Implementation |
|---|---|
| **Hidden states** | at each GPS point, its candidate arcs (from `pairs.csv`) |
| **Emission** | `w × log P_classifier(candidate is the true arc)` |
| **Transition** | `-|route_dist − straight_line_dist| / beta` (Newson & Krumm 2009). `route_dist` = shortest driving distance on the **directed** network from the GPS point's projection on arc *a* to the next point's projection on arc *b* |
| **Unreachable pairs** | no path within `2 × straight-line + 300 m` → probability 0 |
| **Same arc** | distance along the arc; moving backwards up to 15 m is treated as GPS noise |
| **Large GPS gaps** | time gap > `max_gap_s` (30 s) → the sequence is split and each part decoded separately |
| **Disconnected steps** | if no candidate at t+1 is reachable from any candidate at t → HMM break, new segment starts |
| **Decoding** | Viterbi in log space, per segment |
| **Full route** | shortest path inserted between consecutive decoded arcs → a connected arc sequence (breaks only at segment boundaries) |

Distances use a local equirectangular projection around each trip's mean coordinate (metres). Error is negligible at city scale.

## Run (from repo root)

Needs `artifacts/models/<tag>.joblib` and `results/modeling/predictions/<tag>_student_test.csv` from the modelling stage.

```bash
python hmm/run_hmm.py --model-tag xgboost_tuned
```

1. **Out-of-fold predictions** for the training trips (same GroupKFold as model CV), cached in `results/modeling/predictions/<tag>_train_oof.csv`
2. **Decoder grid search** (`beta` × emission weight) on 20 training trips, maximising Top-1. Student-test and evaluator trips are not used
3. **Student-test comparison**: classifier-only vs classifier+HMM

Options: `--tune-trips`, `--betas 2,5,10,20,50`, `--weights 0.5,1,2`, `--max-gap 30`.

## Outputs (`results/hmm/`)

| file | content |
|---|---|
| `decoder_grid_train.csv` | every decoder setting tried and its training-trip score |
| `student_test_per_trip.csv` | per trip: Top-1, continuity, route precision/recall/F1 for classifier-only, HMM and the full route; segments, gap splits, Viterbi time |
| `summary.json` | pooled student-test comparison |
| `student_test_decoded.csv` | per GPS point: classifier arc, HMM arc, segment |
| `student_test_full_routes.csv` | connected predicted route per trip (`-1` = break) |
| `classifier_vs_hmm.png` | per-trip comparison chart |
| `hmm_config.json` | the decoder parameters the API must use |

## Using it in the API

```python
from viterbi import load_trip, prepare_trip, viterbi, full_route
track, nodes, arcs = load_trip(data_root, partition, trip_id)
lat = prepare_trip(cands_df, track, nodes, arcs, trip_id, max_gap_s=cfg["max_gap_s"])
choice, seg = viterbi(lat, cfg["beta"], cfg["emission_weight"])
point_arcs = [lat.cand_arcs[t][k] for t, k in enumerate(choice)]
route = full_route(lat, point_arcs, seg)   # list of arc ids, None = break
```
`cands_df` = the trip's candidate rows with `point_idx`, `arc_id`, `proba` (no `label` needed).

## Metrics

- **Top-1**: share of GPS points whose decoded arc is the reference arc
- **Route continuity**: share of consecutive distinct arcs where one ends where the next starts
- **Route precision / recall / F1**: length-weighted overlap between the set of predicted arcs and the `.route` reference arcs
