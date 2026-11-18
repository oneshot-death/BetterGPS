
from pathlib import Path
import sys

# Find the root of the BetterGPS repository.
ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse Person C's existing HMM implementation.
from hmm.viterbi import prepare_trip, viterbi, full_route


def decode_route(candidates, track, nodes, arcs, trip_id, config):
    """
    Decode candidate road arcs using the existing HMM/Viterbi implementation.
    """

    # Prepare the candidate lattice.
    lattice = prepare_trip(
        candidates,
        track,
        nodes,
        arcs,
        trip_id,
        max_gap_s=config["max_gap_s"],
        route_factor=config["route_factor"],
        route_slack_m=config["route_slack_m"],
        backtrack_tol_m=config["backtrack_tol_m"],
    )

    # Run Viterbi decoding.
    choices, segments = viterbi(
        lattice,
        beta=config["beta"],
        emission_weight=config["emission_weight"],
    )

    # Convert the selected candidate positions into actual road arc IDs.
    chosen_arcs = [
        int(lattice.cand_arcs[i][choice])
        for i, choice in enumerate(choices)
    ]

    # Reconstruct a connected route.
    reconstructed_route = full_route(
        lattice,
        chosen_arcs,
        segments,
    )

    return {
        "point_predictions": chosen_arcs,
        "predicted_arcs": reconstructed_route,
    }
