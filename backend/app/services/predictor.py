
import numpy as np
import pandas as pd

from app.services.ml_pipeline import build_inference_features
from app.services.model_loader import load_model, load_hmm_config
from app.services.decoder import decode_route


def predict_route(track, nodes, arcs, trip_id):
    """
    Run the complete BetterGPS inference pipeline:
    candidate generation -> feature extraction ->
    XGBoost prediction -> HMM route decoding.
    """

    # Convert input data into NumPy arrays.
    track = np.asarray(track, dtype=float)
    nodes = np.asarray(nodes, dtype=float)
    arcs = np.asarray(arcs, dtype=int)

    # Step 1: Generate candidate road arcs and features.
    candidates = build_inference_features(
        track,
        nodes,
        arcs,
    )

    # Step 2: Load the trained model.
    artifact = load_model()
    model = artifact["model"]
    feature_names = artifact["features"]

    # Step 3: Prepare the feature matrix in training order.
    X = candidates[feature_names]

    # Step 4: Predict candidate probabilities.
    probabilities = model.predict_proba(X)[:, 1]
    candidates["proba"] = probabilities

    # Step 5: Load the HMM configuration.
    config = load_hmm_config()

    # Step 6: Decode the route using HMM/Viterbi.
    result = decode_route(
        candidates,
        track,
        nodes,
        arcs,
        trip_id,
        config,
    )

    return {
        "trip_id": trip_id,
        "point_predictions": result["point_predictions"],
        "predicted_arcs": result["predicted_arcs"],
        "prediction_method": "XGBoost + HMM/Viterbi",
    }
