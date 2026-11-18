
from functools import lru_cache
import json
from pathlib import Path

import joblib


# Locate the BetterGPS repository root.
ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = ROOT / "artifacts" / "models" / "xgboost.joblib"
HMM_CONFIG_PATH = ROOT / "results" / "hmm" / "hmm_config.json"


@lru_cache(maxsize=1)
def load_model():
    """
    Load the existing trained XGBoost model.
    The cached result is reused across prediction requests.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    artifact = joblib.load(MODEL_PATH)

    return artifact


@lru_cache(maxsize=1)
def load_hmm_config():
    """
    Load the decoder parameters selected by the ML team.
    """

    if not HMM_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"HMM configuration not found: {HMM_CONFIG_PATH}"
        )

    with open(HMM_CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)
