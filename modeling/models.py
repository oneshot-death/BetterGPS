"""Model zoo: every candidate classifier the team compares.

Each spec records the algorithm family, default hyperparameters, whether features need
scaling, and an optional cap on training rows (for models that are slow on large data).
`build_model(name, **overrides)` is used both by the comparison and by Optuna tuning,
so the tuned and untuned models are built the same way.
"""
import importlib.util

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import SEED

HAS_LGBM = importlib.util.find_spec("lightgbm") is not None
HAS_XGB = importlib.util.find_spec("xgboost") is not None

MODEL_SPECS = {
    "logreg": dict(
        family="Linear", scale=True, max_train_rows=None,
        cls=LogisticRegression,
        defaults=dict(C=1.0, class_weight="balanced", max_iter=2000)),
    "gnb": dict(
        family="Probabilistic", scale=True, max_train_rows=None,
        cls=GaussianNB, defaults=dict(var_smoothing=1e-9)),
    "knn": dict(
        family="Instance-based", scale=True, max_train_rows=200_000,   # KNN prediction is slow
        cls=KNeighborsClassifier,
        defaults=dict(n_neighbors=25, weights="distance", n_jobs=-1)),
    "rf": dict(
        family="Ensemble (bagging)", scale=False, max_train_rows=None,
        cls=RandomForestClassifier,
        defaults=dict(n_estimators=200, min_samples_leaf=5, max_features="sqrt",
                      class_weight="balanced_subsample", n_jobs=-1)),
    "hgb": dict(
        family="Boosting", scale=False, max_train_rows=None,
        cls=HistGradientBoostingClassifier,
        defaults=dict(learning_rate=0.1, max_iter=500, max_leaf_nodes=31,
                      min_samples_leaf=20, l2_regularization=0.0, class_weight="balanced",
                      early_stopping=True, scoring="loss", validation_fraction=0.1,
                      n_iter_no_change=20)),
    "mlp": dict(
        family="Neural network", scale=True, max_train_rows=None,
        cls=MLPClassifier,
        defaults=dict(hidden_layer_sizes=(64, 32), alpha=1e-4, learning_rate_init=1e-3,
                      batch_size=256, max_iter=200, early_stopping=True,
                      validation_fraction=0.1, n_iter_no_change=10)),
}

if HAS_LGBM:
    from lightgbm import LGBMClassifier
    MODEL_SPECS["lightgbm"] = dict(
        family="Boosting", scale=False, max_train_rows=None, cls=LGBMClassifier,
        defaults=dict(n_estimators=500, learning_rate=0.05, num_leaves=31,
                      min_child_samples=20, class_weight="balanced", verbose=-1, n_jobs=-1))

if HAS_XGB:
    from xgboost import XGBClassifier
    MODEL_SPECS["xgboost"] = dict(
        family="Boosting", scale=False, max_train_rows=None, cls=XGBClassifier,
        defaults=dict(n_estimators=500, learning_rate=0.05, max_depth=6,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                      tree_method="hist", n_jobs=-1))

# Estimators that accept a random_state / seed argument
_SEEDED = {"logreg", "rf", "hgb", "mlp", "lightgbm", "xgboost"}


def build_model(name, seed=SEED, **overrides):
    spec = MODEL_SPECS[name]
    params = {**spec["defaults"], **overrides}
    if name in _SEEDED:
        params["random_state"] = seed
    est = spec["cls"](**params)
    return make_pipeline(StandardScaler(), est) if spec["scale"] else est


def final_estimator(model):
    """Unwrap a Pipeline to get the classifier itself."""
    return model.steps[-1][1] if hasattr(model, "steps") else model


def describe(model):
    """Compact one-line description of the non-default configuration (for logs/report)."""
    return " ".join(repr(final_estimator(model)).split())
