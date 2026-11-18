"""Comparison charts and summary table from results/modeling/results.csv.

Uses the most recent run of each model. Usage (from repo root):
    python modeling/plot_results.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import RESULTS_DIR  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def bar(ax, labels, values, err=None, title="", ylabel="", fmt="{:.3f}"):
    x = np.arange(len(labels))
    bars = ax.bar(x, values, yerr=err, capsize=3, color="tab:blue")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    for b, v in zip(bars, values):
        if np.isfinite(v):
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=7)


def main():
    res = pd.read_csv(RESULTS_DIR / "results.csv")
    latest = res.sort_values("run_time").groupby("model").tail(1).sort_values("cv_macro_f1_mean")
    out = RESULTS_DIR / "plots"
    out.mkdir(parents=True, exist_ok=True)
    m = latest["model"].tolist()

    fig, ax = plt.subplots(figsize=(7, 4))
    bar(ax, m, latest["cv_macro_f1_mean"], latest["cv_macro_f1_std"],
        "Cross-validation Macro F1 (GroupKFold by trip, mean ± std)", "Macro F1")
    fig.tight_layout(); fig.savefig(out / "cv_macro_f1.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(m)); w = 0.38
    ax.bar(x - w / 2, latest["test_top1_all"], w, label="Top-1 accuracy")
    ax.bar(x + w / 2, latest["test_macro_f1"], w, label="Macro F1")
    ax.set_xticks(x, m, rotation=30, ha="right")
    ax.set_title("Student-test performance"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "student_test_top1_f1.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    bar(ax, m, latest["test_log_loss"], title="Student-test log loss (lower is better)",
        ylabel="log loss")
    fig.tight_layout(); fig.savefig(out / "student_test_log_loss.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    bar(ax, m, latest["infer_ms_per_1k_rows"], title="Inference time", ylabel="ms per 1,000 candidate rows",
        fmt="{:.1f}")
    fig.tight_layout(); fig.savefig(out / "inference_time.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, latest["train_top1"], w, label="train Top-1")
    ax.bar(x + w / 2, latest["test_top1_all"], w, label="student-test Top-1")
    ax.set_xticks(x, m, rotation=30, ha="right")
    ax.set_title("Train vs student-test Top-1 (overfitting check)"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "overfitting_check.png", dpi=150); plt.close(fig)

    cols = {"model": "Model", "family": "Family", "config": "Key configuration",
            "cv_macro_f1_mean": "CV Macro F1", "test_top1_all": "Top-1 Acc",
            "test_route_continuity": "Route continuity", "test_macro_f1": "Student-test Macro F1",
            "test_log_loss": "Log loss", "infer_ms_per_1k_rows": "Inference ms/1k rows",
            "train_time_s": "Train time s", "test_candidate_miss_rate": "Candidate miss rate"}
    table = latest.sort_values("cv_macro_f1_mean", ascending=False)[list(cols)].rename(columns=cols)
    table.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)
    with pd.option_context("display.max_colwidth", 40, "display.width", 200):
        print(table.round(4).to_string(index=False))
    print(f"\nplots -> {out}\ntable -> {RESULTS_DIR / 'comparison_table.csv'}")


if __name__ == "__main__":
    main()
