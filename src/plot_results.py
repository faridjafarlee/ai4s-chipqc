"""Render the measured model comparison and selected-model diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main() -> None:
    document = json.loads((REPORTS / "evaluation.json").read_text())
    predictions = pd.read_csv(REPORTS / "grouped_oof_predictions.csv")
    selected = document["selected_model"]
    names = list(document["date_grouped"])
    labels = [name.replace("_", " ").title() for name in names]
    random_auc = [document["image_random"][name]["pooled"]["roc_auc"] for name in names]
    grouped_auc = [document["date_grouped"][name]["pooled"]["roc_auc"] for name in names]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    location = np.arange(len(names))
    axis.bar(location - .18, random_auc, .36, label="Image-random 5-fold", color="#a4c9b5")
    axis.bar(location + .18, grouped_auc, .36, label="Date-held-out 5-fold", color="#176a4a")
    axis.set_xticks(location, labels, rotation=18, ha="right")
    axis.set_ylabel("Pooled out-of-fold ROC AUC")
    axis.set_ylim(0, 1)
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(REPORTS / "validation_comparison.png", dpi=180)
    plt.close(figure)

    y = predictions.good.to_numpy()
    p = predictions[selected + "_prob_good"].to_numpy()
    false_positive, true_positive, _ = roc_curve(y, p)
    fraction_positive, mean_predicted = calibration_curve(y, p, n_bins=10)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    axes[0].plot(false_positive, true_positive, color="#176a4a", linewidth=2,
                 label=f"AUC {grouped_auc[names.index(selected)]:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#9baaa1")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate",
                title="Date-held-out ROC")
    axes[0].legend(frameon=False)
    axes[1].plot(mean_predicted, fraction_positive, "o-", color="#176a4a", linewidth=2)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="#9baaa1")
    axes[1].set(xlabel="Mean predicted good probability", ylabel="Observed good fraction",
                title="Date-held-out calibration")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(REPORTS / "selected_model_diagnostics.png", dpi=180)
    plt.close(figure)
    print("Saved validation_comparison.png and selected_model_diagnostics.png")


if __name__ == "__main__":
    main()
