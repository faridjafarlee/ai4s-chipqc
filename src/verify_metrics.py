"""Verify published ChipQC predictions without images, dependencies or training.

Run ``python3 -m src.verify_metrics --output reports/metrics-verification.json``.
The date-bootstrap intervals remain recorded training outputs, not recomputed
intervals. Queue and score-band analyses are retrospective, descriptive checks.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "grouped_oof_predictions.csv": "203bc5fa780ec1c63aae3fdf74c51ee69412b18ee1429f2801644719c5090857",
    "evaluation.json": "e198e572d53d9519a42c9d140f9591be157b75bdc5ec928887047e86980c9020",
}
TOLERANCE = 1e-12


def auc(labels, scores):
    negatives = sorted(score for label, score in zip(labels, scores) if label == 0)
    positives = [score for label, score in zip(labels, scores) if label == 1]
    if not negatives or not positives:
        raise ValueError("ROC AUC needs both classes")
    wins = sum(bisect.bisect_left(negatives, score) +
               0.5 * (bisect.bisect_right(negatives, score) -
                      bisect.bisect_left(negatives, score)) for score in positives)
    return wins / (len(positives) * len(negatives))


def metrics(labels, scores):
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        raise ValueError("Metrics need both classes")
    confusion = [[0, 0], [0, 0]]
    for label, score in zip(labels, scores):
        confusion[label][int(score >= 0.5)] += 1
    # Average precision: add each tied threshold's recall increment times its
    # precision. This preserves ties rather than imposing an image-ID ordering.
    ranked = sorted(zip(scores, labels), reverse=True)
    true_positive = 0
    ap = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        added = sum(label for _, label in ranked[index:end])
        true_positive += added
        ap += (added / positive_count) * (true_positive / end)
        index = end
    return {
        "n": len(labels), "good_share": positive_count / len(labels),
        "roc_auc": auc(labels, scores), "average_precision": ap,
        "balanced_accuracy": 0.5 * (confusion[0][0] / negative_count +
                                     confusion[1][1] / positive_count),
        "brier": sum((score - label) ** 2 for score, label in zip(scores, labels)) / len(labels),
        "confusion_bad_good": confusion,
    }


def require_same(actual, expected, context):
    for key, value in actual.items():
        if isinstance(value, list):
            if value != expected[key]:
                raise ValueError(f"{context}: {key} differs")
        elif not math.isclose(value, expected[key], rel_tol=0, abs_tol=TOLERANCE):
            raise ValueError(f"{context}: {key} differs: {value} versus {expected[key]}")


def review_queues(rows, column, fraction=0.3):
    dates = collections.defaultdict(list)
    for row in rows:
        dates[row["acquisition_date"]].append(row)
    budget = math.ceil(fraction * len(rows))
    allocation = {date: math.floor(fraction * len(items)) for date, items in dates.items()}
    remainder_order = sorted(dates, key=lambda date: (
        -(fraction * len(dates[date]) - allocation[date]), date))
    for date in remainder_order[:budget - sum(allocation.values())]:
        allocation[date] += 1
    # Scores and IDs alone determine both queues; labels enter only the metrics.
    key = lambda row: (float(row[column]), row["imageID"])
    pooled = sorted(rows, key=key)[:budget]
    within_date = [row for date in sorted(dates)
                   for row in sorted(dates[date], key=key)[:allocation[date]]]
    if len(within_date) != budget:
        raise ValueError("Per-date quotas do not match the global budget")
    bad_count = sum(int(row["good"]) == 0 for row in rows)

    def result(reviewed):
        found = sum(int(row["good"]) == 0 for row in reviewed)
        return {"reviewed": len(reviewed), "bad_found": found,
                "bad_recall": found / bad_count, "bad_precision": found / len(reviewed)}

    expected_random = sum(allocation[date] *
                          sum(int(row["good"]) == 0 for row in items) / len(items)
                          for date, items in dates.items())
    return {
        "fraction_requested": fraction, "budget_rule": "ceil(fraction * all_frames)",
        "within_date_quota_rule": "floor per date, largest fractional remainders, date-ID ties",
        "ranking_rule": "ascending P(good), then imageID; no labels used in selection",
        "total_bad_frames": bad_count, "pooled_across_dates": result(pooled),
        "within_each_date": result(within_date),
        "random_same_date_quotas": {"expected_bad_found": expected_random,
                                   "expected_bad_recall": expected_random / bad_count},
        "prospective_workflow_benefit_measured": False,
    }


def verify():
    started = time.perf_counter()
    for name, expected in INPUTS.items():
        actual = hashlib.sha256((ROOT / "reports" / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen input hash differs: {name}")
    with (ROOT / "reports/grouped_oof_predictions.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    original = json.loads((ROOT / "reports/evaluation.json").read_text())
    if len({row["imageID"] for row in rows}) != len(rows):
        raise ValueError("Duplicate image IDs")
    labels = [int(row["good"]) for row in rows]
    if not set(labels) <= {0, 1}:
        raise ValueError("Invalid binary label")
    if any(row["acquisition_date"] != row["imageID"].split("_")[0] for row in rows):
        raise ValueError("Date proxy does not match image ID")

    def probabilities(column):
        values = [float(row[column]) for row in rows]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError(f"Invalid probability in {column}")
        return values

    grouped = {}
    for model in original["date_grouped"]:
        grouped[model] = metrics(labels, probabilities(model + "_prob_good"))
        require_same(grouped[model], original["date_grouped"][model]["pooled"], model)
    selected = original["selected_model"]
    scores = probabilities(selected + "_prob_good")
    random_metrics = metrics(labels, probabilities(selected + "_random_oof_prob_good"))
    require_same(random_metrics, original["image_random"][selected]["pooled"], "selected random OOF")
    confidence = [(label, score) for label, score in zip(labels, scores) if score <= 0.2 or score >= 0.8]
    confident_metrics = metrics([label for label, _ in confidence], [score for _, score in confidence])
    confident_fraction = len(confidence) / len(rows)
    diagnostics = original["selected_model_diagnostics"]
    require_same({"illustrative_20_80_confident_fraction": confident_fraction,
                  "illustrative_20_80_confident_balanced_accuracy": confident_metrics["balanced_accuracy"]},
                 diagnostics, "illustrative score bands")
    accepted = [label for label, score in zip(labels, scores) if score >= 0.8]
    bins = collections.defaultdict(list)
    for label, score in zip(labels, scores):
        bins[min(int(score * 10), 9)].append((label, score))
    ece = sum(len(items) / len(rows) * abs(sum(score for _, score in items) / len(items) -
              sum(label for label, _ in items) / len(items)) for items in bins.values())
    report = {
        "status": "PASS_FROZEN_PREDICTION_METRICS", "checked_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": INPUTS, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": len(rows), "acquisition_dates": len({row["acquisition_date"] for row in rows}),
        "comparison_tolerance": TOLERANCE, "selected_model": selected,
        "date_grouped_metrics_recomputed": grouped, "selected_image_random_metrics_recomputed": random_metrics,
        "selected_random_minus_grouped_auc": random_metrics["roc_auc"] - grouped[selected]["roc_auc"],
        "illustrative_20_80_bands": {"confident_fraction": confident_fraction,
                                    "confident_balanced_accuracy": confident_metrics["balanced_accuracy"],
                                    "n_predicted_good_at_least_0_8": len(accepted),
                                    "bad_fraction_in_predicted_good_at_least_0_8": accepted.count(0) / len(accepted)},
        "equal_width_10_bin_ece": ece,
        "review_queues": review_queues(rows, selected + "_prob_good"),
        "bootstrap_intervals_recomputed": False, "models_or_original_folds_retrained": False,
        "images_or_model_weights_loaded": False,
        "scope": "Checks saved retrospective OOF predictions; does not independently reproduce training or prospective performance.",
        "elapsed_seconds": time.perf_counter() - started,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON receipt; print to stdout regardless")
    args = parser.parse_args()
    report = verify()
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(rendered)
        temporary.replace(args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
