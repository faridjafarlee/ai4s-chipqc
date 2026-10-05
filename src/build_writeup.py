"""Build the required Kaggle writeup sections from verified study artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--video-url", required=True)
    args = parser.parse_args()
    repo = args.repo_url.rstrip("/")
    results = json.loads((ROOT / "reports/evaluation.json").read_text())
    name = results["selected_model"]
    grouped = results["date_grouped"][name]["pooled"]
    random = results["image_random"][name]["pooled"]
    sample = results["dataset"]
    summary = f"""ChipQC helps scientists review brightfield images from organ-on-a-chip cultures. It estimates agreement with an expert good/bad sample-quality label, presents focus and appearance measurements, and directs uncertain frames to human review. Its intended role is a research quality-control aid during imaging review.

The central question is whether an apparently strong image model remains useful when the imaging session changes. We evaluated {sample['n_images']:,} public microscopy frames covering six cell types and {sample['acquisition_dates']} acquisition-date prefixes. Five candidate pipelines compare handcrafted image descriptors, frozen MobileNet features, culture metadata, and their combination. Preprocessing and fitted transformations stay inside training folds. We compare ordinary stratified image-random validation with five-fold validation that excludes entire dates from training.

The selected {name.replace('_', ' ')} model achieved pooled ROC AUC {grouped['roc_auc']:.3f} on excluded dates, compared with {random['roc_auc']:.3f} on random folds. The report includes acquisition-date bootstrap intervals, calibration, subgroup results, and tests that exclude an entire cell type. These results describe one retrospective dataset; prospective performance in another laboratory remains unknown.

The working local application accepts an image, computes its model score, shows image diagnostics, and flags ambiguous estimates or unusual appearance for review. The video uses a checkpoint whose training dates exclude every demonstrated example. The source data, pinned dependencies, model checkpoints, out-of-fold predictions, and exact reproduction commands are documented publicly. Dataset imagery is attributed under CC BY 4.0; original code is MIT licensed. Biological function and drug response require separate experimental validation."""
    word_count = len(summary.split())
    assert 200 <= word_count <= 300, word_count
    report = (ROOT / "reports/TECHNICAL_REPORT.md").read_text()
    report = "\n".join("#" + line if line.startswith("#") else line
                       for line in report.splitlines()[2:])
    github_path = repo.removeprefix("https://github.com/")
    raw = f"https://raw.githubusercontent.com/{github_path}/main/reports/"
    for filename in ("validation_comparison.png", "selected_model_diagnostics.png"):
        report = report.replace(f"]({filename})", f"]({raw}{filename})")
    report = report.replace("](../README.md)", f"]({repo}/blob/main/README.md)")
    content = f"""## Category Declaration

Tool & Platform

## Demo Video

[Watch the actual ChipQC demonstration]({args.video_url})

## Code Repository Link

[Public reproducible repository]({repo})

## Project Summary

{summary}

## Technical Report

{report}
"""
    (ROOT / "reports/KAGGLE_WRITEUP.md").write_text(content)
    print(f"Saved writeup; project summary = {word_count} words")


if __name__ == "__main__":
    main()
