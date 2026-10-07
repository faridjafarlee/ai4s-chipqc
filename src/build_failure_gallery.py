"""Render selected score/label disagreements from frozen held-out predictions.

This reads saved CSVs and four cached source images; it performs no fitting,
inference, aggregate metric calculation, or changes to the source images.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OOF = REPORTS / "grouped_oof_predictions.csv"
HOLDOUT = REPORTS / "demo_holdout_manifest.csv"
SOURCE_INDEX = ROOT / "data/image_manifest.csv"
PROBABILITY_COLUMN = "mobilenet_logistic_prob_good"
TOLERANCE = 1e-7
EXPECTED_INPUTS = {
    OOF: "203bc5fa780ec1c63aae3fdf74c51ee69412b18ee1429f2801644719c5090857",
    HOLDOUT: "bc4739df1d817173031dcb172dc58bc3f975c06ca7305b2b9412c3a04473f959",
}
EXPECTED_IMAGES = {
    "230215_24": "dcb119fd4895681a70edf2c88c78a8c266cefb3aed4f8ab113d2e41d38d5a629",
    "230215_23": "f768e831ad9b4be31b9caa9007b1b42677252b849dea40a84e8909087518db70",
    "230320_142": "973ff13bda4b85a07a0a9afc47510c2f370adb5b8946724ef0fb5ba1b4eb239c",
    "230320_107": "6e06bf0d211022bba0977c0b64e75ddaefb22b2b69b0df9903e1721e5e7bbf60",
}
SELECTION_RULE = (
    "Restrict to the saved first date-held-out validation fold. Select two bad "
    "expert-label frames with the highest saved P(good) >=0.8, then two good "
    "expert-label frames with the lowest saved P(good) <=0.2. Break ties by imageID."
)
ATTRIBUTION = (
    "Images: Movčana et al., Zenodo 2023, DOI 10.5281/zenodo.10203721, "
    "CC BY 4.0; resized."
)
DATASET_URL = "https://doi.org/10.5281/zenodo.10203721"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row, csv_line=line) for line, row in
                enumerate(csv.DictReader(handle), start=2)]


def index_rows(rows: list[dict]) -> dict[str, dict]:
    indexed = {row["imageID"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate imageID in input CSV")
    return indexed


def select_examples() -> tuple[list[dict], dict]:
    for path, expected in EXPECTED_INPUTS.items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen input SHA-256 mismatch: {path.relative_to(ROOT)}")
    holdout = read_rows(HOLDOUT)
    oof = index_rows(read_rows(OOF))
    index_rows(holdout)
    source_index = index_rows(read_rows(SOURCE_INDEX))
    for row in holdout:
        p = float(row["prob_good"])
        saved = oof[row["imageID"]]
        if row["good"] not in {"0", "1"} or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f"Invalid saved label or probability: {row['imageID']}")
        if any(row[key] != saved[key] for key in ("good", "cell_type", "acquisition_date")):
            raise ValueError(f"Holdout/OOF metadata mismatch: {row['imageID']}")
        if abs(p - float(saved[PROBABILITY_COLUMN])) > TOLERANCE:
            raise ValueError(f"Holdout/OOF probability mismatch: {row['imageID']}")
    bad_high = sorted(
        (row for row in holdout if row["good"] == "0" and float(row["prob_good"]) >= .8),
        key=lambda row: (-float(row["prob_good"]), row["imageID"]))[:2]
    good_low = sorted(
        (row for row in holdout if row["good"] == "1" and float(row["prob_good"]) <= .2),
        key=lambda row: (float(row["prob_good"]), row["imageID"]))[:2]
    selected = bad_high + good_low
    if [row["imageID"] for row in selected] != list(EXPECTED_IMAGES):
        raise ValueError("Fixed selection did not produce the four expected image IDs")
    examples = []
    for row in selected:
        image_id = row["imageID"]
        path = ROOT / "data/images" / f"{image_id}.png"
        source = source_index[image_id]
        label = "good" if row["good"] == "1" else "bad"
        image_hash = sha256(path)
        if image_hash != EXPECTED_IMAGES[image_id]:
            raise ValueError(f"Cached source image SHA-256 mismatch: {image_id}")
        if source["quality_label"] != label or source["cell_type"] != row["cell_type"]:
            raise ValueError(f"Source label/cell mismatch: {image_id}")
        if int(source["bytes"]) != path.stat().st_size:
            raise ValueError(f"Source byte count mismatch: {image_id}")
        saved = oof[image_id]
        if source["source_split"] != saved["source_split"]:
            raise ValueError(f"Source split mismatch: {image_id}")
        with Image.open(path) as image:
            if image.mode not in {"L", "RGB"}:
                raise ValueError(f"Unsupported source image mode: {image.mode}")
            width, height = image.size
            mode = image.mode
        examples.append({
            "imageID": image_id, "expert_label": label, "good": int(row["good"]),
            "cell_type": row["cell_type"], "acquisition_date": row["acquisition_date"],
            "date_display": datetime.strptime(row["acquisition_date"], "%y%m%d").date().isoformat(),
            "category": "bad_label_high_score" if label == "bad" else "good_label_low_score",
            "holdout_prob_good_raw": row["prob_good"],
            "grouped_oof_prob_good_raw": saved[PROBABILITY_COLUMN],
            "absolute_probability_difference": abs(float(row["prob_good"]) - float(saved[PROBABILITY_COLUMN])),
            "holdout_csv_line": row["csv_line"], "oof_csv_line": saved["csv_line"],
            "source_archive_path": source["archive_path"], "source_split": source["source_split"],
            "local_image": path.relative_to(ROOT).as_posix(), "image_bytes": path.stat().st_size,
            "image_sha256": image_hash, "source_width": width, "source_height": height,
            "source_mode": mode,
        })
    inputs = {path.relative_to(ROOT).as_posix(): sha256(path)
              for path in (OOF, HOLDOUT, SOURCE_INDEX)}
    return examples, {"sha256": inputs, "holdout_rows_checked": len(holdout),
                      "oof_probability_column": PROBABILITY_COLUMN,
                      "absolute_probability_tolerance": TOLERANCE}


def render_gallery(examples: list[dict], output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14})
    figure = plt.figure(figsize=(16, 15), facecolor="white")
    figure.text(.035, .974, "Selected image-score disagreements", fontsize=24, weight="bold")
    figure.text(.035, .948, "First date-held-out fold · expert source labels · saved MobileNet + logistic scores",
                fontsize=14, color="#37434b")
    for position, example in enumerate(examples):
        x = .035 if position % 2 == 0 else .52
        y = .52 if position < 2 else .105
        axis = figure.add_axes([x, y, .445, .356])
        with Image.open(ROOT / example["local_image"]) as image:
            pixels = np.asarray(image.convert("RGB"))
        axis.imshow(pixels, interpolation="lanczos", aspect="equal")
        axis.set_axis_off()
        caption = (f"{example['imageID']} · {example['cell_type']} · {example['date_display']}\n"
                   f"Expert label: {example['expert_label']}   |   saved P(good): "
                   f"{float(example['holdout_prob_good_raw']):.2%}")
        figure.text(x, y + .367, caption, fontsize=16, linespacing=1.5)
    figure.text(.035, .068, "Fixed selection: two highest scores among bad labels; two lowest among good labels.",
                fontsize=14)
    figure.text(.035, .045, "Selected cases do not estimate error prevalence. The app's appearance alert was not evaluated here.",
                fontsize=13, color="#37434b")
    figure.text(.035, .020, ATTRIBUTION, fontsize=12)
    figure.savefig(output, dpi=150, facecolor="white", metadata={"Software": "ChipQC build_failure_gallery"})
    plt.close(figure)


def write_report(examples: list[dict], figure_hash: str) -> None:
    rows = "\n".join(
        f"| {item['imageID']} | {item['expert_label']} | {item['cell_type']} | "
        f"{item['date_display']} | {item['holdout_prob_good_raw']} | "
        f"{item['grouped_oof_prob_good_raw']} |"
        for item in examples)
    report = f"""# Selected image-score disagreements

![Four saved score/label disagreements](failure_gallery.png)

## Scope and selection

{SELECTION_RULE}

These are deliberately selected disagreements from one validation fold, not an
estimate of error prevalence. Two bad-label examples share acquisition date
230215. The labels are the source experts' visual quality judgments; image
appearance alone does not establish biological failure, label error, or a cause.
The saved image scores are not the combined application's decisions. Its
appearance alert was not evaluated for these cases. No new aggregate performance
metric, threshold calibration, workflow benefit, or time saving is claimed.

| Image ID | Expert label | Cell type | Acquisition date | Holdout P(good) | Grouped OOF P(good) |
|---|---|---|---|---:|---:|
{rows}

## Reproduction and checks

From the repository directory, with the README's Python 3.13 environment and
source images already cached:

```bash
.venv/bin/python -m src.build_failure_gallery
```

The generator verifies frozen CSV hashes, deterministic four-ID selection,
metadata and probability agreement within 1e-7 for every held-out row, and the
four source images' hashes, labels, cell types, splits and byte counts. It reads
saved predictions and images only; it performs no fitting or model inference.
The first date-held-out fold is saved by `src/evaluate.py` in
`reports/demo_holdout_manifest.csv`. Source `train`/`test` archive folders are
metadata; they do not define this acquisition-date-held-out fold.

Exact inputs, CSV row numbers, raw probabilities, image properties and SHA-256
hashes are in [the manifest](failure_gallery_manifest.json). The PNG SHA-256 is
`{figure_hash}`. The source images are shown full-frame with preserved aspect
ratio and source color/grayscale; display resizing is the only image transform.
The original cached PNG files remain unchanged. The existing aggregate metrics
and study outputs are unchanged.

## Attribution and rights

{ATTRIBUTION}

Source: [Organ-on-a-Chip (OOC) Image Dataset]({DATASET_URL}),
Movčana et al., Zenodo, 2023. The figure's source image content remains under
[Creative Commons Attribution 4.0]({LICENSE_URL}); annotations and presentation
were added by the ChipQC project. Full creator attribution is retained in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). This gallery does not add a
clinical or biological validation claim.
"""
    (REPORTS / "FAILURE_GALLERY.md").write_text(report, encoding="utf-8")


def main() -> None:
    examples, inputs = select_examples()
    output = REPORTS / "failure_gallery.png"
    render_gallery(examples, output)
    figure_hash = sha256(output)
    manifest = {
        "schema_version": 1, "selection_rule": SELECTION_RULE,
        "scope": "Selected image-score disagreements from the first date-held-out validation fold",
        "inputs": inputs, "selected": examples,
        "checks": {"frozen_csv_hashes_match": True, "expected_four_ids_match": True,
                   "all_holdout_probabilities_and_metadata_agree": True,
                   "selected_source_image_hashes_and_metadata_match": True},
        "execution": {"model_fitting": False, "model_inference": False,
                      "aggregate_metrics_recomputed": False, "source_images_modified": False},
        "display": {"full_frame": True, "aspect_ratio_preserved": True,
                    "color_grayscale_preserved": True, "resampling": "Lanczos",
                    "output_pixels": [2400, 2250]},
        "attribution": {"visible_text": ATTRIBUTION, "dataset_url": DATASET_URL,
                        "license": "CC BY 4.0", "license_url": LICENSE_URL},
        "figure": {"path": output.relative_to(ROOT).as_posix(), "sha256": figure_hash},
    }
    (REPORTS / "failure_gallery_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_report(examples, figure_hash)
    print(json.dumps({"selected_ids": [item["imageID"] for item in examples],
                      "heldout_rows_checked": inputs["holdout_rows_checked"],
                      "figure_sha256": figure_hash}, indent=2))


if __name__ == "__main__":
    main()
