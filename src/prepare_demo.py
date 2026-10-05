"""Select clearly labeled illustrative examples from the excluded-date fold."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import pandas as pd

from .evaluate import load_data
from .shift import shift_distance


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    bundle = joblib.load(ROOT / "models/chipqc-demo-heldout.joblib")
    manifest = pd.read_csv(ROOT / "reports/demo_holdout_manifest.csv")
    data, image_columns, _ = load_data()
    data = data.set_index("imageID", drop=False)
    training_ids = set(bundle["training_image_ids"])
    training_dates = {value.split("_")[0] for value in training_ids}
    assert not set(manifest.acquisition_date.astype(str)).intersection(training_dates)
    candidates = manifest.copy()
    candidates["appearance_distance"] = [shift_distance(
        bundle["shift_screen"], data.loc[[image_id]], image_columns)
        for image_id in candidates.imageID]
    ordinary = candidates[candidates.appearance_distance <= bundle["shift_screen"]["threshold"]]
    definitions = [
        ("acceptable", ordinary[(ordinary.good == 1) & (ordinary.prob_good >= .8)], .92),
        ("poor", ordinary[(ordinary.good == 0) & (ordinary.prob_good <= .2)], .08),
        ("review", ordinary[(ordinary.prob_good > .2) & (ordinary.prob_good < .8)], .5),
    ]
    destination = ROOT / "demo/capture/images"
    destination.mkdir(parents=True, exist_ok=True)
    examples = []
    for role, frame, target in definitions:
        if frame.empty:
            raise ValueError(f"No suitable held-out illustration for {role}")
        chosen = frame.iloc[(frame.prob_good - target).abs().argmin()]
        image_id = chosen.imageID
        row = data.loc[image_id]
        path = destination / f"{role}.png"
        shutil.copy2(ROOT / "data/images" / f"{image_id}.png", path)
        examples.append({
            "role": role, "imageID": image_id, "path": str(path.relative_to(ROOT)),
            "expert_label": "good" if chosen.good else "bad",
            "acquisition_date": str(chosen.acquisition_date),
            "prob_good": float(chosen.prob_good),
            "appearance_distance": float(chosen.appearance_distance),
            "metadata": {field: None if pd.isna(row[column]) else row[column]
                         for field, column in {
                             "cell_type": "cell_type", "day": "day",
                             "hours": "time after seeding, h", "flow": "flow_ul_min",
                             "density": "seeding_density"}.items()},
        })
    (ROOT / "demo/examples.json").write_text(json.dumps({
        "model_role": bundle["role"], "model_name": bundle["name"],
        "n_training_images": bundle["n_training_images"],
        "note": "Selected illustrations, not a representative performance estimate",
        "source": "https://doi.org/10.5281/zenodo.10203721; CC BY 4.0",
        "examples": examples,
    }, indent=2, default=lambda value: value.item()))
    print("Selected excluded-date examples:", [(x["role"], x["imageID"]) for x in examples])


if __name__ == "__main__":
    main()
