"""Verify a frozen three-image batch run with Python's standard library only.

Checks saved artifacts and current source/model/input bytes; does not infer,
fit, download, render or rerun HTTP tests. It establishes functional parity,
not model accuracy or representative throughput.
"""
from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import html
from html.parser import HTMLParser
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class PageContents(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.images, self.links = [], []

    def handle_starttag(self, tag: str, attributes: list[tuple]) -> None:
        attrs = dict(attributes)
        if tag == "img":
            self.images.append(attrs)
        elif tag == "a":
            self.links.append(attrs.get("href"))


def verify(report: Path, manifest_path: Path, inputs: Path | None) -> dict:
    expected = json.loads(manifest_path.read_text())
    provenance = json.loads((report / "provenance.json").read_text())
    require(provenance["source_hashes"] == expected["source_hashes"], "Run source differs from frozen source")
    for name, value in expected["source_hashes"].items():
        require(digest(ROOT / "src" / name) == value, "Current source changed: " + name)
    model_path = ROOT / expected["model_path"]
    require(digest(model_path) == provenance["model_sha256"] == expected["model_sha256"], "Model checksum mismatch")
    require(provenance["encoder_weights_sha256"] == expected["encoder_weights_sha256"], "Encoder checksum mismatch")
    require(provenance["input_hashes"] == expected["input_hashes"], "Recorded input hashes mismatch")
    if inputs:
        require({p.name for p in inputs.iterdir() if p.is_file()} == set(expected["input_hashes"]), "Input folder identity/coverage changed")
        for name, value in expected["input_hashes"].items():
            require(digest(inputs / name) == value, "Input checksum mismatch: " + name)
    for name, value in provenance["output_hashes"].items():
        require(digest(report / name) == value, "Output checksum mismatch: " + name)
    require(set(provenance["output_hashes"]) == {"predictions.csv", "index.html"}, "Unexpected outputs")
    require(not provenance["model_fitting"] and not provenance["accuracy_measured"], "Functional run mislabeled as training/accuracy")
    require(provenance["model_name"] == "mobilenet_logistic" and provenance["model_role"] == "date_holdout_demo", "Unexpected model role")
    examples_path = ROOT / expected["examples_path"]
    require(digest(examples_path) == expected["examples_sha256"], "Saved demo reference changed")
    examples = json.loads(examples_path.read_text())["examples"]
    reference = {Path(row["path"]).name: row["prob_good"] for row in examples}
    with (report / "predictions.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = [row["image"] for row in rows]
    require(len(names) == len(set(names)) == 3 and set(names) == set(reference) == set(expected["input_hashes"]), "CSV identity/coverage mismatch")
    errors = {}
    for row in rows:
        probability = float(row["prob_good"])
        require(math.isfinite(probability) and 0 <= probability <= 1, "Invalid probability")
        require(all(math.isfinite(float(row[key])) for key in
                    ("appearance_distance", "focus_proxy", "brightness", "edge_fraction")), "Nonfinite diagnostic")
        require(row["image_sha256"] == expected["input_hashes"][row["image"]], "CSV image digest mismatch")
        require(row["session"] == provenance["session"] == expected["session"], "Session mismatch")
        errors[row["image"]] = abs(probability - reference[row["image"]])
    require(max(errors.values()) <= expected["absolute_probability_tolerance"], "Recorded example probability parity failed")
    quota = math.ceil(len(rows) * expected["review_fraction"])
    require(provenance["images"] == 3 and provenance["review_fraction"] == expected["review_fraction"]
            and provenance["review_quota"] == quota, "Review quota mismatch")
    ordered = sorted(rows, key=lambda x: (float(x["prob_good"]), x["image"]))
    for rank, row in enumerate(ordered, 1):
        require(int(row["priority_rank"]) == rank and row["within_budget"] == str(rank <= quota), "Priority/budget mismatch")
    page = (report / "index.html").read_text()
    parsed = PageContents()
    parsed.feed(page)
    require(len(parsed.images) == 3 and {x["alt"] for x in parsed.images} == {"Microscopy frame " + name for name in names}, "HTML preview coverage mismatch")
    for image in parsed.images:
        require(image["src"].startswith("data:image/jpeg;base64,"), "Missing embedded JPEG preview")
        pixels = base64.b64decode(image["src"].split(",", 1)[1], validate=True)
        require(pixels.startswith(b"\xff\xd8") and pixels.endswith(b"\xff\xd9"), "Invalid JPEG preview bytes")
    attribution = expected["image_attribution"]
    require(provenance["image_attribution"] == attribution, "Recorded image attribution mismatch")
    require(attribution["credit"] in html.unescape(page) and attribution["source_url"] in parsed.links
            and attribution["license_url"] in parsed.links and attribution["changes"] in html.unescape(page), "Visible attribution/license/changes missing")
    require("predictions.csv" in parsed.links, "CSV download link missing")
    elapsed = float(provenance["elapsed_seconds"])
    require(math.isfinite(elapsed) and elapsed > 0, "Invalid measured runtime")
    return {"checked_utc": datetime.now(timezone.utc).isoformat(), "status": "CURRENT_SOURCE_NATIVE_BATCH_ARTIFACTS_PASS",
            "images": len(rows), "review_quota": quota,
            "priority_order": [row["image"] for row in ordered],
            "absolute_probability_errors": errors,
            "maximum_absolute_probability_error": max(errors.values()),
            "absolute_probability_tolerance": expected["absolute_probability_tolerance"],
            "native_encoder_device": provenance["native_encoder_device"],
            "native_elapsed_seconds_after_imports": elapsed,
            "source_hashes": provenance["source_hashes"], "model_sha256": provenance["model_sha256"],
            "encoder_weights_sha256": provenance["encoder_weights_sha256"],
            "input_hashes": provenance["input_hashes"], "output_hashes": provenance["output_hashes"],
            "provenance_sha256": digest(report / "provenance.json"), "manifest_sha256": digest(manifest_path),
            "inputs_verified_from_actual_files": bool(inputs), "image_attribution_verified": True,
            "verifier_source_sha256": digest(Path(__file__)),
            "model_inferred_by_verifier": False, "model_fitting": False,
            "accuracy_measured": False, "http_tests_repeated": False,
            "scope": "Three selected recorded illustrations; not new validation or representative throughput"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, help="Optional local cached image folder for byte verification")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.report, args.manifest, args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
