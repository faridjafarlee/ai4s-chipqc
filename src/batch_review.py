"""Analyze an image folder with the existing model and export a review queue."""

from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlparse

from PIL import Image

from .app import MODEL_PATH, load_bundle, load_encoder, predict_image
from .embeddings import WEIGHTS_SHA256, WEIGHTS_URL


EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def thumbnail(path: Path) -> str:
    with Image.open(path) as original:
        image = original.convert("RGB")
        image.thumbnail((480, 320))
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def report_page(rows: list[dict], previews: dict[str, str], session: str, quota: int,
                attribution: dict | None = None) -> str:
    cards = []
    for row in sorted(rows, key=lambda x: x["priority_rank"]):
        name = html.escape(row["image"])
        selected = "In review budget" if row["within_budget"] else "Outside review budget"
        cards.append(f'''<article class="card">
<div class="row"><strong>#{row["priority_rank"]} · {name}</strong><span>{selected}</span></div>
<img src="{previews[row['image']]}" alt="Microscopy frame {name}">
<div class="row"><b>P(good): {row['prob_good']:.1%}</b><span>{html.escape(row['review_flag'])}</span></div>
<p class="muted">Appearance distance: {row['appearance_distance']:.2f} · Focus proxy: {row['focus_proxy']:.2f}</p>
</article>''')
    credits = ""
    if attribution:
        credits = (f'<p class="muted">Images: {html.escape(attribution["credit"])}. '
                   f'<a href="{html.escape(attribution["source_url"], quote=True)}">Source dataset</a> · '
                   f'<a href="{html.escape(attribution["license_url"], quote=True)}">Image license</a>. '
                   'JPEG previews resized and compressed; review annotations added.</p>')
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ChipQC · {html.escape(session)}</title><style>
body{{font:16px/1.5 system-ui,sans-serif;background:#f5f7f5;color:#15332b;margin:0}}
main{{max-width:1120px;margin:auto;padding:32px 20px}}h1{{font-size:42px;margin:0}}
.intro{{max-width:780px}}.muted{{color:#587165;font-size:14px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}}
.card{{background:white;border:1px solid #dbe5df;border-radius:14px;padding:18px}}
.card img{{width:100%;height:240px;object-fit:contain;margin:12px 0;background:#eef1ef}}
.row{{display:flex;gap:12px;justify-content:space-between;flex-wrap:wrap}}
a{{color:#176a4a}}footer{{margin-top:28px}}
</style></head><body><main><h1>ChipQC</h1><h2>{html.escape(session)}</h2>
<p class="intro">{len(rows)} frames · {quota} in the selected review budget. Lowest estimated probability of the expert “good” label comes first. The budget sets review order; it does not mark remaining frames as safe.</p>
<p><a href="predictions.csv" download>Download predictions CSV</a></p>
<div class="grid">{''.join(cards)}</div>
{credits}
<footer class="muted">Research prototype. Score bands at 20% and 80% are illustrative. Appearance flags are an unvalidated novelty heuristic. These results do not measure tissue function, drug response or clinical safety. The session name is supplied by the user; it does not establish chip or patient identity.</footer>
</main></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session", default=None)
    parser.add_argument("--review-fraction", type=float, default=0.3)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--image-credit", help="Optional image creator/dataset credit")
    parser.add_argument("--image-source-url", help="Source URL, required with image credit")
    parser.add_argument("--image-license-url", help="License URL, required with image credit")
    args = parser.parse_args()
    if not args.folder.is_dir() or args.output.exists():
        parser.error("Input must be a folder and output must be a new directory")
    if not 0 <= args.review_fraction <= 1 or not 1 <= args.max_images <= 1000:
        parser.error("Review fraction must be 0–1 and max-images 1–1000")
    attribution = None
    supplied = [args.image_credit, args.image_source_url, args.image_license_url]
    if any(x is not None for x in supplied):
        if not all(x and x.strip() for x in supplied):
            parser.error("Image credit, source URL and license URL must be supplied together")
        if any(urlparse(url).scheme not in {"http", "https"} or not urlparse(url).netloc
               for url in supplied[1:]):
            parser.error("Image source and license must be HTTP(S) URLs")
        attribution = {"credit": args.image_credit, "source_url": args.image_source_url,
                       "license_url": args.image_license_url,
                       "changes": "JPEG previews resized and compressed; review annotations added"}
    paths = sorted(p for p in args.folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths or len(paths) > args.max_images:
        parser.error("Folder must contain 1–max-images supported images")
    bundle = load_bundle()
    if bundle["name"] != "mobilenet_logistic" or "cell_type" in bundle["columns"]:
        parser.error("This folder workflow requires the documented image-only MobileNet model")
    session = args.session or args.folder.name
    model_sha = digest(MODEL_PATH)
    started = time.monotonic()
    rows, previews = [], {}
    for path in paths:
        if path.stat().st_size > 20_000_000:
            raise ValueError(f"Image exceeds 20 MB: {path.name}")
        with Image.open(path) as probe:
            if min(probe.size) < 32 or probe.width * probe.height > 16_000_000:
                raise ValueError(f"Unsupported decoded image dimensions: {path.name}")
            probe.verify()
        prediction = predict_image(path)
        p = prediction["prob_good"]
        flag = ("Appearance shift · review" if prediction["appearance_flag"] else
                "Likely poor · inspect" if p <= .2 else
                "Likely acceptable · verify" if p >= .8 else "Manual review recommended")
        measures = prediction["measurements"]
        rows.append({"image": path.name, "session": session, "prob_good": p,
                     "appearance_distance": prediction["appearance_distance"],
                     "appearance_flag": prediction["appearance_flag"], "review_flag": flag,
                     "focus_proxy": float(measures["center_laplacian_variance"]),
                     "brightness": float(measures["center_mean"]),
                     "edge_fraction": float(measures["center_edge_fraction"]),
                     "image_sha256": digest(path)})
        previews[path.name] = thumbnail(path)
    quota = math.ceil(len(rows) * args.review_fraction)
    for rank, row in enumerate(sorted(rows, key=lambda x: (x["prob_good"], x["image"])), 1):
        row.update(priority_rank=rank, within_budget=rank <= quota)
    if digest(MODEL_PATH) != model_sha:
        raise ValueError("Model changed during inference")
    args.output.mkdir(parents=True)
    with (args.output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "index.html").write_text(report_page(rows, previews, session, quota, attribution))
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "session": session, "images": len(rows), "review_quota": quota,
                  "review_fraction": args.review_fraction, "priority_rule": "lowest P(good), then filename",
                  "model_name": bundle["name"], "model_role": bundle.get("role", "full-data model"),
                  "model_sha256": model_sha, "model_training_images": bundle.get("n_training_images"),
                  "encoder_weights_sha256": WEIGHTS_SHA256, "encoder_weights_url": WEIGHTS_URL,
                  "image_attribution": attribution,
                  "native_encoder_device": load_encoder()[1],
                  "elapsed_seconds": time.monotonic() - started,
                  "input_hashes": {x["image"]: x["image_sha256"] for x in rows},
                  "source_hashes": {name: digest(Path(__file__).parent / name)
                                    for name in ["app.py", "batch_review.py", "features.py", "embeddings.py", "shift.py"]},
                  "output_hashes": {name: digest(args.output / name) for name in ["predictions.csv", "index.html"]},
                  "model_fitting": False, "accuracy_measured": False}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    main()
