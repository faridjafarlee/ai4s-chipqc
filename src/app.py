"""Local research demo for organ-on-a-chip image quality review."""

from __future__ import annotations

import base64
import html
import io
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

from .embeddings import PREPROCESS, load_mobilenet_encoder
from .features import image_features
from .shift import shift_distance


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("CHIPQC_MODEL_PATH", ROOT / "models/chipqc.joblib"))
app = FastAPI(title="ChipQC", description="Research prototype for organ-chip brightfield quality review")


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_encoder() -> tuple[torch.nn.Module, str]:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return load_mobilenet_encoder(device), device


def image_embedding(path: Path) -> list[float]:
    encoder, device = load_encoder()
    with Image.open(path) as original:
        image = original.convert("RGB")
        width, height = image.size
        center = image.crop((int(width * 0.20), int(height * 0.15),
                             int(width * 0.80), int(height * 0.85)))
        views = torch.stack([PREPROCESS(image), PREPROCESS(center)]).to(device)
    with torch.inference_mode():
        return encoder(views).cpu().numpy().reshape(-1).tolist()


def page(result: str = "", message: str = "") -> str:
    status = f'<div class="notice">{html.escape(message)}</div>' if message else ""
    uses_metadata = MODEL_PATH.exists() and "cell_type" in load_bundle()["columns"]
    metadata_fields = """<div class="fields"><div><label for="cell_type">Cell type</label><select id="cell_type" name="cell_type">
<option>A549</option><option>CACO</option><option>HPMEC</option><option>HSAEC</option><option>HUVEC</option><option>NHBE</option></select></div>
<div><label for="day">Day after seeding</label><input id="day" name="day" type="number" min="0" max="365" value="1"></div>
<div><label for="hours">Hours after seeding (optional)</label><input id="hours" name="hours" type="number" min="0" step="0.1"></div>
<div><label for="flow">Flow rate, µL/min (optional)</label><input id="flow" name="flow" type="number" min="0" step="0.01"></div>
<div><label for="density">Seeding density, cells/mL (optional)</label><input id="density" name="density" type="number" min="0"></div></div>""" if uses_metadata else '<p class="quiet">This model uses image appearance. Culture metadata is not required.</p>'
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ChipQC · Organ-on-a-chip image review</title>
<style>
:root {{ color-scheme: light; font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
* {{ box-sizing: border-box; }} body {{ margin:0; background:#f5f7f5; color:#15332b; }}
.shell {{ max-width:1120px; margin:auto; padding:36px 24px 64px; }}
.eyebrow {{ color:#237758; font-size:13px; letter-spacing:.14em; font-weight:800; text-transform:uppercase; }}
h1 {{ font-size:clamp(38px,5vw,64px); line-height:1.04; margin:12px 0; letter-spacing:-.045em; }}
h2 {{ margin:0 0 14px; font-size:22px; }} p {{ line-height:1.55; }}
.intro {{ max-width:720px; color:#426257; font-size:18px; }}
.grid {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:22px; margin-top:28px; }}
.card {{ background:white; border:1px solid #dbe5df; box-shadow:0 12px 34px #16372b10; border-radius:20px; padding:28px; }}
label {{ display:block; font-size:13px; font-weight:750; margin:15px 0 6px; }}
input,select {{ width:100%; padding:12px 13px; border:1px solid #becfc3; border-radius:10px; background:#fbfdfb; font:inherit; color:#19372b; }}
input[type=file] {{ padding:20px 12px; border:2px dashed #88b7a2; }}
.fields {{ display:grid; grid-template-columns:1fr 1fr; gap:0 14px; }}
button {{ width:100%; border:0; border-radius:12px; padding:15px; background:#176a4a; color:white; font:inherit; font-weight:760; cursor:pointer; margin-top:22px; }}
button:hover {{ background:#0d5138; }}
.quiet {{ color:#6b8175; font-size:13px; }} .notice {{ padding:14px; border-radius:10px; background:#fff4de; color:#795218; }}
.score {{ font-size:54px; letter-spacing:-.05em; font-weight:800; margin:4px 0; }}
.tag {{ display:inline-block; padding:8px 12px; border-radius:100px; background:#e2f2e9; color:#176a4a; font-weight:750; }}
.tag.review {{ background:#fff0d4; color:#8b580b; }} .tag.poor {{ background:#ffe7e4; color:#a1372d; }}
.preview {{ max-width:100%; max-height:285px; object-fit:contain; border-radius:10px; background:#eef1ef; margin:18px 0; }}
.metric {{ display:flex; justify-content:space-between; border-top:1px solid #e9efeb; padding:10px 0; font-size:14px; }}
footer {{ color:#647c70; font-size:13px; margin-top:26px; }}
@media(max-width:760px) {{ .grid,.fields {{ grid-template-columns:1fr; }} .shell {{ padding:24px 16px; }} }}
</style></head><body><div class="shell">
<div class="eyebrow">Life science · research prototype</div>
<h1>ChipQC</h1><p class="intro">A second look at organ-on-a-chip brightfield images. Upload a microscopy frame to estimate expert-rated sample quality and flag uncertain cases for human review.</p>
<div class="grid"><section class="card"><h2>Analyze a sample</h2>{status}
<form action="/analyze" method="post" enctype="multipart/form-data">
<label for="image">Brightfield image</label><input id="image" name="image" type="file" accept="image/png,image/jpeg,image/tiff" required>
{metadata_fields}
<button type="submit">Analyze image</button><p class="quiet">Runs locally on your computer.</p>
</form></section><section class="card"><h2>Review result</h2>{result or '<p class="quiet">Your probability, review flag, and measured image diagnostics will appear here.</p>'}</section></div>
<footer>Research quality-control aid only. This estimate does not measure drug response, tissue function, or clinical safety. The source data and evaluation protocol are documented in the repository.</footer>
</div></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    if not MODEL_PATH.exists():
        return page(message="No trained model yet. Run the documented data, feature, and evaluation commands first.")
    return page()


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(image: UploadFile = File(...), cell_type: str = Form("A549"),
                  day: float = Form(1), hours: float | None = Form(None),
                  flow: float | None = Form(None), density: float | None = Form(None)) -> str:
    if not MODEL_PATH.exists():
        return page(message="Train a model before analyzing images.")
    contents = await image.read()
    if len(contents) > 20_000_000:
        return page(message="Image exceeds the 20 MB demo limit.")
    try:
        with Image.open(io.BytesIO(contents)) as probe:
            if min(probe.size) < 32:
                return page(message="Please use an image at least 32 pixels wide and tall.")
            probe.verify()
    except Exception:
        return page(message="Please upload a valid PNG, JPEG, or TIFF microscopy image.")
    suffix = Path(image.filename or "image.png").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        suffix = ".png"
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / f"upload{suffix}"
        path.write_bytes(contents)
        measurements = image_features(path)
        bundle = load_bundle()
        record = {**measurements, "cell_type": cell_type, "day": day,
                  "time after seeding, h": hours, "flow_ul_min": flow,
                  "seeding_density": density}
        if bundle["name"] == "mobilenet_logistic":
            record.update(zip(bundle["embedding_columns"], image_embedding(path)))
        row = pd.DataFrame([record])
        probability = float(bundle["model"].predict_proba(row[bundle["columns"]])[0, 1])
        novelty = shift_distance(bundle["shift_screen"], row, bundle["image_columns"])
        shift_flag = novelty > bundle["shift_screen"]["threshold"]
    if shift_flag:
        label, style = "Appearance shift · manual review", "review"
    elif probability >= 0.8:
        label, style = "Likely acceptable · verify", ""
    elif probability <= 0.2:
        label, style = "Likely poor · inspect", "poor"
    else:
        label, style = "Manual review recommended", "review"
    image_source = "data:image/" + ("jpeg" if suffix in {".jpg", ".jpeg"} else "png")
    encoded = base64.b64encode(contents).decode("ascii") if suffix in {".png", ".jpg", ".jpeg"} else ""
    preview = f'<img class="preview" alt="Uploaded microscopy frame" src="{image_source};base64,{encoded}">' if encoded else ""
    diagnostic_rows = [
        ("Focus proxy · Laplacian variance", measurements["center_laplacian_variance"]),
        ("Central edge fraction", measurements["center_edge_fraction"]),
        ("Central brightness", measurements["center_mean"]),
        ("Appearance distance", novelty),
    ]
    rows = "".join(f'<div class="metric"><span>{html.escape(name)}</span><strong>{value:.2f}</strong></div>'
                   for name, value in diagnostic_rows)
    result = (f'<span class="tag {style}">{label}</span><div class="score">{probability:.1%}</div>'
              '<p class="quiet">Estimated probability of the expert “good” label. '
              'The 20%/80% review thresholds are illustrative, not a prospective guarantee.</p>'
              f'{preview}<h2>Image diagnostics</h2>{rows}'
              '<p class="quiet">Appearance distance is compared with a cross-date training threshold. '
              'It is a novelty heuristic, not a validated safety alarm. Other descriptors summarize image appearance; '
              'they are not a causal explanation of cell health.</p>')
    return page(result=result)
