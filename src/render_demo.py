"""Assemble a captioned demo from actual application captures and study plots."""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "demo/capture"
SIZE = (1280, 720)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    paths = [Path("/System/Library/Fonts/Supplemental") /
             ("Arial Bold.ttf" if bold else "Arial.ttf"),
             Path("/usr/share/fonts/truetype/dejavu") /
             ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")]
    for path in paths:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def slide(name: str, title: str, paragraphs: list[str],
          picture: Path | None = None) -> Path:
    canvas = Image.new("RGB", SIZE, "#f4f7f3")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, SIZE[0], 12), fill="#176a4a")
    draw.text((62, 38), "CHIPQC  /  AI4S OPEN INNOVATION", font=font(18, True), fill="#237758")
    draw.text((60, 80), title, font=font(42, True), fill="#15332b")
    y = 160
    for paragraph in paragraphs:
        for line in textwrap.wrap(paragraph, 76):
            draw.text((64, y), line, font=font(27), fill="#355549")
            y += 40
        y += 20
    if picture:
        with Image.open(picture) as source:
            image = source.convert("RGB")
            image.thumbnail((1150, 480), Image.Resampling.LANCZOS)
            canvas.paste(image, ((SIZE[0] - image.width) // 2, 165))
    draw.text((64, 678), "Farid Jafarli · research prototype · captioned demonstration",
              font=font(16), fill="#647c70")
    path = CAPTURE / "slides" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def caption_frame(source: Path, caption: str) -> Path:
    with Image.open(source) as image:
        canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", SIZE)
    draw = ImageDraw.Draw(overlay)
    lines = textwrap.wrap(caption, 84)
    top = SIZE[1] - len(lines) * 29 - 30
    draw.rectangle((0, top, SIZE[0], SIZE[1]), fill=(15, 34, 26, 230))
    for index, line in enumerate(lines):
        draw.text((SIZE[0] // 2, top + 14 + index * 29), line,
                  anchor="mt", font=font(21), fill="white")
    destination = CAPTURE / "captioned" / source.name
    destination.parent.mkdir(exist_ok=True)
    Image.alpha_composite(canvas, overlay).convert("RGB").save(destination)
    return destination


def main() -> None:
    results = json.loads((ROOT / "reports/evaluation.json").read_text())
    recording = json.loads((CAPTURE / "timeline.json").read_text())
    selected = results["selected_model"]
    grouped = results["date_grouped"][selected]["pooled"]
    random = results["image_random"][selected]["pooled"]
    interval = results["selected_model_diagnostics"]["date_bootstrap_95pct_auc_interval"]
    n = results["dataset"]["n_images"]
    items: list[tuple[Path, float]] = [
        (slide("title", "ChipQC", [
            "Organ-on-a-chip quality review across imaging sessions",
            "Tool & Platform · working local application and reproducible study",
        ]), 7),
        (slide("problem", "Does the model handle a new imaging session?", [
            f"{n:,} public brightfield frames · six cell types · 59 acquisition-date prefixes",
            "Compare random image folds with folds that exclude entire acquisition dates.",
            "Estimate the source experts' good/bad sample-quality label, then keep a scientist in the review loop.",
        ]), 13),
    ]
    offset = sum(duration for _, duration in items)
    frames = recording["frames"]
    for index, frame in enumerate(frames):
        end = frames[index + 1]["t"] if index + 1 < len(frames) else recording["duration"]
        path = CAPTURE / frame["file"]
        with Image.open(path) as image:
            if image.size != SIZE:
                raise ValueError(f"Capture has unexpected dimensions: {image.size}")
        event = [item for item in recording["events"] if item["t"] <= frame["t"]][-1]
        path = caption_frame(path, event["caption"])
        items.append((path, end - frame["t"]))
    items += [
        (slide("validation", "Measured validation across five candidates", [],
               ROOT / "reports/validation_comparison.png"), 14),
        (slide("results", "Results and their limits", [
            f"Selected model: {selected.replace('_', ' ')}",
            f"ROC AUC with dates excluded: {grouped['roc_auc']:.3f}; random folds: {random['roc_auc']:.3f}",
            f"Acquisition-date bootstrap 95% interval: {interval[0]:.3f} to {interval[1]:.3f}",
            "The review thresholds and appearance alert are exploratory. New laboratories and biological function remain untested.",
        ]), 17),
        (slide("reproduce", "Reproduce the study and demo", [
            "Install the pinned dependencies from requirements.txt.",
            "python -m src.fetch_ooc --limit 0 --workers 8",
            "python -m src.features",
            "python -m src.embeddings",
            "python -m src.evaluate",
            "Start the held-out demo checkpoint using the README command.",
        ]), 17),
        (slide("credits", "Source, attribution, and intended use", [
            "Microscopy: Movčana et al., OOC Image Dataset, Zenodo 2023.",
            "DOI: 10.5281/zenodo.10203721 · CC BY 4.0",
            "Original code: MIT. Pretrained MobileNet weights: fetched separately from the official PyTorch source.",
            "A research aid for reviewing expert sample-quality labels. Scientists retain the decision.",
        ]), 12),
    ]
    concat = CAPTURE / "video-frames.txt"
    lines = []
    for path, duration in items:
        if duration <= 0:
            raise ValueError("Capture frame durations must be positive")
        lines.extend([f"file '{path}'", f"duration {duration:.6f}"])
    lines.append(f"file '{items[-1][0]}'")
    concat.write_text("\n".join(lines) + "\n")
    captions = []
    for index, event in enumerate(recording["events"]):
        end = (recording["events"][index + 1]["t"] if index + 1 < len(recording["events"])
               else recording["duration"])
        captions.extend([str(index + 1),
                         f"{timestamp(offset + event['t'])} --> {timestamp(offset + end)}",
                         "\n".join(textwrap.wrap(event["caption"], 84)), ""])
    srt = ROOT / "demo/chipqc-demo.srt"
    srt.write_text("\n".join(captions))
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("Install ffmpeg to render the demonstration")
    output = ROOT / "demo/chipqc-demo.mp4"
    intended_duration = sum(value for _, value in items)
    video_filter = "fps=24,format=yuv420p"
    subprocess.run([executable, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(concat), "-vf", video_filter, "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "20", "-movflags", "+faststart",
                    "-t", f"{intended_duration:.6f}",
                    str(output)], check=True)
    probe = subprocess.run([shutil.which("ffprobe") or "ffprobe", "-v", "error",
                            "-show_entries", "format=duration", "-of", "json",
                            str(output)], check=True, capture_output=True, text=True)
    duration = float(json.loads(probe.stdout)["format"]["duration"])
    if abs(duration - intended_duration) > 0.1 or duration > 300:
        raise ValueError(f"Unexpected rendered video duration: {duration}")
    (ROOT / "demo/video_metadata.json").write_text(json.dumps({
        "duration_seconds": duration, "format": "1280x720 H.264, captioned",
        "model": selected, "study_images": n,
        "source_credit": "Movčana et al., Zenodo 2023, DOI 10.5281/zenodo.10203721, CC BY 4.0",
        "capture_note": "Frames captured from the running application; illustrative excluded-date cases",
    }, indent=2))
    print(f"Rendered {output}; verified duration {duration:.1f} seconds")


if __name__ == "__main__":
    main()
