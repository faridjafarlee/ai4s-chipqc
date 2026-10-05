"""Extract interpretable brightfield image descriptors for organ-chip QC."""

from __future__ import annotations

import argparse
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


IMAGE_ROOT = Path("data/images")
OUTPUT = Path("data/handcrafted_features.csv")


def _entropy(values: np.ndarray, bins: int = 32) -> float:
    counts = np.histogram(values, bins=bins, range=(0, 256))[0].astype(float)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def _cooccurrence(gray: np.ndarray, axis: int) -> tuple[float, float]:
    quantized = (gray // 16).astype(np.int32)
    left = quantized[:, :-1] if axis == 1 else quantized[:-1, :]
    right = quantized[:, 1:] if axis == 1 else quantized[1:, :]
    pairs = np.bincount((left * 16 + right).ravel(), minlength=256).reshape(16, 16)
    matrix = pairs / max(pairs.sum(), 1)
    distance = np.abs(np.arange(16)[:, None] - np.arange(16)[None, :])
    return float((matrix * distance**2).sum()), float((matrix / (1 + distance)).sum())


def _region_features(region: np.ndarray, prefix: str) -> dict[str, float]:
    gray = cv2.resize(region, (256, 256), interpolation=cv2.INTER_AREA)
    gray = np.ascontiguousarray(gray)
    values = gray.astype(np.float32)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    edges = cv2.Canny(gray, 60, 120)
    results = {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_entropy": _entropy(gray),
        f"{prefix}_laplacian_variance": float(laplacian.var()),
        f"{prefix}_gradient_mean": float(magnitude.mean()),
        f"{prefix}_gradient_std": float(magnitude.std()),
        f"{prefix}_edge_fraction": float((edges > 0).mean()),
        f"{prefix}_dark_fraction": float((gray < 40).mean()),
        f"{prefix}_bright_fraction": float((gray > 220).mean()),
        f"{prefix}_horizontal_contrast": float(np.abs(gx).mean()),
        f"{prefix}_vertical_contrast": float(np.abs(gy).mean()),
    }
    for percentile in (5, 25, 50, 75, 95):
        results[f"{prefix}_p{percentile}"] = float(np.percentile(gray, percentile))
    for axis, name in ((1, "horizontal"), (0, "vertical")):
        contrast, homogeneity = _cooccurrence(gray, axis)
        results[f"{prefix}_{name}_texture_contrast"] = contrast
        results[f"{prefix}_{name}_texture_homogeneity"] = homogeneity
    angle = (np.arctan2(gy, gx) + math.pi) % math.pi
    orientation = np.minimum((angle / math.pi * 8).astype(np.int32), 7)
    histogram = np.bincount(orientation.ravel(), weights=magnitude.ravel(), minlength=8)
    histogram = histogram / max(histogram.sum(), 1e-9)
    for index, value in enumerate(histogram):
        results[f"{prefix}_orientation_{index}"] = float(value)
    fft = np.abs(np.fft.rfft2(values - values.mean())) ** 2
    fy = np.fft.fftfreq(values.shape[0])[:, None]
    fx = np.fft.rfftfreq(values.shape[1])[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    total = float(fft[radius > 0].sum()) + 1e-9
    for low, high, name in ((0.0, 0.03, "low"), (0.03, 0.10, "mid"),
                            (0.10, 0.25, "high"), (0.25, 0.75, "very_high")):
        results[f"{prefix}_fft_{name}"] = float(fft[(radius >= low) & (radius < high)].sum() / total)
    return results


def image_features(path: Path) -> dict[str, float | str]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"Cannot read image: {path}")
    height, width = image.shape
    center = image[int(height * 0.15):int(height * 0.85),
                   int(width * 0.20):int(width * 0.80)]
    row: dict[str, float | str] = {"imageID": path.stem}
    row.update(_region_features(image, "full"))
    row.update(_region_features(center, "center"))
    row["image_width"] = width
    row["image_height"] = height
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    existing = pd.read_csv(OUTPUT) if OUTPUT.exists() else pd.DataFrame()
    known = set(existing.imageID) if not existing.empty else set()
    selected = set(pd.read_csv("data/selection.csv").imageID) if Path("data/selection.csv").exists() else None
    images = [path for path in IMAGE_ROOT.glob("*.png") if path.stem not in known
              and (selected is None or path.stem in selected)]
    print(f"Existing={len(known)}; to extract={len(images)}", flush=True)
    pending: list[dict[str, float | str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(image_features, path): path for path in images}
        for number, future in enumerate(as_completed(futures), 1):
            pending.append(future.result())
            if number % 100 == 0 or number == len(images):
                data = pd.concat([existing, pd.DataFrame(pending)], ignore_index=True)
                data.sort_values("imageID").to_csv(OUTPUT, index=False)
                print(f"Extracted {number}/{len(images)}; cached={len(data)}", flush=True)
    if not images:
        print(f"Cached={len(existing)}", flush=True)


if __name__ == "__main__":
    main()
