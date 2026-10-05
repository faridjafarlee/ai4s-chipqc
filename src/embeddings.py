"""Extract frozen ImageNet MobileNetV3 embeddings from organ-chip images."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data/images"
OUTPUT = ROOT / "data/mobilenet_embeddings.npz"
WEIGHTS_URL = "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"
WEIGHTS_SHA256 = "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_mobilenet_encoder(device: str) -> torch.nn.Module:
    """Download official weights with verified TLS and a pinned full checksum."""
    path = Path(torch.hub.get_dir()) / "checkpoints" / Path(WEIGHTS_URL).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with requests.get(WEIGHTS_URL, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with temporary.open("wb") as file:
                for chunk in response.iter_content(1024 * 1024):
                    file.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != WEIGHTS_SHA256:
            raise OSError("Pretrained weights failed SHA-256 verification")
        temporary.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != WEIGHTS_SHA256:
        raise OSError("Cached pretrained weights differ from the pinned source")
    model = mobilenet_v3_small(weights=None)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.classifier = torch.nn.Identity()
    return model.eval().to(device)


class ImageViews(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        path = self.paths[index]
        with Image.open(path) as original:
            image = original.convert("RGB")
            width, height = image.size
            center = image.crop((int(width * 0.20), int(height * 0.15),
                                 int(width * 0.80), int(height * 0.85)))
            views = torch.stack([PREPROCESS(image), PREPROCESS(center)])
        return path.stem, views


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if OUTPUT.exists():
        old = np.load(OUTPUT)
        ids = old["ids"].astype(str).tolist()
        values = old["features"].astype(np.float32).tolist()
    else:
        ids, values = [], []
    known = set(ids)
    selection = ROOT / "data/selection.csv"
    selected = set(pd.read_csv(selection).imageID) if selection.exists() else None
    paths = [path for path in sorted(IMAGES.glob("*.png")) if path.stem not in known
             and (selected is None or path.stem in selected)]
    print(f"Cached={len(ids)}; embedding={len(paths)}", flush=True)
    if not paths:
        return
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_mobilenet_encoder(device)
    loader = DataLoader(ImageViews(paths), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    with torch.inference_mode():
        for batch, (batch_ids, images) in enumerate(loader, 1):
            count = images.shape[0]
            embeddings = model(images.reshape(count * 2, 3, 224, 224).to(device))
            embeddings = embeddings.reshape(count, -1).cpu().numpy().astype(np.float32)
            ids.extend(batch_ids)
            values.extend(embeddings.tolist())
            if batch % 7 == 0 or batch * args.batch_size >= len(paths):
                np.savez_compressed(OUTPUT, ids=np.asarray(ids),
                                    features=np.asarray(values, dtype=np.float32))
                print(f"Embedded {min(batch * args.batch_size, len(paths))}/{len(paths)}; "
                      f"cached={len(ids)}; device={device}", flush=True)


if __name__ == "__main__":
    main()
