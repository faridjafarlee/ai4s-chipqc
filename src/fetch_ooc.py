"""Fetch a reproducible subset of the CC BY 4.0 Zenodo OoC image archive.

HTTP byte ranges avoid downloading the complete 6.7 GB archive when a small
sample is sufficient for development. The source archive remains on Zenodo.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import struct
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


URL = "https://zenodo.org/api/records/10203721/files/OOC_image_dataset.zip/content"
SHEET_URL = "https://zenodo.org/api/records/10203721/files/OOC_datasheet.xlsx/content"
SHEET_MD5 = "a3d4875e3da4da83bac45c0ce727cb40"
SHEET_PATH = Path("data/raw/OOC_datasheet.xlsx")
IMAGE_ROOT = Path("data/images")
MANIFEST_PATH = Path("data/image_manifest.csv")
SELECTION_PATH = Path("data/selection.csv")
_local = threading.local()
_request_lock = threading.Lock()
_next_request_at = 0.0


def wait_for_request_slot() -> None:
    """Stay below Zenodo's published per-minute request budget."""
    global _next_request_at
    with _request_lock:
        now = time.monotonic()
        wait = max(0.0, _next_request_at - now)
        _next_request_at = max(now, _next_request_at) + 0.6
    if wait:
        time.sleep(wait)


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size
        self.position = 0
        self.session = requests.Session()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        origin = (0, self.position, self.size)[whence]
        self.position = origin + offset
        return self.position

    def read(self, count: int = -1) -> bytes:
        if count < 0:
            count = self.size - self.position
        count = min(count, self.size - self.position)
        if count <= 0:
            return b""
        start, end = self.position, self.position + count - 1
        for attempt in range(12):
            try:
                wait_for_request_slot()
                response = self.session.get(
                    self.url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=(15, 120),
                )
                if response.status_code == 429:
                    time.sleep(float(response.headers.get("Retry-After", "60")))
                    continue
                response.raise_for_status()
                if response.status_code != 206 or len(response.content) != count:
                    raise OSError(f"Expected {count} ranged bytes; got {response.status_code}")
                self.position += count
                return response.content
            except (requests.RequestException, OSError):
                if attempt == 11:
                    raise
                time.sleep(min(2**attempt, 30))
        raise AssertionError("unreachable")


def archive_size() -> int:
    response = requests.head(URL, allow_redirects=True, timeout=30)
    response.raise_for_status()
    return int(response.headers["Content-Length"])


def fetch_datasheet() -> None:
    if SHEET_PATH.exists() and hashlib.md5(SHEET_PATH.read_bytes()).hexdigest() == SHEET_MD5:
        return
    response = requests.get(SHEET_URL, timeout=(15, 60))
    response.raise_for_status()
    if hashlib.md5(response.content).hexdigest() != SHEET_MD5:
        raise OSError("Datasheet differs from the published Zenodo checksum")
    SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = SHEET_PATH.with_suffix(".tmp")
    temporary.write_bytes(response.content)
    temporary.replace(SHEET_PATH)


def fetch_member(info: zipfile.ZipInfo, size: int) -> bytes:
    """Fetch a ZIP member in one range, with a fallback for long local headers."""
    if not hasattr(_local, "reader"):
        _local.reader = HTTPRangeReader(URL, size)
    reader = _local.reader
    reader.seek(info.header_offset)
    header_budget = 4096
    block = reader.read(min(info.compress_size + header_budget,
                            size - info.header_offset))
    if block[:4] != b"PK\x03\x04":
        raise OSError(f"Invalid ZIP member header: {info.filename}")
    fields = struct.unpack("<4s5H3I2H", block[:30])
    flags, method = fields[2], fields[3]
    payload_start = 30 + fields[-2] + fields[-1]
    payload_end = payload_start + info.compress_size
    if flags & 1 or method != info.compress_type:
        raise OSError(f"Unsupported ZIP member: {info.filename}")
    if payload_end > len(block):
        block += reader.read(payload_end - len(block))
    compressed = block[payload_start:payload_end]
    if method == zipfile.ZIP_STORED:
        contents = compressed
    elif method == zipfile.ZIP_DEFLATED:
        contents = zlib.decompress(compressed, -15)
    else:
        raise OSError(f"Unsupported ZIP compression method {method}")
    if len(contents) != info.file_size or zlib.crc32(contents) != info.CRC:
        raise OSError(f"ZIP member failed size/CRC validation: {info.filename}")
    return contents


def image_id(info: zipfile.ZipInfo) -> str:
    return Path(info.filename).stem


def fetch_image(info: zipfile.ZipInfo, size: int) -> tuple[str, bool]:
    destination = IMAGE_ROOT / f"{image_id(info)}.png"
    if destination.exists() and destination.stat().st_size == info.file_size:
        return image_id(info), False
    contents = fetch_member(info, size)
    if len(contents) != info.file_size:
        raise OSError(f"Unexpected size for {info.filename}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_bytes(contents)
    temporary.replace(destination)
    return image_id(info), True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="0 downloads all images")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    fetch_datasheet()
    size = archive_size()
    with zipfile.ZipFile(HTTPRangeReader(URL, size)) as archive:
        infos = [item for item in archive.infolist() if item.filename.endswith(".png")]
    assert len(infos) == 3072, len(infos)
    assert len({image_id(info) for info in infos}) == len(infos)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["imageID", "archive_path", "source_split", "quality_label", "cell_type", "bytes"])
        for info in sorted(infos, key=lambda item: item.filename):
            parts = Path(info.filename).parts
            writer.writerow([image_id(info), info.filename, parts[1], parts[2],
                             parts[3].removeprefix("cell_type_").upper(), info.file_size])
    if args.limit:
        infos.sort(key=lambda item: hashlib.sha256(image_id(item).encode()).hexdigest())
        infos = infos[:args.limit]
    with SELECTION_PATH.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["imageID"])
        writer.writerows([image_id(info)] for info in infos)
    print(f"Indexed 3072 images; fetching {len(infos)} with {args.workers} workers", flush=True)
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fetch_image, info, size) for info in infos]
        for future in as_completed(futures):
            image, fetched = future.result()
            completed += 1
            if completed % 25 == 0 or completed == len(infos):
                print(f"{completed}/{len(infos)}; latest={image}; downloaded={fetched}", flush=True)


if __name__ == "__main__":
    main()
