"""Download the MaleCNS v1.0 dataset (https://male-cns.janelia.org/download/).

    .venv/bin/python -m flybrain.fetch_data [--only malecns|mesh]

malecns   three Arrow feather tables (annotations, neurotransmitters, 1.05 GB
          connectome weights) -> data/malecns/
mesh      JRCFIB2022M brain shell (FlyEM MaleCNS ROI point-cloud shells) -> data/fly/

Sizes and sha256 pinned (mirrors of the official download page's bulk files).
Idempotent: a file present with the right digest is not re-downloaded.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from flybrain import FLY_DIR, LOCAL, MALECNS_DIR

MALECNS_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
MESH_URL = "https://storage.googleapis.com/flyem-male-cns/rois/pointcloud-shells/"


@dataclass(frozen=True)
class File:
    url: str
    path: Path
    size: int
    sha256: str


DATASETS: dict[str, list[File]] = {
    "malecns": [
        File(MALECNS_URL + "body-annotations-male-cns-v1.0-minconf-0.5.feather",
             MALECNS_DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather",
             14483314, "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2"),
        File(MALECNS_URL + "body-neurotransmitters-male-cns-v1.0.feather",
             MALECNS_DIR / "body-neurotransmitters-male-cns-v1.0.feather",
             43282834, "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621"),
        File(MALECNS_URL + "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
             MALECNS_DIR / "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
             1051241946, "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1"),
    ],
    "mesh": [
        File(MESH_URL + "JRCFIB2022M_brain.ply", FLY_DIR / "JRCFIB2022M_brain.ply",
             1255587, "13ea41a7ce2677eae00738c23b2be3c2f29ac17c6976c2209998a85efccaa427"),
    ],
}
PROGRESS_BYTES = 64 << 20


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(f: File) -> None:
    size = f.path.stat().st_size
    if size != f.size:
        raise RuntimeError(f"{f.path}: {size} bytes, expected {f.size}")
    digest = sha256_of(f.path)
    if digest != f.sha256:
        raise RuntimeError(f"{f.path}: sha256 {digest}, expected {f.sha256}")


def fetch(f: File) -> bool:
    """Download unless present and verified. Returns True if downloaded."""
    if f.path.exists():
        verify(f)
        return False
    f.path.parent.mkdir(parents=True, exist_ok=True)
    part = f.path.with_name(f.path.name + ".part")
    h = hashlib.sha256()
    n = 0
    with urllib.request.urlopen(f.url, timeout=300) as r, part.open("wb") as out:
        while block := r.read(1 << 20):
            out.write(block)
            h.update(block)
            if (n + len(block)) // PROGRESS_BYTES != n // PROGRESS_BYTES:
                print(f"  {f.path.name}: {(n + len(block)) / 1e6:.0f} / {f.size / 1e6:.0f} MB",
                      file=sys.stderr, flush=True)
            n += len(block)
    if n != f.size:
        raise RuntimeError(f"{f.url}: received {n} bytes, expected {f.size}")
    if h.hexdigest() != f.sha256:
        raise RuntimeError(f"{f.url}: sha256 {h.hexdigest()}, expected {f.sha256}")
    os.replace(part, f.path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=sorted(DATASETS), action="append",
                        help="restrict to these datasets (repeatable); default all")
    args = parser.parse_args()
    names = args.only or list(DATASETS)
    print(f"data root {LOCAL}", flush=True)
    for name in names:
        for f in DATASETS[name]:
            downloaded = fetch(f)
            print(f"{name:8s} {'downloaded' if downloaded else 'present   '} {f.path.relative_to(LOCAL)} "
                  f"({f.size / 1e6:.1f} MB, sha256 {f.sha256[:12]})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
