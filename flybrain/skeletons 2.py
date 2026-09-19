"""Skeletons for rendering: download SWCs for the circuit's neurons, parse, cache.

    .venv/bin/python -m flybrain.skeletons          # -> data/fly/skeletons.npz

Coordinates: SWC in 8 nm voxels -> micrometres. One packed SkeletonSet for every
neuron of data/fly/brain_circuit.npz (order matches graph["body_id"]).
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from flybrain import FLY_DIR

SWC_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/{body_id}.swc"
VOXEL_NM = 8.0
SWC_DIR = FLY_DIR / "skeletons"
CACHE = FLY_DIR / "skeletons.npz"


def parse_swc(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SWC -> (xyz um (N,3) f32, parent idx i32 (-1 root), radius um f32)."""
    rows = [ln for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]
    a = np.array(" ".join(rows).split(), dtype=np.float64).reshape(-1, 7)
    ids = a[:, 0].astype(np.int64)
    par = a[:, 6].astype(np.int64)
    order = np.argsort(ids)
    ids_sorted = ids[order]
    has_parent = par >= 0
    pos = np.searchsorted(ids_sorted, par[has_parent])
    if np.any(pos >= len(ids)) or np.any(ids_sorted[np.minimum(pos, len(ids) - 1)] != par[has_parent]):
        raise ValueError(f"{path}: parent id not found")
    parent = np.full(len(ids), -1, dtype=np.int32)
    parent[has_parent] = order[pos]
    scale = VOXEL_NM / 1000.0
    return (a[:, 2:5] * scale).astype(np.float32), parent, (a[:, 5] * scale).astype(np.float32)


def _fetch(body_id: int) -> int:
    dst = SWC_DIR / f"{body_id}.swc"
    if dst.exists():
        return 0
    tmp = dst.with_suffix(".part")
    with urllib.request.urlopen(SWC_URL.format(body_id=body_id), timeout=120) as r:
        data = r.read()
    if not data.startswith(b"#") and not data.lstrip()[:1].isdigit():
        raise RuntimeError(f"{body_id}: unexpected SWC payload")
    tmp.write_bytes(data)
    os.replace(tmp, dst)
    return len(data)


def build(body_ids: np.ndarray, workers: int = 24) -> dict[str, np.ndarray]:
    """Download missing SWCs for body_ids, pack into one array set (cached)."""
    if CACHE.exists():
        z = np.load(CACHE)
        if np.array_equal(z["body_ids"], body_ids):
            return {k: z[k] for k in z.files}
    SWC_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = nbytes = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for got in ex.map(lambda b: _fetch(int(b)), body_ids):
            n += int(got > 0)
            nbytes += got
    print(f"skeletons: downloaded {n} files ({nbytes / 1e6:.0f} MB) in {time.time() - t0:.0f} s", file=sys.stderr)

    xyzs, parents, radii, offsets = [], [], [], [0]
    for b in body_ids:
        xyz, parent, radius = parse_swc(SWC_DIR / f"{int(b)}.swc")
        base = offsets[-1]
        parents.append(np.where(parent >= 0, parent + base, -1).astype(np.int32))
        xyzs.append(xyz)
        radii.append(radius)
        offsets.append(base + len(xyz))
    out = dict(
        body_ids=np.asarray(body_ids, np.int64),
        xyz=np.concatenate(xyzs), parent=np.concatenate(parents),
        radius=np.concatenate(radii), offsets=np.asarray(offsets, np.int64),
    )
    np.savez(CACHE, **out)
    print(f"packed {len(body_ids)} skeletons, {len(out['xyz']):,} nodes -> {CACHE}", file=sys.stderr)
    return out


def load() -> dict[str, np.ndarray]:
    z = np.load(CACHE)
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    graph = dict(np.load(FLY_DIR / "brain_circuit.npz", allow_pickle=True))
    build(graph["body_id"])
