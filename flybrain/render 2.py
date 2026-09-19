"""3D brain-activity renderer (adapted from TMNF-C brain_render.py).

Neuron skeletons as polylines, colour driven by per-neuron activity, inside the
JRCFIB2022M brain shell. Off-screen pyvista -> (H, W, 3) uint8 frames.

    skel = skeletons.load()
    r = BrainRenderer(skel, Camera.frontal())
    img = r.frame(activity)   # activity: (n_neurons,) in [0, 1], circuit order
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True

from flybrain import FLY_DIR

BRAIN_MESH = FLY_DIR / "JRCFIB2022M_brain.ply"
BRAIN_CENTER = np.array([386.0, 228.0, 211.0], dtype=np.float32)  # um, shell bounds centre

# activity -> (r, g, b, alpha): 0 dim translucent grey-blue .. 1 hot white-yellow
COLOR_STOPS = np.array([
    [0.00, 0.13, 0.17, 0.30, 0.03],
    [0.20, 0.26, 0.30, 0.46, 0.18],
    [0.45, 0.95, 0.45, 0.12, 0.85],
    [0.75, 1.00, 0.78, 0.30, 1.00],
    [1.00, 1.00, 0.98, 0.88, 1.00],
], dtype=np.float64)
BACKGROUND = (0.015, 0.016, 0.025)
SHELL_COLOR = (0.30, 0.38, 0.60)


@dataclass(frozen=True)
class Camera:
    position: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    view_up: tuple[float, float, float]
    view_angle: float = 30.0
    orbit_deg_per_s: float = 0.0

    @classmethod
    def frontal(cls, distance: float = 900.0, elevation_deg: float = 14.0, azimuth_deg: float = -24.0,
                orbit_deg_per_s: float = 0.0) -> "Camera":
        el, az = math.radians(elevation_deg), math.radians(azimuth_deg)
        d = np.array([math.sin(az) * math.cos(el), -math.sin(el), -math.cos(az) * math.cos(el)])
        pos = BRAIN_CENTER + distance * d
        return cls(tuple(pos), tuple(BRAIN_CENTER), (0.0, -1.0, 0.0), 30.0, orbit_deg_per_s)


# ---------------------------------------------------------------- geometry

def decimate(xyz: np.ndarray, parent: np.ndarray, offsets: np.ndarray, spacing: float):
    """Drop chain nodes >= spacing apart along the arbor; keep roots/branch/tips.

    Returns (points, segments, owner) where owner maps each kept node to its neuron.
    """
    n = len(xyz)
    counts = np.diff(offsets)
    has_par = parent >= 0
    seglen = np.zeros(n, dtype=np.float32)
    seglen[has_par] = np.linalg.norm(xyz[has_par] - xyz[parent[has_par]], axis=1)

    d = seglen.astype(np.float64)           # path length from neuron root
    anc = parent.copy()
    while True:
        m = anc >= 0
        if not m.any():
            break
        d_prev, anc_prev = d.copy(), anc.copy()
        d[m] += d_prev[anc_prev[m]]
        anc[m] = anc_prev[anc_prev[m]]

    nchild = np.bincount(parent[has_par], minlength=n)
    neuron = np.repeat(np.arange(len(counts)), counts)
    # restart path-length at each neuron boundary: recompute per-neuron roots handled
    # by offsets; the pointer doubling above crosses neurons only if parent == -1 at
    # neuron starts (guaranteed by SWC packing), so d is per-neuron path length.
    bucket = np.floor(d / spacing).astype(np.int64)
    keep = ~has_par | (nchild != 1)
    keep[has_par] |= bucket[has_par] != bucket[parent[has_par]]

    anc = parent.copy()
    while True:
        bad = (anc >= 0) & ~keep[np.maximum(anc, 0)]
        if not bad.any():
            break
        anc[bad] = parent[anc[bad]]

    new_index = np.cumsum(keep) - 1
    kept = np.nonzero(keep)[0]
    kept_anc = anc[kept]
    child = kept[kept_anc >= 0]
    segments = np.stack([new_index[child], new_index[kept_anc[kept_anc >= 0]]], axis=1)
    owner = neuron[kept]
    return xyz[kept], segments, owner


class BrainRenderer:
    """Draw the circuit's skeletons coloured by per-neuron activity in [0, 1]."""

    def __init__(self, skel: dict[str, np.ndarray], camera: Camera, size: tuple[int, int] = (960, 540),
                 spacing_um: float = 2.0, brain_mesh: Path | None = BRAIN_MESH,
                 shell_opacity: float = 0.07, line_width: float = 1.0):
        self.camera = camera
        self.size = size
        t0 = time.time()
        self.plotter = pv.Plotter(off_screen=True, window_size=list(size), lighting="none")
        self.plotter.set_background(BACKGROUND)
        self.plotter.add_light(pv.Light(light_type="headlight", intensity=1.0))
        pts, segs, self.owner = decimate(skel["xyz"], skel["parent"], skel["offsets"], spacing_um)
        self.n_neurons = len(skel["body_ids"])
        cells = np.empty((len(segs), 3), dtype=np.int64)
        cells[:, 0] = 2
        cells[:, 1:] = segs
        mesh = pv.PolyData(pts, lines=cells.ravel())
        mesh.point_data["activity"] = np.zeros(len(pts), dtype=np.float32)
        import vtk
        from vtk.util import numpy_support
        x = np.linspace(0.0, 1.0, 256)
        table = np.stack([np.interp(x, COLOR_STOPS[:, 0], COLOR_STOPS[:, k]) for k in (1, 2, 3, 4)], axis=1) * 255.0
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        lut.SetRange(0.0, 1.0)
        lut.SetTable(numpy_support.numpy_to_vtk(table.astype(np.uint8), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR))
        actor = self.plotter.add_mesh(mesh, scalars="activity", clim=(0.0, 1.0), show_scalar_bar=False,
                                      line_width=line_width, render_lines_as_tubes=False, lighting=False)
        actor.mapper.lookup_table = lut
        actor.mapper.scalar_range = (0.0, 1.0)
        self.mesh = mesh
        self.scalars = mesh.point_data["activity"]
        if brain_mesh is not None and Path(brain_mesh).exists():
            shell = pv.read(brain_mesh)
            shell.points = (shell.points / 1000.0).astype(np.float32)   # nm -> um
            self.plotter.add_mesh(shell, color=SHELL_COLOR, opacity=shell_opacity,
                                  smooth_shading=True, specular=0.3, lighting=True)
        self._apply_camera(0.0)
        self.plotter.show(auto_close=False, interactive=False)
        self.build_seconds = time.time() - t0
        print(f"BrainRenderer: {self.n_neurons} neurons, {len(pts)} points, {len(segs)} segments, "
              f"built in {self.build_seconds:.1f} s", file=sys.stderr)

    def _apply_camera(self, t: float) -> None:
        cam = self.plotter.camera
        c = self.camera
        cam.position = c.position
        cam.focal_point = c.focal_point
        cam.up = c.view_up
        cam.view_angle = c.view_angle
        if c.orbit_deg_per_s:
            cam.Azimuth(c.orbit_deg_per_s * t)
        self.plotter.renderer.ResetCameraClippingRange()

    def frame(self, activity: np.ndarray, t: float = 0.0) -> np.ndarray:
        """activity: (n_neurons,) in [0,1] in circuit body_id order. Returns (H, W, 3) uint8."""
        a = np.asarray(activity, np.float32)
        if a.shape != (self.n_neurons,):
            raise ValueError(f"activity shape {a.shape}, expected ({self.n_neurons},)")
        self.scalars[:] = a[self.owner]
        self.scalars.VTKObject.Modified()
        self._apply_camera(t)
        self.plotter.render()
        return self.plotter.screenshot(return_img=True)

    def close(self) -> None:
        self.plotter.close()
