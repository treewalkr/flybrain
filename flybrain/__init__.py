"""Paths for the flybrain package. Everything big lives under data/ (gitignored)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / "data"
MALECNS_DIR = LOCAL / "malecns"
FLY_DIR = LOCAL / "fly"
OUT_DIR = LOCAL / "out"

GRAPH_PATH = FLY_DIR / "brain_graph.npz"
CIRCUIT_PATH = FLY_DIR / "brain_circuit.npz"
RUNS_DIR = LOCAL / "runs"
