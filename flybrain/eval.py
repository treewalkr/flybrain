"""Held-out evaluation of a trained run on the frozen EVAL_SEEDS.

    .venv/bin/python -m flybrain.eval data/runs/cem_v0

Prints and emits machine-readable lines:
    eval_reward=<mean> catches=<mean> episodes=<n> catch_rate=<frac>

Eval seeds are frozen in flybrain.game.EVAL_SEEDS and are NEVER used in
training (train.py draws only from its own stream).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain.brain import BrainModel as BatchedBrain
from flybrain.eval_util import eval_run


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--w", type=Path, default=None, help="params file (default run/mu.npy)")
    a = p.parse_args()
    cfg = json.loads((a.run / "config.json").read_text())
    graph = dict(np.load(cfg["graph"], allow_pickle=True))
    brain = BatchedBrain(graph, dt=0.005, gain=cfg.get("gain", 1.0))
    params = np.load(a.w or a.run / "mu.npy")
    stats = eval_run(brain, params, cfg, seeds=None)
    print(f"eval_reward={stats['eval_reward']:.4f} catch_rate={stats['catch_rate']:.4f} "
          f"catches={stats['catches']:.2f} episodes={stats['episodes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
