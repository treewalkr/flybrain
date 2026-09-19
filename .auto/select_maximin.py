"""Maximin per-zone selection over cached artifacts (no new training).

Scores every data/runs/ms_* artifact on validation seeds 8500-8547 with
landing-position tracking; selects by WORST-quintile catch rate (tiebreak:
validation reward) - attacks the cold-cell / hemifield pathology directly.
Winner is confirmed once on the frozen eval seeds.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from flybrain.brain import BrainModel as BatchedBrain          # noqa: E402
from flybrain.eval_util import eval_run                        # noqa: E402
from flybrain.game import CatchEnv                             # noqa: E402


class TrackedEnv(CatchEnv):
    """Records (landing_x, caught) for every resolved ball."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.landings = []
        self._last_r = 0.0

    def _new_ball(self, first: bool = False) -> None:
        if not first:
            self.landings.append((self.ball_x, self._last_r > 0))
        super()._new_ball(first)

    def step(self, action: int):
        r, done = super().step(action)
        if r != 0.0:
            self._last_r = r
            if done:                                  # last ball: no _new_ball call
                self.landings.append((self.ball_x, r > 0))
        return r, done


CFG = json.load(open(".auto/config.json"))
VAL_SEEDS = list(range(8500, 8548))
t0 = time.time()

rows = []
for d in sorted(Path("data/runs").glob("ms_*")):
    if not (d / "mu.npy").exists():
        continue
    cfg = json.load(open(d / "config.json"))
    cfg["feat_idx_path"] = str(d / "feat_idx.npy")
    graph = dict(np.load(cfg["graph"], allow_pickle=True))
    brain = BatchedBrain(graph, dt=0.005, gain=cfg["gain"])
    landings: list = []
    v = eval_run(brain, np.load(d / "mu.npy"), cfg, seeds=VAL_SEEDS, batch=48,
                 env_cls=TrackedEnv, landing_log=landings)
    xs = np.array([x for x, _ in landings])
    caught = np.array([c for _, c in landings])
    zone = np.clip((xs / 0.2).astype(int), 0, 4)
    zc = [float(caught[zone == z].mean()) if (zone == z).sum() >= 8 else None
          for z in range(5)]
    worst = min(c for c in zc if c is not None)
    rows.append((d.name, v["eval_reward"], worst, zc))
    print(f"{d.name}: val {v['eval_reward']:+6.2f}  zones "
          f"{' '.join(f'{c:.2f}' if c is not None else ' -- ' for c in zc)}"
          f"  worst {worst:.2f}  ({time.time()-t0:.0f}s)", flush=True)

name, val, worst, zc = max(rows, key=lambda r: (r[2], r[1]))
print(f"maximin selected {name} (val {val:+.2f}, worst-zone {worst:.2f}) -> frozen eval", flush=True)
d = Path("data/runs") / name
cfg = json.load(open(d / "config.json"))
cfg["feat_idx_path"] = str(d / "feat_idx.npy")
graph = dict(np.load(cfg["graph"], allow_pickle=True))
brain = BatchedBrain(graph, dt=0.005, gain=cfg["gain"])
r = eval_run(brain, np.load(d / "mu.npy"), cfg)
print(f"METRIC eval_reward={r['eval_reward']:.4f}")
print(f"METRIC catch_rate={r['catch_rate']:.4f}")
print(f"METRIC train_reward={val:.4f}")
print(f"METRIC gap={r['eval_reward'] - val:.4f}")
print(f"METRIC hist_last_best={json.load(open(d / 'history.json'))[-1]['best']:.4f}")
print(f"METRIC train_seconds={time.time() - t0:.1f}")
print(f"METRIC act_l={r['act_l']:.3f} act_s={r['act_s']:.3f} act_r={r['act_r']:.3f}")
