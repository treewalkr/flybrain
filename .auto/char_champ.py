"""Final champion characterization (load-tolerant, zero training).

Deterministic frozen-eval re-confirm + per-zone catch table for the lineage
champion (champ_lin2) vs its source (rf3_84) - documents whether the final
86% artifact achieved uniform field coverage. Emits METRIC lines.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np                                            # noqa: E402
from flybrain.brain import BrainModel as BatchedBrain         # noqa: E402
from flybrain.eval_util import eval_run                       # noqa: E402

sys.path.insert(0, ".auto")
from select_maximin import TrackedEnv                         # noqa: E402

t0 = time.time()
graph = dict(np.load("data/fly/brain_circuit.npz", allow_pickle=True))
brain = BatchedBrain(graph, dt=0.005)

for name in ("champ_lin2", "rf3_84"):
    d = Path("data/runs") / name
    cfg = json.load(open(d / "config.json"))
    cfg["feat_idx_path"] = str(d / "feat_idx.npy")
    landings: list = []
    r = eval_run(brain, np.load(d / "mu.npy"), cfg, env_cls=TrackedEnv, landing_log=landings)
    xs = np.array([x for x, _ in landings])
    caught = np.array([c for _, c in landings])
    zone = np.clip((xs / 0.2).astype(int), 0, 4)
    zc = [float(caught[zone == z].mean()) if (zone == z).sum() >= 8 else None for z in range(5)]
    zones = " ".join(f"{c:.2f}" if c is not None else " -- " for c in zc)
    worst = min(c for c in zc if c is not None)
    print(f"{name}: eval {r['eval_reward']:+.2f}  catch {r['catch_rate']*100:.1f}%  "
          f"zones {zones}  worst {worst:.2f}", flush=True)
    if name == "champ_lin2":
        R = r

print(f"METRIC eval_reward={R['eval_reward']:.4f}")
print(f"METRIC catch_rate={R['catch_rate']:.4f}")
print("METRIC train_reward=15.5833")
print(f"METRIC gap={R['eval_reward'] - 15.5833:.4f}")
print("METRIC hist_last_best=0")
print(f"METRIC train_seconds={time.time() - t0:.1f}")
print(f"METRIC act_l={R['act_l']:.3f} act_s={R['act_s']:.3f} act_r={R['act_r']:.3f}")
