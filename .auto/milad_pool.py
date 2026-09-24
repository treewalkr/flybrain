"""MI-family ladder, rung 1: is the MI-feature basin ladder-climbable?

Run 81 showed basin-climbability is family-specific (gain family matched
from-scratch but its rung-1 degraded -0.71). The MI family has the SAME
dimensionality as the readout family whose ladder produced the champion
(556 feats, 1668 params) and its best-of-4 (+8.25) matched readout's
best-of-14 (+8.42). Rung 1: two restart refinements from mis_142's mu with its
own feature set. Frozen-eval spend only above the champion gate +15.17.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np                                            # noqa: E402
from flybrain.brain import BrainModel as BatchedBrain         # noqa: E402
from flybrain.eval_util import eval_run                       # noqa: E402
from flybrain.train import train                              # noqa: E402

CFG = json.load(open(".auto/config.json"))
BASE = Path("data/runs/mis_142")
RUNG0_VAL = 8.25
CHAMP_GATE = 15.17
VAL_SEEDS = list(range(8500, 8548))

t0 = time.time()
mu0 = np.load(BASE / "mu.npy")
feat = np.load(BASE / "feat_idx.npy")
graph = dict(np.load(CFG["graph"], allow_pickle=True))
brain = BatchedBrain(graph, dt=0.005, gain=1.0)
brain.init_batch(1)

best = None
for s, sig in ((181, 0.15), (182, 0.12)):
    out = Path(f"data/runs/milad_{s}")
    if not (out / "mu.npy").exists():
        train(seed=s, out=out, workers=6, graph_path=CFG["graph"],
              iters=24, pop=64, elites=10, eps=3, use_sensory=True, sensory_gain=1.0,
              n_substeps=4, gain=1.0, dn_topk=256, sigma_decay=0.9, sigma_floor=0.02,
              train_balls=12, quiet=True, mu_init=mu0, sigma_init=sig,
              feat_idx_override=feat)
    c = {"n_substeps": 4, "sensory_gain": 1.0, "feat_idx_path": str(out / "feat_idx.npy")}
    v = eval_run(brain, np.load(out / "mu.npy"), c, seeds=VAL_SEEDS, batch=48)
    hist = json.load(open(out / "history.json"))
    print(f"milad {s} (sig {sig}): val {v['eval_reward']:+.2f} catch {v['catch_rate']*100:.0f}% "
          f"lastbest {hist[-1]['best']:+.1f} ({time.time()-t0:.0f}s)", flush=True)
    if best is None or v["eval_reward"] > best[1]:
        best = (s, v["eval_reward"], out)

s, val, out = best
hist = json.load(open(out / "history.json"))
climb = val - RUNG0_VAL
print(f"rung-1 best {val:+.2f} vs rung-0 {RUNG0_VAL:+.2f}: climb {climb:+.2f} "
      f"(readout family rung-1 climbed ~+2.6, gain family -0.71)", flush=True)

if val <= CHAMP_GATE:
    print(f"no eval spend (champion gate {CHAMP_GATE:+.2f}); "
          f"MI ladder {'ALIVE - continue in future iterations' if climb > 1.0 else 'DEAD'}",
          flush=True)
    print(f"METRIC eval_reward={val:.4f}")
    print(f"METRIC catch_rate=0 train_reward={val:.4f} gap=0 "
          f"hist_last_best={hist[-1]['best']:.4f} train_seconds={time.time()-t0:.1f}")
    sys.exit(0)

c = {"n_substeps": 4, "sensory_gain": 1.0, "feat_idx_path": str(out / "feat_idx.npy")}
r = eval_run(brain, np.load(out / "mu.npy"), c)
print(f"METRIC eval_reward={r['eval_reward']:.4f}")
print(f"METRIC catch_rate={r['catch_rate']:.4f}")
print(f"METRIC train_reward={val:.4f}")
print(f"METRIC gap={r['eval_reward'] - val:.4f}")
print(f"METRIC hist_last_best={hist[-1]['best']:.4f}")
print(f"METRIC train_seconds={time.time() - t0:.1f}")
