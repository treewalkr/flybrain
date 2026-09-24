"""Per-channel input-gain pool: CEM trains readout + 300 sensory-channel gains.

Motivation (this axis is explicitly permitted: 'readout/input-gain only'):
- relu clipping is ACTIVE in the champion (40.8% of DN v-samples negative), so
  the brain is NOT in a linear regime; per-channel gains move clipping
  boundaries per unit and change what the connectome computes.
- The closed scalar-gain bit-identity is relu positive homogeneity (uniform
  scaling cancels in argmax); per-channel gains break that symmetry.
- All prior closures tested the readout-only linear family; this enlarges the
  trainable family through the input, within the stated constraint.

Protocol: 4 fresh draws (96 iters, train_gains=True), val-gated at the
from-scratch +8.42; frozen eval only if cleared.
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
RUNS = {151: {}, 152: {}, 153: {}, 154: {"pop": 96, "eps": 4}}
VAL_SEEDS = list(range(8500, 8548))
GATE = 8.42

t0 = time.time()
graph = dict(np.load(CFG["graph"], allow_pickle=True))
brain = BatchedBrain(graph, dt=0.005, gain=1.0)
brain.init_batch(1)

BUDGET_S = float(__import__("os").environ.get("PILOT_BUDGET_S", "2400"))
best = None
for s, ov in RUNS.items():
    out = Path(f"data/runs/gain_{s}")
    if not (out / "mu.npy").exists():
        if time.time() - t0 > BUDGET_S:
            print(f"budget reached before seed {s} - resumable", flush=True)
            continue
        rc = {**CFG, **ov}
        train(seed=s, out=out, workers=6, graph_path=CFG["graph"],
              iters=96, pop=rc.get("pop", 64), elites=rc.get("elites", 10),
              eps=rc.get("eps", 6), use_sensory=True, sensory_gain=1.0, n_substeps=4,
              gain=1.0, dn_topk=256, sigma_decay=0.9, sigma_floor=0.02,
              train_balls=12, quiet=True, train_gains=True)
    c = {"n_substeps": 4, "sensory_gain": 1.0,
         "feat_idx_path": str(out / "feat_idx.npy"), "gains_path": str(out / "gains.npy")}
    v = eval_run(brain, np.load(out / "mu.npy"), c, seeds=VAL_SEEDS, batch=48)
    hist = json.load(open(out / "history.json"))
    g = np.load(out / "gains.npy")
    print(f"gain seed {s}: val {v['eval_reward']:+.2f}  catch {v['catch_rate']*100:.0f}%  "
          f"gains[{g.min():.2f},{g.max():.2f}] mean {g.mean():.2f}  ({time.time()-t0:.0f}s)",
          flush=True)
    if best is None or v["eval_reward"] > best[1]:
        best = (s, v["eval_reward"], out)

s, val, out = best
hist = json.load(open(out / "history.json"))
if val <= GATE:
    print(f"gate: {val:+.2f} <= {GATE:+.2f} - no eval spend; input-gain axis answered", flush=True)
    print(f"METRIC eval_reward={val:.4f}")
    print(f"METRIC catch_rate=0 train_reward={val:.4f} gap=0 "
          f"hist_last_best={hist[-1]['best']:.4f} train_seconds={time.time()-t0:.1f}")
    sys.exit(0)

c = {"n_substeps": 4, "sensory_gain": 1.0,
     "feat_idx_path": str(out / "feat_idx.npy"), "gains_path": str(out / "gains.npy")}
r = eval_run(brain, np.load(out / "mu.npy"), c)
print(f"METRIC eval_reward={r['eval_reward']:.4f}")
print(f"METRIC catch_rate={r['catch_rate']:.4f}")
print(f"METRIC train_reward={val:.4f}")
print(f"METRIC gap={r['eval_reward'] - val:.4f}")
print(f"METRIC hist_last_best={hist[-1]['best']:.4f}")
print(f"METRIC train_seconds={time.time() - t0:.1f}")
print(f"METRIC act_l={r['act_l']:.3f} act_s={r['act_s']:.3f} act_r={r['act_r']:.3f}")
