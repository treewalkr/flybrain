"""Projection-lottery pool: does the W_s seed move the from-scratch ceiling?

Pre-check: DN landing-info top-256 mean |r_land| varies 0.150-0.263 across
projection seeds (seed 7 = 0.224 incumbent, seed 3 = 0.263 richest, seed 5 =
0.150 poorest; some seeds leave obs dims with ZERO readers). Run 78 showed
readout-side feature info is ceiling-neutral; the W_s seed instead changes what
the CONNECTOME receives - a different regime. 2 fresh draws each under proj
seeds 3 and 5; seed-7 reference: best-of-14 +8.42 (single draws scatter 0-8).
Frozen-eval spend only above the champion gate +15.17.
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
RUNS = {191: 3, 193: 3, 192: 5, 194: 5}                      # train seed -> proj seed
CHAMP_GATE = 15.17
VAL_SEEDS = list(range(8500, 8548))

t0 = time.time()
graph = dict(np.load(CFG["graph"], allow_pickle=True))
brain = BatchedBrain(graph, dt=0.005, gain=1.0)
brain.init_batch(1)

results = {}
BUDGET_S = float(__import__("os").environ.get("PILOT_BUDGET_S", "2400"))
best = None
for s, ps in RUNS.items():
    out = Path(f"data/runs/proj_{s}")
    if not (out / "mu.npy").exists():
        if time.time() - t0 > BUDGET_S:
            print(f"budget reached before seed {s} - resumable", flush=True)
            continue
        train(seed=s, out=out, workers=6, graph_path=CFG["graph"],
              iters=96, pop=64, elites=10, eps=6, use_sensory=True, sensory_gain=1.0,
              n_substeps=4, gain=1.0, dn_topk=256, sigma_decay=0.9, sigma_floor=0.02,
              train_balls=12, quiet=True, proj_seed=ps)
    c = {"n_substeps": 4, "sensory_gain": 1.0, "proj_seed": ps,
         "feat_idx_path": str(out / "feat_idx.npy")}
    v = eval_run(brain, np.load(out / "mu.npy"), c, seeds=VAL_SEEDS, batch=48)
    results.setdefault(ps, []).append(v["eval_reward"])
    hist = json.load(open(out / "history.json"))
    print(f"train {s} @ proj {ps}: val {v['eval_reward']:+.2f} catch {v['catch_rate']*100:.0f}% "
          f"({time.time()-t0:.0f}s)", flush=True)
    if best is None or v["eval_reward"] > best[1]:
        best = (s, ps, v["eval_reward"], out)

for ps, vals in sorted(results.items()):
    print(f"proj seed {ps}: draws {[f'{x:+.2f}' for x in vals]}  best {max(vals):+.2f}", flush=True)

s, ps, val, out = best
hist = json.load(open(out / "history.json"))
if val <= CHAMP_GATE:
    top = max(max(v) for v in results.values())
    print(f"no eval spend (champion gate {CHAMP_GATE:+.2f}); best across proj seeds {top:+.2f}", flush=True)
    print(f"METRIC eval_reward={top:.4f}")
    print(f"METRIC catch_rate=0 train_reward={top:.4f} gap=0 "
          f"hist_last_best={hist[-1]['best']:.4f} train_seconds={time.time()-t0:.1f}")
    sys.exit(0)

c = {"n_substeps": 4, "sensory_gain": 1.0, "proj_seed": ps,
     "feat_idx_path": str(out / "feat_idx.npy")}
r = eval_run(brain, np.load(out / "mu.npy"), c)
print(f"METRIC eval_reward={r['eval_reward']:.4f}")
print(f"METRIC catch_rate={r['catch_rate']:.4f}")
print(f"METRIC train_reward={val:.4f}")
print(f"METRIC gap={r['eval_reward'] - val:.4f}")
print(f"METRIC hist_last_best={hist[-1]['best']:.4f}")
print(f"METRIC train_seconds={time.time() - t0:.1f}")
