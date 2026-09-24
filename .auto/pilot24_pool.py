"""Retina-24 pilot: precision WITHOUT the sensory bottleneck.

Diagnostic (.auto/landing_probe.py): 52% of champion misses are marginal
(landing 0.10-0.15 from paddle, window 0.10) and the policy reaches within
0.05 of the true landing for 91% of balls - input quantization is a real
binding constraint. Retina-32 failed on dilution (387 obs dims -> 300 sensory
neurons; 6 fresh draws all landed 38-53% catch). Retina-24 gives obs_dim
291 <= 300 sensory neurons (no squeeze) with quantization error +/-0.021
vs +/-0.031, at density-preserving degree 6 (readers/dim 6.2 == retina-16).

Protocol: 4 fresh runs scored on validation only; the single frozen-eval
confirmation is spent ONLY if the best val beats the from-scratch gate +8.42.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["FLY_RETINA_W"] = "24"
os.environ["FLY_PROJ_DEGREE"] = "6"
sys.path.insert(0, ".")

import numpy as np                                            # noqa: E402
from flybrain.brain import BrainModel as BatchedBrain         # noqa: E402
from flybrain.eval_util import eval_run                       # noqa: E402
from flybrain.game import OBS_DIM, RETINA_W                  # noqa: E402
from flybrain.train import train                             # noqa: E402

CFG = json.load(open(".auto/config.json"))
RUNS = {61: {}, 62: {}, 63: {}, 64: {"pop": 96, "eps": 4}}
VAL_SEEDS = list(range(8500, 8548))
GATE = 8.42

print(f"retina {RETINA_W}x12, OBS_DIM {OBS_DIM}, degree 6", flush=True)
t0 = time.time()
BUDGET_S = float(os.environ.get("PILOT_BUDGET_S", "3600"))
best = None
for s, ov in RUNS.items():
    out = Path(f"data/runs/r24_{s}")
    if not (out / "mu.npy").exists():
        if time.time() - t0 > BUDGET_S:
            print(f"budget reached before seed {s} - resumable, rerun to continue", flush=True)
            continue
        rc = {**CFG, **ov}
        train(seed=s, out=out, workers=int(rc.get("workers", 6)), graph_path=rc["graph"],
              iters=rc.get("iters", 96), pop=rc.get("pop", 64), elites=rc.get("elites", 10),
              eps=rc.get("eps", 6), use_sensory=True, sensory_gain=1.0, n_substeps=4,
              gain=1.0, dn_topk=rc.get("dn_topk", 256), sigma_decay=0.9, sigma_floor=0.02,
              train_balls=12, quiet=True)
    cfg = json.load(open(out / "config.json"))
    cfg["feat_idx_path"] = str(out / "feat_idx.npy")
    graph = dict(np.load(cfg["graph"], allow_pickle=True))
    brain = BatchedBrain(graph, dt=0.005, gain=cfg["gain"])
    v = eval_run(brain, np.load(out / "mu.npy"), cfg, seeds=VAL_SEEDS, batch=48)
    print(f"r24 seed {s}: val {v['eval_reward']:+.2f}  catch {v['catch_rate']*100:.0f}%  ({time.time()-t0:.0f}s)", flush=True)
    if best is None or v["eval_reward"] > best[1]:
        best = (s, v["eval_reward"], out)

if best is None:
    print("no artifacts", flush=True)
    sys.exit(1)

s, val, out = best
if val <= GATE:
    print(f"gate: best val {val:+.2f} <= {GATE:+.2f} - no eval spend; axis closed", flush=True)
    print(f"METRIC eval_reward={val:.4f}")
    print("METRIC catch_rate=0 train_reward=0 gap=0 hist_last_best=0")
    print(f"METRIC train_seconds={time.time() - t0:.1f}")
    sys.exit(0)

print(f"gate passed: r24 seed {s} val {val:+.2f} > {GATE:+.2f} -> frozen eval", flush=True)
cfg = json.load(open(out / "config.json"))
cfg["feat_idx_path"] = str(out / "feat_idx.npy")
graph = dict(np.load(cfg["graph"], allow_pickle=True))
brain = BatchedBrain(graph, dt=0.005, gain=cfg["gain"])
r = eval_run(brain, np.load(out / "mu.npy"), cfg)
print(f"METRIC eval_reward={r['eval_reward']:.4f}")
print(f"METRIC catch_rate={r['catch_rate']:.4f}")
print(f"METRIC train_reward={val:.4f}")
print(f"METRIC gap={r['eval_reward'] - val:.4f}")
print(f"METRIC hist_last_best={json.load(open(out / 'history.json'))[-1]['best']:.4f}")
print(f"METRIC train_seconds={time.time() - t0:.1f}")
print(f"METRIC act_l={r['act_l']:.3f} act_s={r['act_s']:.3f} act_r={r['act_r']:.3f}")
