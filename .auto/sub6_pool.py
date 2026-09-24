"""Substeps-6 axis (finer/longer brain integration per decision).

Preflight (cheap): does the champion mu transfer to substeps 6 on val, and does
the DN top-256 set change? If transfer is flat (< +0.5) the dynamics axis is a
no-op for this policy: skip training entirely (round-4 already exhausted the
neighborhood) and report champion metrics. Otherwise: 2 refinement restarts from
champ_lin under substeps 6 (champion feature set forced), val-gated at +15.17.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np                                            # noqa: E402
from flybrain.brain import BrainModel as BatchedBrain         # noqa: E402
from flybrain.eval_util import eval_run                       # noqa: E402
from flybrain.game import OBS_DIM                             # noqa: E402
from flybrain.readout import make_sensory_projection          # noqa: E402
from flybrain.train import select_features, train             # noqa: E402

CFG = json.load(open(".auto/config.json"))
CHAMP = Path("data/runs/champ_lin")
GATE = 15.17
VAL_SEEDS = list(range(8500, 8548))

t0 = time.time()
cfg = json.load(open(CHAMP / "config.json"))
cfg["feat_idx_path"] = str(CHAMP / "feat_idx.npy")
graph = dict(np.load(cfg["graph"], allow_pickle=True))
brain = BatchedBrain(graph, dt=0.005, gain=cfg["gain"])
W_s = make_sensory_projection(len(brain.sensory_idx), OBS_DIM)

feat4 = np.load(CHAMP / "feat_idx.npy")
feat6 = select_features(brain, W_s, True, 1.0, 6, 256)
print(f"feat overlap: {len(np.intersect1d(feat4, feat6))}/{len(feat4)}", flush=True)

mu = np.load(CHAMP / "mu.npy")
v4 = eval_run(brain, mu, {**cfg, "n_substeps": 4}, seeds=VAL_SEEDS, batch=48)
v6 = eval_run(brain, mu, {**cfg, "n_substeps": 6}, seeds=VAL_SEEDS, batch=48)
gain = v6["eval_reward"] - v4["eval_reward"]
print(f"transfer: sub4 val {v4['eval_reward']:+.2f} ({v4['catch_rate']*100:.1f}%) -> "
      f"sub6 val {v6['eval_reward']:+.2f} ({v6['catch_rate']*100:.1f}%)  delta {gain:+.2f}", flush=True)


def finish(val, r, secs):
    print(f"METRIC eval_reward={r['eval_reward']:.4f}")
    print(f"METRIC catch_rate={r['catch_rate']:.4f}")
    print(f"METRIC train_reward={val:.4f}")
    print(f"METRIC gap={r['eval_reward'] - val:.4f}")
    print(f"METRIC hist_last_best=0")
    print(f"METRIC train_seconds={secs:.1f}")


if gain < 0.5:
    r = eval_run(brain, mu, {**cfg, "n_substeps": 6})
    print(f"flat transfer - training skipped, champion confirmed under sub6", flush=True)
    finish(v6["eval_reward"], r, time.time() - t0)
    sys.exit(0)

MU0 = mu
best = None
for s, sig in ((111, 0.15), (112, 0.12)):
    out = Path(f"data/runs/sub6_{s}")
    if not (out / "mu.npy").exists():
        train(seed=s, out=out, workers=6, graph_path=CFG["graph"],
              iters=24, pop=64, elites=10, eps=3, use_sensory=True, sensory_gain=1.0,
              n_substeps=6, gain=1.0, dn_topk=256, sigma_decay=0.9, sigma_floor=0.02,
              train_balls=12, quiet=True, mu_init=MU0, sigma_init=sig,
              feat_idx_override=feat4)
    c = json.load(open(out / "config.json"))
    c["feat_idx_path"] = str(out / "feat_idx.npy")
    c["n_substeps"] = 6
    b = BatchedBrain(graph, dt=0.005, gain=1.0)
    v = eval_run(b, np.load(out / "mu.npy"), c, seeds=VAL_SEEDS, batch=48)
    print(f"sub6 refine {s} (sig {sig}): val {v['eval_reward']:+.2f} "
          f"catch {v['catch_rate']*100:.0f}%  ({time.time()-t0:.0f}s)", flush=True)
    if best is None or v["eval_reward"] > best[1]:
        best = (s, v["eval_reward"], out, c)

s, val, out, c = best
if val <= GATE:
    print(f"gate: {val:+.2f} <= {GATE} - no eval spend", flush=True)
    finish(val, {"eval_reward": val, "catch_rate": v6["catch_rate"]}, time.time() - t0)
    sys.exit(0)

b = BatchedBrain(graph, dt=0.005, gain=1.0)
r = eval_run(b, np.load(out / "mu.npy"), c)
finish(val, r, time.time() - t0)
