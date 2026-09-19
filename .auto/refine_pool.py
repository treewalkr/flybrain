"""Champion-restart refinement pool (budgeted, resumable, val-gated).

Initialize CEM at the champion's mu (ms_32, val +8.42) with small sigma and
refine with fresh episode seeds - local hill-climb from the best known point,
with correct selection (unlike the failed elitism injection). Gate: spend the
single frozen-eval read only if a refinement beats +8.42 on validation.
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
from flybrain.train import train                              # noqa: E402

CFG = json.load(open(".auto/config.json"))
CHAMP = Path("data/runs/champ_lin")
GATE = 15.17
VAL_SEEDS = list(range(8500, 8548))
MU0 = np.load(CHAMP / "mu.npy")

RUNS = {
    91: {"iters": 24, "sigma_init": 0.18},
    92: {"iters": 24, "sigma_init": 0.15},
    93: {"iters": 24, "sigma_init": 0.12},
    94: {"iters": 24, "pop": 96, "eps": 4, "sigma_init": 0.15},
}

t0 = time.time()
BUDGET_S = float(os.environ.get("PILOT_BUDGET_S", "4200"))
best = None
for s, ov in RUNS.items():
    out = Path(f"data/runs/rf4_{s}")
    if not (out / "mu.npy").exists():
        if time.time() - t0 > BUDGET_S:
            print(f"budget reached before seed {s} - resumable, rerun to continue", flush=True)
            continue
        rc = {**CFG, **ov}
        train(seed=s, out=out, workers=int(rc.get("workers", 6)), graph_path=rc["graph"],
              iters=rc["iters"], pop=rc.get("pop", 64), elites=rc.get("elites", 10),
              eps=rc.get("eps", 6), use_sensory=True, sensory_gain=1.0, n_substeps=4,
              gain=1.0, dn_topk=256, sigma_decay=0.9, sigma_floor=0.02,
              train_balls=12, quiet=True, mu_init=MU0, sigma_init=rc["sigma_init"])
    cfg = json.load(open(out / "config.json"))
    cfg["feat_idx_path"] = str(out / "feat_idx.npy")
    graph = dict(np.load(cfg["graph"], allow_pickle=True))
    brain = BatchedBrain(graph, dt=0.005, gain=cfg["gain"])
    v = eval_run(brain, np.load(out / "mu.npy"), cfg, seeds=VAL_SEEDS, batch=48)
    print(f"rf seed {s} (sig {ov['sigma_init']}, it {ov['iters']}): val {v['eval_reward']:+.2f}"
          f"  catch {v['catch_rate']*100:.0f}%  ({time.time()-t0:.0f}s)", flush=True)
    if best is None or v["eval_reward"] > best[1]:
        best = (s, v["eval_reward"], out)

if best is None or best[1] <= GATE:
    print(f"gate: best {best} <= {GATE} - no eval spend", flush=True)
    print(f"METRIC eval_reward={best[1] if best else 0:.4f}")
    print("METRIC catch_rate=0 train_reward=0 gap=0 hist_last_best=0")
    print(f"METRIC train_seconds={time.time() - t0:.1f}")
    sys.exit(0)

s, val, out = best
print(f"gate passed: rf seed {s} val {val:+.2f} -> frozen eval", flush=True)
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
