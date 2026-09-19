"""Multi-seed training + validation-selected artifact (anti-trajectory-luck).

Protocol (overfit-safe):
  - train N seeds with the champion recipe;
  - score each artifact on VALIDATION seeds 8500..8547 — disjoint from both the
    CEM sampling stream [0, 8000) and the frozen EVAL_SEEDS [9000, 9096);
  - the argmax-on-validation artifact is confirmed ONCE on the frozen eval seeds.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from flybrain.brain import BrainModel as BatchedBrain          # noqa: E402
from flybrain.eval_util import eval_run                        # noqa: E402
from flybrain.train import train                               # noqa: E402

CFG = json.load(open(".auto/config.json"))
SEEDS = list(range(11, 31))
# recipe diversity: per-seed overrides over the champion recipe - fatten the top
# tail of the pool instead of drawing more seeds from the same distribution
RECIPES = {
    31: {"iters": 144, "eps": 4},          # deeper CEM
    32: {"pop": 96, "eps": 4},             # wider population
    33: {"elites": 16},                    # smoother elite statistics
    34: {"sigma_floor": 0.04},             # more late exploration
    35: {"iters": 192},                    # double depth
}
SEEDS += list(RECIPES)
VAL_SEEDS = list(range(8500, 8548))

t0 = time.time()
graph = dict(np.load(CFG["graph"], allow_pickle=True))
scores = []
best = None
for s in SEEDS:
    out = Path(f"data/runs/ms_{s}")
    if not (out / "mu.npy").exists():          # resume: reuse trained artifacts
        rc = {**CFG, **RECIPES.get(s, {})}
        train(seed=s, out=out, workers=int(rc.get("workers", 0)), graph_path=rc["graph"],
              iters=rc["iters"], pop=rc["pop"], elites=rc["elites"], eps=rc["eps"],
              use_sensory=rc["use_sensory"], sensory_gain=rc["sensory_gain"],
              n_substeps=rc["n_substeps"], gain=rc["gain"], dn_topk=rc["dn_topk"],
              sigma_decay=rc["sigma_decay"], sigma_floor=rc["sigma_floor"],
              train_balls=rc["train_balls"], quiet=True)
    mu = np.load(out / "mu.npy")
    cfg = json.load(open(out / "config.json"))
    cfg["feat_idx_path"] = str(out / "feat_idx.npy")
    brain = BatchedBrain(graph, dt=0.005, gain=cfg["gain"])
    v = eval_run(brain, mu, cfg, seeds=VAL_SEEDS, batch=48)
    scores.append(v["eval_reward"])
    print(f"seed {s}: val {v['eval_reward']:+.2f}  catch {v['catch_rate']*100:.0f}%  ({time.time()-t0:.0f}s)", flush=True)
    if best is None or v["eval_reward"] > best[1]:
        best = (s, v["eval_reward"])

win = Path(f"data/runs/ms_{best[0]}")
print(f"selected seed {best[0]} (val {best[1]:+.2f}) -> frozen-eval confirmation", flush=True)
cfg = json.load(open(win / "config.json"))
cfg["feat_idx_path"] = str(win / "feat_idx.npy")
brain = BatchedBrain(graph, dt=0.005, gain=cfg["gain"])
r = eval_run(brain, np.load(win / "mu.npy"), cfg)
print(f"METRIC eval_reward={r['eval_reward']:.4f}")
print(f"METRIC catch_rate={r['catch_rate']:.4f}")
print(f"METRIC train_reward={best[1]:.4f}")
print(f"METRIC gap={r['eval_reward'] - best[1]:.4f}")
print(f"METRIC hist_last_best={json.load(open(win / 'history.json'))[-1]['best']:.4f}")
print(f"METRIC train_seconds={time.time() - t0:.1f}")
print(f"METRIC act_l={r['act_l']:.3f} act_s={r['act_s']:.3f} act_r={r['act_r']:.3f}")
