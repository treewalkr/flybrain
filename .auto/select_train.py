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
SEEDS = [11, 12, 13, 14, 15, 16, 17, 18]
VAL_SEEDS = list(range(8500, 8548))

t0 = time.time()
graph = dict(np.load(CFG["graph"], allow_pickle=True))
scores = []
best = None
for s in SEEDS:
    out = Path(f"data/runs/ms_{s}")
    if not (out / "mu.npy").exists():          # resume: reuse trained artifacts
        train(seed=s, out=out, workers=int(CFG.get("workers", 0)), graph_path=CFG["graph"],
              iters=CFG["iters"], pop=CFG["pop"], elites=CFG["elites"], eps=CFG["eps"],
              use_sensory=CFG["use_sensory"], sensory_gain=CFG["sensory_gain"],
              n_substeps=CFG["n_substeps"], gain=CFG["gain"], dn_topk=CFG["dn_topk"],
              sigma_decay=CFG["sigma_decay"], sigma_floor=CFG["sigma_floor"],
              train_balls=CFG["train_balls"], quiet=True)
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
