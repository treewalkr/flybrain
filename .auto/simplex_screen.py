"""Ensemble weight-simplex search over the refinement ladder (zero training).

Members share one feature basis, so mu-weighting is exact score ensembling.
Two-stage validation selection (anti-val-overfit):
  stage 1: ~150 Dirichlet(1) weight vectors + seeds (uniform, ens_1 weights),
           scored on VAL 8500..8547 (48 episodes);
  stage 2: top 20 rescored on VAL 8500..8595 (96 episodes);
  gate:    final score must beat ens_1's +15.1667 to spend ONE frozen-eval read.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np                                            # noqa: E402
from flybrain.brain import BrainModel as BatchedBrain         # noqa: E402
from flybrain.eval_util import eval_run                       # noqa: E402

CFG = json.load(open(".auto/config.json"))
MEMBERS = ["rf3_84", "rf3_83", "rf2_71", "rf4_98", "rf4_93"]
GATE = 15.1667                     # ens_1 validation score
VAL1 = list(range(8500, 8548))
VAL2 = list(range(8500, 8596))

graph = dict(np.load(CFG["graph"], allow_pickle=True))
cfg = json.load(open("data/runs/rf3_84/config.json"))
cfg["feat_idx_path"] = "data/runs/rf3_84/feat_idx.npy"
brain = BatchedBrain(graph, dt=0.005, gain=cfg["gain"])

basis = None
mus = []
for n in MEMBERS:
    fi = np.load(f"data/runs/{n}/feat_idx.npy")
    assert basis is None or np.array_equal(fi, basis), f"{n} basis differs"
    basis = fi
    mus.append(np.load(f"data/runs/{n}/mu.npy"))
mus = np.stack(mus)                                          # (M, K)
M = len(MEMBERS)
print("members share feature basis:", ", ".join(MEMBERS), flush=True)


def score(w, seeds):
    mu = (np.asarray(w, np.float64) @ mus).astype(np.float32)
    return eval_run(brain, mu, cfg, seeds=seeds, batch=48)["eval_reward"]


t0 = time.time()
rng = np.random.default_rng(0)
cands = [np.full(M, 1 / M), np.array([0.5, 0.3, 0.2, 0.0, 0.0])]   # uniform, ens_1
cands += list(rng.dirichlet(np.ones(M), size=148))
scores = [score(w, VAL1) for w in cands]
order = np.argsort(scores)[::-1][:20]
print(f"stage 1: best {scores[order[0]]:+.2f} @ {np.round(cands[order[0]], 2)}", flush=True)

best = (None, GATE)
for i in order:
    s2 = score(cands[i], VAL2)
    if s2 > best[1]:
        best = (cands[i], s2)
print(f"stage 2 best: {best[1]:+.4f} @ {np.round(best[0], 3) if best[0] is not None else None} "
      f"(gate {GATE})", flush=True)

if best[0] is None:
    print(f"gate: best <= {GATE} - no eval spend", flush=True)
    print(f"METRIC eval_reward={best[1]:.4f}")
    print("METRIC catch_rate=0 train_reward=0 gap=0 hist_last_best=0")
    print(f"METRIC train_seconds={time.time() - t0:.1f}")
    sys.exit(0)

w = np.asarray(best[0], np.float32)
mu = (w.astype(np.float64) @ mus).astype(np.float32)
out = Path("data/runs/ens_2")
out.mkdir(exist_ok=True)
np.save(out / "mu.npy", mu)
np.save(out / "feat_idx.npy", basis)
np.save(out / "sigma.npy", np.load("data/runs/rf3_84/sigma.npy"))
json.dump({**{k: v for k, v in cfg.items() if k != "feat_idx_path"},
           "ensemble": dict(zip(MEMBERS, [float(x) for x in w]))},
          open(out / "config.json", "w"), indent=1)

print(f"gate passed -> frozen eval (saved {out})", flush=True)
r = eval_run(brain, mu, cfg)
print(f"METRIC eval_reward={r['eval_reward']:.4f}")
print(f"METRIC catch_rate={r['catch_rate']:.4f}")
print(f"METRIC train_reward={best[1]:.4f}")
print(f"METRIC gap={r['eval_reward'] - best[1]:.4f}")
print("METRIC hist_last_best=0")
print(f"METRIC train_seconds={time.time() - t0:.1f}")
print(f"METRIC act_l={r['act_l']:.3f} act_s={r['act_s']:.3f} act_r={r['act_r']:.3f}")
