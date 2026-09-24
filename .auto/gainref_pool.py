"""Champion-anchored gain refinement: does the gain-family lift the ceiling
NEAR an optimum?

mu_init = [champion readout, ones(300)] -> initial behavior is EXACTLY the
champion (gains=1 identity). CEM refines readout and gains jointly. Gate +15.17
(champion val). If draws land below init like runs 73/77, the richer family is
locally converged too and the input-gain axis is closed at both regimes.
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
CHAMP = Path("data/runs/champ_lin")
GATE = 15.17
VAL_SEEDS = list(range(8500, 8548))

t0 = time.time()
mu_c = np.load(CHAMP / "mu.npy")
feat = np.load(CHAMP / "feat_idx.npy")
mu0 = np.concatenate([mu_c, np.ones(300, np.float32)])
graph = dict(np.load(CFG["graph"], allow_pickle=True))
brain = BatchedBrain(graph, dt=0.005, gain=1.0)
brain.init_batch(1)

# sanity: init == champion
np.save(CHAMP / "gains_one.npy", np.ones(300, np.float32))
v0 = eval_run(brain, mu_c, {"n_substeps": 4, "sensory_gain": 1.0,
                            "feat_idx_path": CHAMP / "feat_idx.npy",
                            "gains_path": CHAMP / "gains_one.npy"},
              seeds=VAL_SEEDS[:12], batch=12)
vc = eval_run(brain, mu_c, {"n_substeps": 4, "sensory_gain": 1.0,
                            "feat_idx_path": CHAMP / "feat_idx.npy"},
              seeds=VAL_SEEDS[:12], batch=12)
print(f"init sanity: with-ones-gains {v0['eval_reward']:+.2f} vs champion {vc['eval_reward']:+.2f}",
      flush=True)
assert abs(v0["eval_reward"] - vc["eval_reward"]) < 1e-9

best = None
for s, sig in ((161, 0.15), (162, 0.12)):
    out = Path(f"data/runs/gainref_{s}")
    if not (out / "mu.npy").exists():
        train(seed=s, out=out, workers=6, graph_path=CFG["graph"],
              iters=24, pop=64, elites=10, eps=3, use_sensory=True, sensory_gain=1.0,
              n_substeps=4, gain=1.0, dn_topk=256, sigma_decay=0.9, sigma_floor=0.02,
              train_balls=12, quiet=True, mu_init=mu0, sigma_init=sig,
              feat_idx_override=feat, train_gains=True)
    c = {"n_substeps": 4, "sensory_gain": 1.0,
         "feat_idx_path": str(out / "feat_idx.npy"), "gains_path": str(out / "gains.npy")}
    v = eval_run(brain, np.load(out / "mu.npy"), c, seeds=VAL_SEEDS, batch=48)
    hist = json.load(open(out / "history.json"))
    g = np.load(out / "gains.npy")
    print(f"gainref {s} (sig {sig}): val {v['eval_reward']:+.2f} catch {v['catch_rate']*100:.0f}% "
          f"gains[{g.min():.2f},{g.max():.2f}] lastbest {hist[-1]['best']:+.1f} ({time.time()-t0:.0f}s)",
          flush=True)
    if best is None or v["eval_reward"] > best[1]:
        best = (s, v["eval_reward"], out)

s, val, out = best
hist = json.load(open(out / "history.json"))
if val <= GATE:
    print(f"gate: {val:+.2f} <= {GATE} - no eval spend; gain family locally converged too", flush=True)
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
