"""Lineage-ensemble screen (zero training, load-tolerant).

Run 47 closed averaging over INDEPENDENT trajectories. The refinement ladder
(ms_32 -> rf_63 -> rf2_71 -> rf3_84) is a CORRELATED lineage - each restart
began at the previous champion's mu. Tail-averaging within a run already
works; this tests it across the ladder. Gate: beat rf3_84's val +14.96.
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
VAL_SEEDS = list(range(8500, 8548))
GATE = 14.96

graph = dict(np.load(CFG["graph"], allow_pickle=True))
brain = BatchedBrain(graph, dt=0.005, gain=CFG.get("gain", 1.0))
cfg0 = json.load(open("data/runs/rf3_84/config.json"))

lineage = ["ms_32", "rf_63", "rf2_71", "rf3_84"]              # oldest -> newest
vals = {"ms_32": 8.42, "rf_63": 11.46, "rf2_71": 13.50, "rf3_84": 14.96}
mus, shared = {}, None
for n in lineage:
    fi = np.load(f"data/runs/{n}/feat_idx.npy")
    assert shared is None or np.array_equal(fi, shared), f"{n} basis differs"
    shared = fi
    mus[n] = np.load(f"data/runs/{n}/mu.npy")
print("ladder shares one feature basis - exact ensembling", flush=True)


def score(mu, tag):
    cfg = dict(cfg0)
    cfg["feat_idx_path"] = "data/runs/rf3_84/feat_idx.npy"
    v = eval_run(brain, np.asarray(mu, np.float32), cfg, seeds=VAL_SEEDS, batch=48)
    print(f"{tag:26s} val {v['eval_reward']:+6.2f}  catch {v['catch_rate']*100:.0f}%", flush=True)
    return v["eval_reward"]


t0 = time.time()
combos = {
    "avg(rf3,rf2)": 0.5 * (mus["rf3_84"] + mus["rf2_71"]),
    "avg(rf3,rf)": 0.5 * (mus["rf3_84"] + mus["rf_63"]),
    "avg(ladder4)": np.mean([mus[n] for n in lineage], axis=0),
    "avg(ladder3)": np.mean([mus[n] for n in lineage[1:]], axis=0),
    "w .5/.3/.2": 0.5 * mus["rf3_84"] + 0.3 * mus["rf2_71"] + 0.2 * mus["rf_63"],
    "w .4/.3/.2/.1": (0.4 * mus["rf3_84"] + 0.3 * mus["rf2_71"] + 0.2 * mus["rf_63"] + 0.1 * mus["ms_32"]),
    "w .34/.33/.33": (mus["rf3_84"] + mus["rf2_71"] + mus["rf_63"]) / 3,
}
best = ("none", GATE)
for tag, mu in combos.items():
    v = score(mu, tag)
    if v > best[1]:
        best = (tag, v)

if best[1] <= GATE:
    print(f"gate: best {best[0]} {best[1]:+.2f} <= {GATE} - no eval spend", flush=True)
    print(f"METRIC eval_reward={best[1]:.4f}")
    print("METRIC catch_rate=0 train_reward=0 gap=0 hist_last_best=0")
    print(f"METRIC train_seconds={time.time() - t0:.1f}")
    sys.exit(0)

mu = combos[best[0]]
print(f"gate passed: {best[0]} val {best[1]:+.2f} -> frozen eval", flush=True)
cfg = dict(cfg0)
cfg["feat_idx_path"] = "data/runs/rf3_84/feat_idx.npy"
r = eval_run(brain, np.asarray(mu, np.float32), cfg)
print(f"METRIC eval_reward={r['eval_reward']:.4f}")
print(f"METRIC catch_rate={r['catch_rate']:.4f}")
print(f"METRIC train_reward={best[1]:.4f}")
print(f"METRIC gap={r['eval_reward'] - best[1]:.4f}")
print("METRIC hist_last_best=0")
print(f"METRIC train_seconds={time.time() - t0:.1f}")
print(f"METRIC act_l={r['act_l']:.3f} act_s={r['act_s']:.3f} act_r={r['act_r']:.3f}")
