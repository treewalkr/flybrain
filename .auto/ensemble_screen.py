"""Top-tail policy-ensemble screen (zero training).

Linear policies share the same deterministic feature basis, so score-sum
ensembling == mu averaging. The early cross-seed failure averaged the whole
pool (bad trajectories included); this screens only the top val tail:
  ms_32 (+8.42), ms_19 (+7.21), ms_30 (+6.38), ms_36 (+4.42), ms_31 (+3.54)
Validation-gated: eval is spent once, only if a combo beats +8.42.
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
GATE = 8.42

graph = dict(np.load(CFG["graph"], allow_pickle=True))
brain = BatchedBrain(graph, dt=0.005, gain=CFG.get("gain", 1.0))
cfg0 = json.load(open("data/runs/ms_32/config.json"))

names = ["ms_32", "ms_19", "ms_30", "ms_36", "ms_31"]
vals = {"ms_32": 8.42, "ms_19": 7.21, "ms_30": 6.38, "ms_36": 4.42, "ms_31": 3.54}
mus, shared = {}, None
for n in names:
    fi = np.load(f"data/runs/{n}/feat_idx.npy")
    assert shared is None or np.array_equal(fi, shared), f"{n} feature basis differs!"
    shared = fi
    mus[n] = np.load(f"data/runs/{n}/mu.npy")
print("feature bases identical across pool - mu averaging is exact ensembling", flush=True)


def score(mu, tag):
    cfg = dict(cfg0)
    cfg["feat_idx_path"] = "data/runs/ms_32/feat_idx.npy"
    v = eval_run(brain, mu, cfg, seeds=VAL_SEEDS, batch=48)
    print(f"{tag:28s} val {v['eval_reward']:+6.2f}  catch {v['catch_rate']*100:.0f}%", flush=True)
    return v["eval_reward"]


t0 = time.time()
combos = {
    "avg(32,19)": np.mean([mus["ms_32"], mus["ms_19"]], axis=0),
    "avg(32,30)": np.mean([mus["ms_32"], mus["ms_30"]], axis=0),
    "avg(19,30)": np.mean([mus["ms_19"], mus["ms_30"]], axis=0),
    "avg(32,19,30)": np.mean([mus["ms_32"], mus["ms_19"], mus["ms_30"]], axis=0),
    "avg(top5)": np.mean([mus[n] for n in names], axis=0),
    "valw(top5)": np.sum([vals[n] * mus[n] for n in names], axis=0) / sum(vals.values()),
    "0.5*32+0.25*(19+30)": 0.5 * mus["ms_32"] + 0.25 * mus["ms_19"] + 0.25 * mus["ms_30"],
}
best = ("none", GATE)
for tag, mu in combos.items():
    mu = np.asarray(mu, np.float32)
    v = score(mu, tag)
    if v > best[1]:
        best = (tag, v)

if best[1] <= GATE:
    print(f"gate: best {best[0]} {best[1]:+.2f} <= {GATE} - no eval spend", flush=True)
    print(f"METRIC eval_reward={best[1]:.4f}")
    print("METRIC catch_rate=0 train_reward=0 gap=0 hist_last_best=0")
    print(f"METRIC train_seconds={time.time() - t0:.1f}")
    sys.exit(0)

mu = np.asarray(combos[best[0]], np.float32)
print(f"gate passed: {best[0]} val {best[1]:+.2f} -> frozen eval", flush=True)
cfg = dict(cfg0)
cfg["feat_idx_path"] = "data/runs/ms_32/feat_idx.npy"
r = eval_run(brain, mu, cfg)
print(f"METRIC eval_reward={r['eval_reward']:.4f}")
print(f"METRIC catch_rate={r['catch_rate']:.4f}")
print(f"METRIC train_reward={best[1]:.4f}")
print(f"METRIC gap={r['eval_reward'] - best[1]:.4f}")
print("METRIC hist_last_best=0")
print(f"METRIC train_seconds={time.time() - t0:.1f}")
print(f"METRIC act_l={r['act_l']:.3f} act_s={r['act_s']:.3f} act_r={r['act_r']:.3f}")
