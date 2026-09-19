"""Validation-noise audit (zero training, load-tolerant).

Every gate decision compared candidate val scores against incumbents. This
measures the noise floor of a 48-episode val score: champion scored on three
disjoint blocks (two used historically for selection, one fresh, never used
for anything). Reports mean +/- spread - the yardstick for how much a
"below gate" margin actually means.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np                                            # noqa: E402
from flybrain.brain import BrainModel as BatchedBrain         # noqa: E402
from flybrain.eval_util import eval_run                       # noqa: E402

t0 = time.time()
graph = dict(np.load("data/fly/brain_circuit.npz", allow_pickle=True))
brain = BatchedBrain(graph, dt=0.005)

d = Path("data/runs/champ_lin2")
cfg = json.load(open(d / "config.json"))
cfg["feat_idx_path"] = str(d / "feat_idx.npy")
mu = np.load(d / "mu.npy")

blocks = {
    "val A 8500-8523": list(range(8500, 8524)),
    "val B 8524-8547": list(range(8524, 8548)),
    "fresh  8548-8595": list(range(8548, 8596)),   # never used: not selection, not eval
}
scores = {}
for tag, seeds in blocks.items():
    v = eval_run(brain, mu, cfg, seeds=seeds, batch=48)
    scores[tag] = v["eval_reward"]
    print(f"{tag}: {v['eval_reward']:+6.2f}  catch {v['catch_rate']*100:.1f}%", flush=True)

vals = np.array(list(scores.values()))
print(f"spread {vals.min():+.2f}..{vals.max():+.2f}  half-range {(vals.max()-vals.min())/2:.2f}", flush=True)

# frozen-eval number for the ledger
r = eval_run(brain, mu, cfg)
print(f"METRIC eval_reward={r['eval_reward']:.4f}")
print(f"METRIC catch_rate={r['catch_rate']:.4f}")
print("METRIC train_reward=15.5833")
print(f"METRIC gap={r['eval_reward'] - 15.5833:.4f}")
print("METRIC hist_last_best=0")
print(f"METRIC train_seconds={time.time() - t0:.1f}")
print(f"METRIC act_l={r['act_l']:.3f} act_s={r['act_s']:.3f} act_r={r['act_r']:.3f}")
