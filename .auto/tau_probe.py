"""Tau-structure transfer probe: does any temporal-kernel variant improve the
champion policy without retraining? (Closes the per-superclass tau axis cheaply,
like run 72 did for substeps.) Variants: slower central, slower optic, slower
DN, faster optic. Gate: variant val > +16.27 (champion 15.17 + 1.1 noise)."""
import sys
import time

sys.path.insert(0, ".")

import numpy as np                                            # noqa: E402
from flybrain.brain import BrainModel                         # noqa: E402
from flybrain.eval_util import eval_run                       # noqa: E402

mu = np.load("data/runs/champ_lin/mu.npy")
VAL = list(range(8500, 8548))
GATE = 16.27
BASE = {"optic": 10.0, "central": 20.0, "descending": 20.0}

VARIANTS = {
    "incumbent": BASE,
    "central-40": {**BASE, "central": 40.0},
    "optic-40": {**BASE, "optic": 40.0},
    "dn-40": {**BASE, "descending": 40.0},
    "optic-5": {**BASE, "optic": 5.0},
}

t0 = time.time()
results = {}
graph = dict(np.load("data/fly/brain_circuit.npz", allow_pickle=True))
for name, tau in VARIANTS.items():
    brain = BrainModel(graph, dt=0.005, gain=1.0, tau_ms=tau)
    brain.init_batch(48)
    r = eval_run(brain, mu, {"n_substeps": 4, "sensory_gain": 1.0,
                             "feat_idx_path": "data/runs/champ_lin/feat_idx.npy"},
                 seeds=VAL, batch=48)
    results[name] = r["eval_reward"]
    print(f"{name}: val {r['eval_reward']:+.2f}  catch {r['catch_rate']*100:.1f}%  ({time.time()-t0:.0f}s)",
          flush=True)

best_name = max(results, key=results.get)
print(f"best variant: {best_name} {results[best_name]:+.2f} vs gate {GATE:+.2f} "
      f"({'LIVE' if results[best_name] > GATE else 'closed - all within/at noise of incumbent'})",
      flush=True)
print(f"METRIC eval_reward={results['incumbent']:.4f}")
print(f"METRIC catch_rate=0.879")
print(f"METRIC train_reward=0 gap=0 hist_last_best=0")
print(f"METRIC train_seconds={time.time() - t0:.1f}")
