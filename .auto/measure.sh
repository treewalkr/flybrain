#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Fast prechecks
.venv/bin/python -m py_compile flybrain/*.py

# Train on the training stream (fixed budget), then evaluate on frozen eval seeds.
# Emits METRIC lines parsed by run_experiment.
RUN=$(.venv/bin/python - <<'PY'
import time, json
from pathlib import Path
import numpy as np
from flybrain.train import train
from flybrain.brain import BrainModel as BatchedBrain
from flybrain.eval_util import eval_run

cfg = json.loads(Path('.auto/config.json').read_text()) if Path('.auto/config.json').exists() else {}
t0 = time.time()
ENS = cfg.get('ensemble', 1)
mus, hists = [], []
for k in range(ENS):
    mu_k, sigma, hist = train(
        iters=cfg.get('iters', 24), pop=cfg.get('pop', 48), elites=cfg.get('elites', 8),
        eps=cfg.get('eps', 1), seed=cfg.get('seed', 0) + 100 * k, use_sensory=cfg.get('use_sensory', True),
        sensory_gain=cfg.get('sensory_gain', 1.0), n_substeps=cfg.get('n_substeps', 4),
        gain=cfg.get('gain', 1.0), dn_topk=cfg.get('dn_topk'), quiet=True,
        sigma_decay=cfg.get('sigma_decay', 0.9), sigma_floor=cfg.get('sigma_floor', 0.02),
        train_balls=cfg.get("train_balls", 20), workers=int(cfg.get("workers", 0)), elitism=cfg.get("elitism", False),
        out=Path(f'data/runs/measure_{k}'))
    mus.append(mu_k); hists.append(hist)
mu = np.mean(mus, axis=0).astype(np.float32)
hist = hists[0]
train_s = time.time() - t0

# training-seed reference (last generation's elite fitness, already measured)
train_fit = hist[-1]['elite_mean'] if hist else 0.0

# held-out evaluation on FROZEN eval seeds
import numpy as np
from flybrain.brain import BrainModel as BatchedBrain
from flybrain.eval_util import eval_run
cfg2 = dict(cfg)
cfg2['feat_idx_path'] = 'data/runs/measure_0/feat_idx.npy'
tgraph = dict(np.load(cfg.get('graph', 'data/fly/brain_circuit.npz'), allow_pickle=True))
brain = BatchedBrain(tgraph, dt=0.005, gain=cfg.get('gain', 1.0))
stats = eval_run(brain, mu, cfg2)
print(f"METRIC eval_reward={stats['eval_reward']:.4f}")
print(f"METRIC catch_rate={stats['catch_rate']:.4f}")
print(f"METRIC train_reward={train_fit:.4f}")
print(f"METRIC gap={stats['eval_reward'] - train_fit:.4f}")
print(f"METRIC train_seconds={train_s:.1f}")
print(f"METRIC hist_last_best={hist[-1]['best'] if hist else 0:.4f}")
print(f"METRIC act_l={stats['act_l']:.3f}")
print(f"METRIC act_s={stats['act_s']:.3f}")
print(f"METRIC act_r={stats['act_r']:.3f}")
PY
)
echo "$RUN"
