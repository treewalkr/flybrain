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
from flybrain.brain import BatchedBrain
from flybrain.eval_util import eval_run

cfg = json.loads(Path('.auto/config.json').read_text()) if Path('.auto/config.json').exists() else {}
t0 = time.time()
mu, sigma, hist = train(
    iters=cfg.get('iters', 24), pop=cfg.get('pop', 48), elites=cfg.get('elites', 8),
    eps=cfg.get('eps', 1), seed=cfg.get('seed', 0), use_sensory=cfg.get('use_sensory', True),
    sensory_gain=cfg.get('sensory_gain', 1.0), n_substeps=cfg.get('n_substeps', 4),
    gain=cfg.get('gain', 1.0), quiet=True)
train_s = time.time() - t0

# training-seed reference (last generation's elite fitness, already measured)
train_fit = hist[-1]['elite_mean'] if hist else 0.0

# held-out evaluation on FROZEN eval seeds
tgraph = dict(np.load(cfg.get('graph', 'data/fly/brain_circuit.npz'), allow_pickle=True))
brain = BatchedBrain(tgraph, dt=0.005, gain=cfg.get('gain', 1.0))
stats = eval_run(brain, mu, {'use_sensory': cfg.get('use_sensory', True),
                             'sensory_gain': cfg.get('sensory_gain', 1.0),
                             'n_substeps': cfg.get('n_substeps', 4)})
print(f"METRIC eval_reward={stats['eval_reward']:.4f}")
print(f"METRIC catch_rate={stats['catch_rate']:.4f}")
print(f"METRIC train_reward={train_fit:.4f}")
print(f"METRIC gap={stats['eval_reward'] - train_fit:.4f}")
print(f"METRIC train_seconds={train_s:.1f}")
print(f"METRIC hist_last_best={hist[-1]['best'] if hist else 0:.4f}")
PY
)
echo "$RUN"
