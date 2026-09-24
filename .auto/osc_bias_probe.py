"""Oscillation-vs-bias protocol: diagnostic on fresh train-stream seeds +
champion frozen-eval reference lines (informational, like char/valnoise).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np                                            # noqa: E402
from flybrain.brain import BrainModel                         # noqa: E402
from flybrain.eval_util import eval_run                       # noqa: E402
from flybrain.game import OBS_DIM, CatchEnv, observation      # noqa: E402
from flybrain.readout import make_sensory_projection          # noqa: E402
from flybrain.train import N_ACTIONS, batch_actions           # noqa: E402

CHAMP = Path("data/runs/champ_lin")
SEEDS = list(range(7000, 7048))
TOL = 0.10


def true_landing(x, y, vx, vy, dt=0.05):
    while y > 0.0:
        x += vx * dt
        y += vy * dt
        if x < 0.02 or x > 0.98:
            vx = -vx
            x = float(np.clip(x, 0.02, 0.98))
    return x


t0 = time.time()
cfg = json.load(open(CHAMP / "config.json"))
cfg["feat_idx_path"] = str(CHAMP / "feat_idx.npy")
graph = dict(np.load(cfg["graph"], allow_pickle=True))
brain = BrainModel(graph, dt=0.005, gain=cfg["gain"])
mu = np.load(CHAMP / "mu.npy")
feat_idx = np.load(CHAMP / "feat_idx.npy")
W_s = make_sensory_projection(len(brain.sensory_idx), OBS_DIM)
P = mu.reshape(1, N_ACTIONS, len(feat_idx))

miss_final, miss_med = [], []
for s in SEEDS:
    brain.init_batch(1)
    brain.reset_batch()
    env = CatchEnv(s)
    obs = observation(env)
    trail = []
    while True:
        trail.append((true_landing(env.ball_x, env.ball_y, env.ball_vx, env.ball_vy), env.paddle_x))
        S = np.maximum(W_s @ obs[None].T, 0.0)
        brain.clamp_sensory_batch(S)
        brain.step_batch(cfg.get("n_substeps", 4))
        feats = brain.r[feat_idx].T
        act = int(batch_actions(P, feats)[0])
        balls_prev = env.balls
        _, done = env.step(act)
        if env.balls > balls_prev:
            tail = trail[-10:]
            lands = np.array([t[0] for t in tail])
            pads = np.array([t[1] for t in tail])
            if abs(lands[-1] - pads[-1]) > TOL:
                miss_final.append(abs(lands[-1] - pads[-1]))
                miss_med.append(abs(np.median(lands) - np.median(pads)))
            trail = []
        if done:
            break
        obs = observation(env)

miss_final, miss_med = np.array(miss_final), np.array(miss_med)
close = (miss_med <= 0.05).sum()
print(f"missed {len(miss_final)}: final median {np.median(miss_final):.3f}, "
      f"10-step median-pair {np.median(miss_med):.3f}, within-0.05 {close}/{len(miss_med)}", flush=True)
print(f"VERDICT: {'BIAS' if np.median(miss_med) > 0.05 else 'OSCILLATION'} - "
      f"paddle {'settles off-target' if np.median(miss_med) > 0.05 else 'straddles target'}", flush=True)

r = eval_run(brain, mu, cfg)
print(f"METRIC eval_reward={r['eval_reward']:.4f}")
print(f"METRIC catch_rate={r['catch_rate']:.4f}")
print(f"METRIC train_reward=0")
print(f"METRIC gap=0")
print(f"METRIC hist_last_best=0")
print(f"METRIC train_seconds={time.time() - t0:.1f}")
