"""Exact miss-margin analysis: reconstruct each ball's true landing x by
deterministic forward simulation (ball physics only, no paddle), and measure
|landing - paddle| at resolution + convergence timing.

If misses cluster just outside the 0.10 window -> input resolution (retina-16
quantization 0.0625/cell) is the binding constraint -> wide-retina axis has a
real prior. If misses are gross (0.2+) -> the policy mispredicts landing
entirely -> expressivity ceiling, retina-32 won't help.
"""
import sys

sys.path.insert(0, ".")

import json

import numpy as np
from flybrain.brain import BrainModel
from flybrain.game import OBS_DIM, CatchEnv, observation
from flybrain.readout import make_sensory_projection
from flybrain.train import N_ACTIONS, batch_actions

CHAMP = "data/runs/champ_lin"
SEEDS = list(range(7000, 7048))
TOL = 0.10  # catch tolerance (PADDLE_W/2 + 0.03)


def true_landing(x, y, vx, vy, dt=0.05):
    """Forward-simulate ball to y<=0 (same physics as CatchEnv.step)."""
    while y > 0.0:
        x += vx * dt
        y += vy * dt
        if x < 0.02 or x > 0.98:
            vx = -vx
            x = float(np.clip(x, 0.02, 0.98))
    return x


cfg = json.load(open(f"{CHAMP}/config.json"))
cfg["feat_idx_path"] = f"{CHAMP}/feat_idx.npy"
graph = dict(np.load(cfg["graph"], allow_pickle=True))
brain = BrainModel(graph, dt=0.005, gain=cfg["gain"])
mu = np.load(f"{CHAMP}/mu.npy")
feat_idx = np.load(f"{CHAMP}/feat_idx.npy")
W_s = make_sensory_projection(len(brain.sensory_idx), OBS_DIM)
P = mu.reshape(1, N_ACTIONS, len(feat_idx))

misses, catches = [], []
arrive_steps = []          # steps before landing when |land-pad| first < 0.05
final_err = []             # |land-pad| sampled at resolution
for s in SEEDS:
    brain.init_batch(1)
    brain.reset_batch()
    env = CatchEnv(s)
    obs = observation(env)
    arrived = False
    while True:
        land = true_landing(env.ball_x, env.ball_y, env.ball_vx, env.ball_vy)
        d = abs(land - env.paddle_x)
        if not arrived and d < 0.05:
            arrived = True
        S = np.maximum(W_s @ obs[None].T, 0.0)
        brain.clamp_sensory_batch(S)
        brain.step_batch(cfg.get("n_substeps", 4))
        feats = brain.r[feat_idx].T
        act = int(batch_actions(P, feats)[0])
        r_prev, balls_prev = 0.0, env.balls
        _, done = env.step(act)
        if env.balls > balls_prev:                    # ball resolved
            caught = r_prev if False else None
            final_err.append(d)
            arrive_steps.append(arrived)
            if abs(d) <= TOL:
                catches.append(d)
            else:
                misses.append(d)
            arrived = False
        if done:
            break
        obs = observation(env)

misses = np.array(misses)
catches = np.array(catches)
final_err = np.array(final_err)
print(f"balls {len(final_err)}: caught {len(catches)} missed {len(misses)} "
      f"({len(misses)/len(final_err)*100:.1f}%)")
print(f"miss margins |land-pad|: median {np.median(misses):.3f}  mean {misses.mean():.3f}")
for lo, hi in ((0.10, 0.12), (0.12, 0.15), (0.15, 0.20), (0.20, 0.30), (0.30, 1.0)):
    n = int(((misses >= lo) & (misses < hi)).sum())
    print(f"  {lo:.2f}-{hi:.2f}: {n}  ({n/max(len(misses),1)*100:.0f}% of misses)")
print(f"marginal (0.10-0.15): {int(((misses >= 0.10) & (misses < 0.15)).sum())}/{len(misses)}"
      f"  gross (>0.20): {int((misses >= 0.20).sum())}/{len(misses)}")
arr = np.array([a for a in arrive_steps if a is not None], dtype=object)
print(f"policy ever within 0.05 of true landing: {sum(bool(a) for a in arrive_steps)}/{len(arrive_steps)}")
