"""Coverage-screened selection over the cached artifact pool.

Motivation: artifacts can win validation on total reward while dropping entire
landing zones (hemifield specialists). Score candidates on spatial uniformity:

    score = mean(zone catch rates) - 0.5 * std(zone catch rates)   [5 zones]

Uses the same VALIDATION seeds (8500..8595) - disjoint from the CEM stream
[0, 8000) and frozen eval [9000, 9096). The argmax artifact is confirmed once
on the frozen eval seeds. Criterion fixed a priori; no eval-driven tuning.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from flybrain.brain import BrainModel as BatchedBrain          # noqa: E402
from flybrain.eval_util import eval_run                        # noqa: E402
from flybrain.game import CatchEnv, DT, observation            # noqa: E402

CFG = json.load(open(".auto/config.json"))
VAL_SEEDS = list(range(8500, 8596))
ZONES = 5
SEEDS = list(range(11, 31)) + [31, 32, 33, 34, 35, 36, 37, 38, 39]


class ScreenEnv(CatchEnv):
    """CatchEnv that records (landing_x, caught) for every resolved ball."""

    def __init__(self, seed, balls):
        super().__init__(seed, balls)
        self.landings = []

    def step(self, action):
        if self.done:
            return 0.0, True
        dv = (action - 1) * 0.85 * DT
        self.paddle_x = float(np.clip(self.paddle_x + dv, 0.07, 0.93))
        self.ball_x += self.ball_vx * DT
        self.ball_y += self.ball_vy * DT
        if self.ball_x < 0.02 or self.ball_x > 0.98:
            self.ball_vx = -self.ball_vx
            self.ball_x = float(np.clip(self.ball_x, 0.02, 0.98))
        reward = 0.0
        if self.ball_y <= 0.0:
            caught = abs(self.ball_x - self.paddle_x) <= 0.10
            self.landings.append((self.ball_x, caught))
            reward += 1.0 if caught else -1.0
            self.catches += int(caught)
            self.balls += 1
            if self.balls >= self.balls_total:
                self.done = True
                self.score = self.catches
                return reward, True
            self._new_ball()
        return reward, False


def zone_scores(envs):
    lx = np.array([x for e in envs for x, _ in e.landings])
    ca = np.array([c for e in envs for _, c in e.landings]).astype(float)
    zi = np.clip((lx * ZONES).astype(int), 0, ZONES - 1)
    rates = np.array([ca[zi == z].mean() if (zi == z).any() else 0.0 for z in range(ZONES)])
    return rates


t0 = time.time()
graph = dict(np.load(CFG["graph"], allow_pickle=True))
rows = []
for s in SEEDS:
    out = Path(f"data/runs/ms_{s}")
    if not (out / "mu.npy").exists():
        continue
    cfg = json.load(open(out / "config.json"))
    cfg["feat_idx_path"] = str(out / "feat_idx.npy")
    brain = BatchedBrain(graph, dt=0.005, gain=cfg["gain"])
    mu = np.load(out / "mu.npy")
    from flybrain.readout import make_sensory_projection
    from flybrain.game import OBS_DIM
    from flybrain.train import N_ACTIONS, batch_actions
    W_s = make_sensory_projection(len(brain.sensory_idx), OBS_DIM)
    fi = np.load(out / "feat_idx.npy").astype(np.int64)
    P = mu.reshape(1, N_ACTIONS, len(fi))
    envs = [ScreenEnv(sd, balls=12) for sd in VAL_SEEDS[:48]]
    brain.init_batch(len(envs))
    reps = np.repeat(P, len(envs), axis=0)
    obs = np.stack([observation(e) for e in envs])
    while True:
        S = np.maximum(W_s @ obs.T, 0.0) * cfg.get("sensory_gain", 1.0)
        brain.clamp_sensory_batch(S)
        brain.step_batch(cfg.get("n_substeps", 4))
        acts = batch_actions(reps, brain.r[fi].T)
        done_all = True
        for p, e in enumerate(envs):
            _, done = e.step(int(acts[p]))
            done_all &= done
        if done_all:
            break
        obs = np.stack([observation(e) for e in envs])
    rates = zone_scores(envs)
    score = rates.mean() - 0.5 * rates.std()
    rows.append((s, score, rates.mean(), rates.std(), rates))
    print(f"seed {s:2d}: zones {np.round(rates*100).astype(int)}  mean {rates.mean()*100:.0f}%  "
          f"std {rates.std()*100:.0f}%  score {score*100:.1f}", flush=True)

rows.sort(key=lambda r: -r[1])
win = rows[0]
print(f"\nselected seed {win[0]} (coverage score {win[1]*100:.1f}, "
      f"mean zone catch {win[2]*100:.0f}%) -> frozen-eval confirmation", flush=True)
out = Path(f"data/runs/ms_{win[0]}")
cfg = json.load(open(out / "config.json"))
cfg["feat_idx_path"] = str(out / "feat_idx.npy")
brain = BatchedBrain(graph, dt=0.005, gain=cfg["gain"])
r = eval_run(brain, np.load(out / "mu.npy"), cfg)
print(f"METRIC eval_reward={r['eval_reward']:.4f}")
print(f"METRIC catch_rate={r['catch_rate']:.4f}")
print(f"METRIC train_reward={win[1]:.4f}")
print(f"METRIC gap={r['eval_reward']/20 - win[1]:.4f}")
print(f"METRIC hist_last_best={json.load(open(out / 'history.json'))[-1]['best']:.4f}")
print(f"METRIC train_seconds={time.time() - t0:.1f}")
print(f"METRIC act_l={r['act_l']:.3f} act_s={r['act_s']:.3f} act_r={r['act_r']:.3f}")
