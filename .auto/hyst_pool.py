"""Margin-argmax decoding (hysteresis) - readout-side dither filter.

Diagnosis (.auto/miss_probe.py): champion act_l/s/r = 0.50/0.02/0.48 - the
policy NEVER holds still; it switches every ~2.8 steps and travels 1.5 field
widths per ball. Misses are diffuse (no regime concentration), so the lever is
the decoder, not features: keep the current action unless the best alternative
score exceeds the current action's score by margin m. m=0 ~ plain argmax.
Grid m on val (selection only), gate vs +15.17, single frozen-eval spend if cleared.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np                                            # noqa: E402
from flybrain.brain import BrainModel as BatchedBrain         # noqa: E402
from flybrain.game import EVAL_SEEDS, OBS_DIM, CatchEnv, observation  # noqa: E402
from flybrain.readout import make_sensory_projection          # noqa: E402
from flybrain.train import N_ACTIONS                           # noqa: E402

CHAMP = Path("data/runs/champ_lin")
VAL_SEEDS = list(range(8500, 8548))
GATE = 15.17

cfg = json.load(open(CHAMP / "config.json"))
feat_idx = np.load(CHAMP / "feat_idx.npy")
mu = np.load(CHAMP / "mu.npy")
P = mu.reshape(1, N_ACTIONS, len(feat_idx))
SUB = cfg.get("n_substeps", 4)


def eval_hyst(brain, m, seeds, batch=48):
    """Batched greedy eval with margin-argmax decoding."""
    W_s = make_sensory_projection(len(brain.sensory_idx), OBS_DIM)
    total, catches, n = 0.0, 0, 0
    for c0 in range(0, len(seeds), batch):
        chunk = seeds[c0:c0 + batch]
        k = len(chunk)
        brain.init_batch(k)
        brain.reset_batch()
        envs = [CatchEnv(s) for s in chunk]
        reps = np.repeat(P, k, axis=0)
        acts = np.full(k, -1, np.int64)          # -1: force first-step argmax
        ep_r = np.zeros(k)
        obs = np.stack([observation(e) for e in envs])
        while True:
            S = np.maximum(W_s @ obs.T, 0.0)
            brain.clamp_sensory_batch(S)
            brain.step_batch(SUB)
            F = brain.r[feat_idx].T
            scores = np.einsum("pf,paf->pa", F, reps)
            arg = scores.argmax(axis=1)
            cur = np.where(acts < 0, arg, acts)
            margin = scores[np.arange(k), arg] - scores[np.arange(k), cur]
            acts = np.where((acts < 0) | (margin >= m), arg, cur)
            done_all = True
            for p, e in enumerate(envs):
                r, done = e.step(int(acts[p]))
                ep_r[p] += r
                done_all &= done
            if done_all:
                break
            obs = np.stack([observation(e) for e in envs])
        total += float(ep_r.sum())
        catches += sum(e.catches for e in envs)
        n += k
    balls = envs[0].balls
    return total / n, catches / (n * balls)


t0 = time.time()
graph = dict(np.load(cfg["graph"], allow_pickle=True))
results = []
for m in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8):
    brain = BatchedBrain(graph, dt=0.005, gain=cfg["gain"])
    v, c = eval_hyst(brain, m, VAL_SEEDS)
    results.append((m, v, c))
    print(f"m={m:.2f}: val {v:+.2f}  catch {c*100:.1f}%  ({time.time()-t0:.0f}s)", flush=True)

best = max(results, key=lambda r: r[1])
print(f"best: m={best[0]} val {best[1]:+.2f} (plain-argmax ref "
      f"{results[0][1]:+.2f})", flush=True)

if best[1] <= GATE:
    print(f"gate: {best[1]:+.2f} <= {GATE} - no eval spend; decoder axis closed", flush=True)
    print(f"METRIC eval_reward={best[1]:.4f}")
    print(f"METRIC catch_rate=0 train_reward={best[1]:.4f} gap=0 hist_last_best=0 "
          f"train_seconds={time.time()-t0:.1f}")
    sys.exit(0)

m, v, c = best
brain = BatchedBrain(graph, dt=0.005, gain=cfg["gain"])
r, cc = eval_hyst(brain, m, list(EVAL_SEEDS))
print(f"FROZEN EVAL: m={m} reward {r:+.2f} catch {cc*100:.1f}%", flush=True)
print(f"METRIC eval_reward={r:.4f}")
print(f"METRIC catch_rate={cc:.4f}")
print(f"METRIC train_reward={v:.4f}")
print(f"METRIC gap={r - v:.4f}")
print(f"METRIC hist_last_best=0")
print(f"METRIC train_seconds={time.time() - t0:.1f}")
