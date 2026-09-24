"""From-scratch MI-feature pool: landing-correlation ranking replaces variance.

The decisive feature-selection test. dn_info_probe: variance-top-256 contains
only 58/256 of the top landing encoders (mean |r_land| 0.169 vs 0.54 max).
Augmentation failed (run 77: CEM can't reorganize around extras from champion
init); from-scratch answers whether CEM finds better policies when the readout
is built on the strong features from the start.

Ranking: |Pearson r| between DN rate and true landing x, computed on the SAME
random-action calibration rollout protocol as select_features (training stream
only). Pool: 4 fresh draws (96 iters), val-gated at the from-scratch +8.42.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np                                            # noqa: E402
from flybrain.brain import BrainModel as BatchedBrain         # noqa: E402
from flybrain.eval_util import eval_run                       # noqa: E402
from flybrain.game import OBS_DIM, CatchEnv, observation      # noqa: E402
from flybrain.readout import make_sensory_projection          # noqa: E402
from flybrain.train import train                              # noqa: E402

CFG = json.load(open(".auto/config.json"))
RUNS = {141: {}, 142: {}, 143: {}, 144: {"pop": 96, "eps": 4}}
VAL_SEEDS = list(range(8500, 8548))
GATE = 8.42


def true_landing(x, y, vx, vy, dt=0.05):
    while y > 0.0:
        x += vx * dt
        y += vy * dt
        if x < 0.02 or x > 0.98:
            vx = -vx
            x = float(np.clip(x, 0.02, 0.98))
    return x


t0 = time.time()
graph = dict(np.load(CFG["graph"], allow_pickle=True))
brain = BatchedBrain(graph, dt=0.005, gain=1.0)
W_s = make_sensory_projection(len(brain.sensory_idx), OBS_DIM)

# random-action calibration rollout (same protocol/seed as select_features)
rng = np.random.default_rng(123)
batch = 16
brain.init_batch(batch)
brain.reset_batch()
envs = [CatchEnv(int(rng.integers(0, 8000))) for _ in range(batch)]
R, LAND = [], []
obs = np.stack([observation(e) for e in envs])
for _ in range(120):
    LAND.append(np.array([true_landing(e.ball_x, e.ball_y, e.ball_vx, e.ball_vy) for e in envs]))
    S = np.maximum(W_s @ obs.T, 0.0)
    brain.clamp_sensory_batch(S)
    brain.step_batch(4)
    R.append(brain.r[brain.dn_idx].copy())
    for e in envs:
        e.step(int(rng.integers(0, 3)))
    obs = np.stack([observation(e) for e in envs])
R = np.concatenate(R, axis=1).T
L = np.concatenate(LAND)
Rc = R - R.mean(0)
lz = (L - L.mean()) / L.std()
r_land = np.abs((Rc / (R.std(0) + 1e-12)).T @ lz / len(lz))
dn_idx = brain.dn_idx
mi_top = dn_idx[np.argsort(r_land)[::-1][:256]]
feat = np.unique(np.concatenate([np.sort(mi_top), brain.sensory_idx]))
print(f"MI features: {len(feat)} (top-256 by |r_land| under random calibration, "
      f"r {np.sort(r_land)[::-1][:256].min():.3f}..{np.sort(r_land)[::-1][0]:.3f})", flush=True)
np.save("data/runs/feat_mi_scratch.npy", feat)
brain.init_batch(1)

BUDGET_S = float(__import__("os").environ.get("PILOT_BUDGET_S", "2400"))
best = None
for s, ov in RUNS.items():
    out = Path(f"data/runs/mis_{s}")
    if not (out / "mu.npy").exists():
        if time.time() - t0 > BUDGET_S:
            print(f"budget reached before seed {s} - resumable", flush=True)
            continue
        rc = {**CFG, **ov}
        train(seed=s, out=out, workers=6, graph_path=CFG["graph"],
              iters=96, pop=rc.get("pop", 64), elites=rc.get("elites", 10),
              eps=rc.get("eps", 6), use_sensory=True, sensory_gain=1.0, n_substeps=4,
              gain=1.0, dn_topk=256, sigma_decay=0.9, sigma_floor=0.02,
              train_balls=12, quiet=True, feat_idx_override=feat)
    c = {"n_substeps": 4, "feat_idx_path": str(out / "feat_idx.npy"), "sensory_gain": 1.0}
    v = eval_run(brain, np.load(out / "mu.npy"), c, seeds=VAL_SEEDS, batch=48)
    hist = json.load(open(out / "history.json"))
    print(f"mis seed {s}: val {v['eval_reward']:+.2f}  catch {v['catch_rate']*100:.0f}%  ({time.time()-t0:.0f}s)",
          flush=True)
    if best is None or v["eval_reward"] > best[1]:
        best = (s, v["eval_reward"], out)

s, val, out = best
hist = json.load(open(out / "history.json"))
if val <= GATE:
    print(f"gate: {val:+.2f} <= {GATE:+.2f} - no eval spend; feature-selection question answered",
          flush=True)
    print(f"METRIC eval_reward={val:.4f}")
    print(f"METRIC catch_rate=0 train_reward={val:.4f} gap=0 "
          f"hist_last_best={hist[-1]['best']:.4f} train_seconds={time.time()-t0:.1f}")
    sys.exit(0)

c = {"n_substeps": 4, "feat_idx_path": str(out / "feat_idx.npy"), "sensory_gain": 1.0}
r = eval_run(brain, np.load(out / "mu.npy"), c)
print(f"METRIC eval_reward={r['eval_reward']:.4f}")
print(f"METRIC catch_rate={r['catch_rate']:.4f}")
print(f"METRIC train_reward={val:.4f}")
print(f"METRIC gap={r['eval_reward'] - val:.4f}")
print(f"METRIC hist_last_best={hist[-1]['best']:.4f}")
print(f"METRIC train_seconds={time.time() - t0:.1f}")
print(f"METRIC act_l={r['act_l']:.3f} act_s={r['act_s']:.3f} act_r={r['act_r']:.3f}")
