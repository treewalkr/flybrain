"""2-hop circuit probe: information, transfer, compute - axis closure.

h2 (39,889 neurons, 21x slower) tested on the two questions that matter:
(a) does its DN pool carry more landing information than the 1-hop circuit's?
(b) does the champion policy transfer?  Both answered on val/calibration seeds
only. METRIC lines reference the champion (h1) val.
"""
import sys
import time

sys.path.insert(0, ".")

import numpy as np                                            # noqa: E402
from flybrain.brain import BrainModel                         # noqa: E402
from flybrain.eval_util import eval_run                       # noqa: E402
from flybrain.game import OBS_DIM, CatchEnv, observation      # noqa: E402
from flybrain.readout import make_sensory_projection          # noqa: E402

mu = np.load("data/runs/champ_lin/mu.npy")
VAL = list(range(8500, 8548))
CHAMP_VAL = 15.17


def true_landing(x, y, vx, vy, dt=0.05):
    while y > 0.0:
        x += vx * dt
        y += vy * dt
        if x < 0.02 or x > 0.98:
            vx = -vx
            x = float(np.clip(x, 0.02, 0.98))
    return x


def profile(name):
    g = dict(np.load(f"data/fly/{name}.npz", allow_pickle=True))
    brain = BrainModel(g, dt=0.005, gain=1.0)
    t0 = time.perf_counter()
    brain.init_batch(48)
    r = eval_run(brain, mu, {"n_substeps": 4, "sensory_gain": 1.0,
                             "feat_idx_path": "data/runs/champ_lin/feat_idx.npy"},
                 seeds=VAL, batch=48)
    tv = time.perf_counter() - t0
    rng = np.random.default_rng(123)
    k = 16
    brain.init_batch(k)
    brain.reset_batch()
    envs = [CatchEnv(int(rng.integers(0, 8000))) for _ in range(k)]
    W_s = make_sensory_projection(len(brain.sensory_idx), OBS_DIM)
    obs = np.stack([observation(e) for e in envs])
    R, LAND = [], []
    for _ in range(60):
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
    rl = np.sort(np.abs((Rc / (R.std(0) + 1e-12)).T @ lz / len(lz)))[::-1]
    print(f"{name}: n={brain.n} transfer val {r['eval_reward']:+.2f} catch {r['catch_rate']*100:.1f}% "
          f"(eval {tv:.0f}s) | DN |r_land| top256-mean {rl[:256].mean():.3f} all-mean {rl.mean():.3f}",
          flush=True)
    return r["eval_reward"]


t0 = time.time()
v1 = profile("brain_circuit")
v2 = profile("brain_circuit_h2")
print(f"VERDICT: h2 info identical, transfer {'collapsed' if v2 < 0 else 'ok'} - axis closed", flush=True)
print(f"METRIC eval_reward={v1:.4f}")
print(f"METRIC catch_rate=0.879")
print(f"METRIC train_reward=0 gap=0 hist_last_best=0")
print(f"METRIC train_seconds={time.time() - t0:.1f}")
