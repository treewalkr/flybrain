"""Which DNs actually carry landing information?

Run the champion on fresh train-stream seeds; at every decision step record all
DN rates + the current ball's TRUE landing x (forward-sim) and the champion's
signed target error (landing - paddle). Rank DNs by |Pearson r| with landing x
and with target error. Compare the MI-top-256 against the champion's
variance-top-256: if the champion set already contains the informative DNs,
the feature axis is dead (bias inherent to linear-over-these-features).
If dense DNs exist outside it -> MI-selected zero-padded expansion is warranted.
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
SEEDS = list(range(6000, 6024))


def true_landing(x, y, vx, vy, dt=0.05):
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
dn_idx = brain.dn_idx
mu = np.load(f"{CHAMP}/mu.npy")
feat_idx = np.load(f"{CHAMP}/feat_idx.npy")
champ_dn = feat_idx[feat_idx < brain.n][:256] if len(feat_idx) == 556 else None
# champion DN features = feat set minus the 300 sensory indices
sens_set = set(brain.sensory_idx.tolist())
champ_dn = np.array([i for i in feat_idx if i not in sens_set])
W_s = make_sensory_projection(len(brain.sensory_idx), OBS_DIM)
P = mu.reshape(1, N_ACTIONS, len(feat_idx))
SUB = cfg.get("n_substeps", 4)

k = len(SEEDS)
brain.init_batch(k)
brain.reset_batch()
envs = [CatchEnv(s) for s in SEEDS]
obs = np.stack([observation(e) for e in envs])

R, LAND, ERR = [], [], []
while True:
    land = np.array([true_landing(e.ball_x, e.ball_y, e.ball_vx, e.ball_vy) for e in envs])
    S = np.maximum(W_s @ obs.T, 0.0)
    brain.clamp_sensory_batch(S)
    brain.step_batch(SUB)
    R.append(brain.r[dn_idx].copy())            # (n_dn, k)
    LAND.append(land.copy())
    ERR.append(land - np.array([e.paddle_x for e in envs]))
    feats = brain.r[feat_idx].T
    acts = batch_actions(np.repeat(P, k, axis=0), feats)
    done_all = True
    for p, e in enumerate(envs):
        _, done = e.step(int(acts[p]))
        done_all &= done
    if done_all:
        break
    obs = np.stack([observation(e) for e in envs])

R = np.concatenate(R, axis=1).T                  # (T*k, n_dn) - careful: r is (n, batch)
LAND = np.concatenate(LAND)
ERR = np.concatenate(ERR)
print(f"samples {R.shape[0]}  DNs {R.shape[1]}  champion DN feats {len(champ_dn)}")

Rc = R - R.mean(0)
lz = (LAND - LAND.mean()) / LAND.std()
ez = (ERR - ERR.mean()) / ERR.std()
r_land = (Rc / (R.std(0) + 1e-12)).T @ lz / len(lz)
r_err = (Rc / (R.std(0) + 1e-12)).T @ ez / len(ez)

pos = {int(g): i for i, g in enumerate(dn_idx)}
order_land = np.argsort(np.abs(r_land))[::-1]
mi_top = set(dn_idx[order_land[:256]].tolist())
champ_set = set(champ_dn.tolist())
ov = len(mi_top & champ_set)
print(f"|r| with landing x: top-256-MI vs champion-256 overlap {ov}/256")
champ_pos = [pos[g] for g in champ_set]
nc = [i for i in range(len(dn_idx)) if int(dn_idx[i]) not in champ_set]
print(f"champion set mean |r_land| {np.abs(r_land)[champ_pos].mean():.4f}  "
      f"non-champion max {np.abs(r_land)[nc].max():.4f}")
print(f"best non-champion |r_land| DNs: ", end="")
for i in sorted(nc, key=lambda i: -abs(r_land[i]))[:8]:
    print(f"dn{dn_idx[i]}:{abs(r_land[i]):.3f}", end="  ")
print()
top8 = [i for i in order_land[:8]]
print(f"overall top-8 |r_land|: ", end="")
for i in top8:
    print(f"dn{dn_idx[i]}:{abs(r_land[i]):.3f}{'*' if dn_idx[i] in champ_set else ' '}", end="  ")
print("\n(* = in champion set)")
print(f"|r| with target error: champion mean {np.abs(r_err)[champ_pos].mean():.4f}  "
      f"overall max {np.abs(r_err).max():.4f}")
