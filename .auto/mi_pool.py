"""MI-selected feature expansion: zero-padded champion + landing-informative DNs.

dn_info_probe found the variance-selected champion features are near-noise for
landing (mean |r| 0.169) while DNs outside the set reach |r| 0.54. Run 73's
expansion failed because variance-ranks 257-512 are low-signal by construction;
this expansion adds the TOP landing encoders instead.

Protocol: rank DNs by |Pearson r| with true landing x on champion rollouts
(train-stream seeds only); stability-check the ranking across seed halves;
extras = top-128 non-champion; embed champion mu zero-padded (init == champion);
2 refinement draws, val-gated at +15.17; frozen eval only if cleared.
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
from flybrain.train import N_ACTIONS, batch_actions, train    # noqa: E402

CFG = json.load(open(".auto/config.json"))
CHAMP = Path("data/runs/champ_lin")
GATE = 15.17
VAL_SEEDS = list(range(8500, 8548))


def true_landing(x, y, vx, vy, dt=0.05):
    while y > 0.0:
        x += vx * dt
        y += vy * dt
        if x < 0.02 or x > 0.98:
            vx = -vx
            x = float(np.clip(x, 0.02, 0.98))
    return x


def landing_corr(brain, seeds, mu, feat_idx, cfg):
    """|Pearson r| between each DN rate and the current ball's true landing."""
    W_s = make_sensory_projection(len(brain.sensory_idx), OBS_DIM)
    P = mu.reshape(1, N_ACTIONS, len(feat_idx))
    k = len(seeds)
    brain.init_batch(k)
    brain.reset_batch()
    envs = [CatchEnv(s) for s in seeds]
    obs = np.stack([observation(e) for e in envs])
    R, LAND = [], []
    while True:
        LAND.append(np.array([true_landing(e.ball_x, e.ball_y, e.ball_vx, e.ball_vy) for e in envs]))
        S = np.maximum(W_s @ obs.T, 0.0)
        brain.clamp_sensory_batch(S)
        brain.step_batch(cfg.get("n_substeps", 4))
        R.append(brain.r[brain.dn_idx].copy())
        feats = brain.r[feat_idx].T
        acts = batch_actions(np.repeat(P, k, axis=0), feats)
        done_all = True
        for p, e in enumerate(envs):
            _, done = e.step(int(acts[p]))
            done_all &= done
        if done_all:
            break
        obs = np.stack([observation(e) for e in envs])
    R = np.concatenate(R, axis=1).T
    L = np.concatenate(LAND)
    Rc = R - R.mean(0)
    lz = (L - L.mean()) / L.std()
    return np.abs((Rc / (R.std(0) + 1e-12)).T @ lz / len(lz))


t0 = time.time()
cfg = json.load(open(CHAMP / "config.json"))
graph = dict(np.load(cfg["graph"], allow_pickle=True))
brain = BatchedBrain(graph, dt=0.005, gain=cfg["gain"])
mu = np.load(CHAMP / "mu.npy")
feat_idx = np.load(CHAMP / "feat_idx.npy")
sens_set = set(brain.sensory_idx.tolist())
champ_dn = np.array([i for i in feat_idx if i not in sens_set])
dn_idx = brain.dn_idx

rA = landing_corr(brain, list(range(6000, 6012)), mu, feat_idx, cfg)
rB = landing_corr(brain, list(range(6012, 6024)), mu, feat_idx, cfg)
topA = set(dn_idx[np.argsort(rA)[::-1][:256]].tolist())
topB = set(dn_idx[np.argsort(rB)[::-1][:256]].tolist())
print(f"ranking stability: top-256 overlap across seed halves {len(topA & topB)}/256", flush=True)

r_full = landing_corr(brain, list(range(6000, 6024)), mu, feat_idx, cfg)
champ_pos = {int(g): i for i, g in enumerate(dn_idx)}
order = np.argsort(r_full)[::-1]
extras = []
for i in order:
    g = int(dn_idx[i])
    if g not in set(champ_dn.tolist()):
        extras.append(g)
    if len(extras) == 128:
        break
print(f"extras: top-128 non-champion by |r_land|, r range "
      f"{r_full[champ_pos[extras[-1]]]:.3f}..{r_full[champ_pos[extras[0]]]:.3f} "
      f"(champion mean {r_full[[champ_pos[int(g)] for g in champ_dn]].mean():.3f})", flush=True)

new_feat = np.sort(np.concatenate([feat_idx, np.array(extras)]))
n_new = len(new_feat)
pos = np.searchsorted(new_feat, feat_idx)
mu0 = np.zeros(N_ACTIONS * n_new, np.float32)
P0 = mu.reshape(N_ACTIONS, len(feat_idx))
P1 = mu0.reshape(N_ACTIONS, n_new)
P1[:, pos] = P0
np.save(CHAMP / "feat_mi.npy", new_feat)
v_emb = eval_run(brain, P1.reshape(-1), {"n_substeps": 4, "feat_idx_path": CHAMP / "feat_mi.npy",
                                          "sensory_gain": 1.0}, seeds=VAL_SEEDS[:12], batch=12)
v_cha = eval_run(brain, mu, {"n_substeps": 4, "feat_idx_path": CHAMP / "feat_idx.npy",
                             "sensory_gain": 1.0}, seeds=VAL_SEEDS[:12], batch=12)
print(f"embedding sanity: {v_emb['eval_reward']:+.2f} vs {v_cha['eval_reward']:+.2f}", flush=True)
assert abs(v_emb["eval_reward"] - v_cha["eval_reward"]) < 1e-9

best = None
for s, sig in ((131, 0.15), (132, 0.12)):
    out = Path(f"data/runs/mi_{s}")
    if not (out / "mu.npy").exists():
        train(seed=s, out=out, workers=6, graph_path=CFG["graph"],
              iters=24, pop=64, elites=10, eps=3, use_sensory=True, sensory_gain=1.0,
              n_substeps=4, gain=1.0, dn_topk=256, sigma_decay=0.9, sigma_floor=0.02,
              train_balls=12, quiet=True, mu_init=mu0, sigma_init=sig,
              feat_idx_override=new_feat)
    c = {"n_substeps": 4, "feat_idx_path": str(out / "feat_idx.npy"), "sensory_gain": 1.0}
    v = eval_run(brain, np.load(out / "mu.npy"), c, seeds=VAL_SEEDS, batch=48)
    hist = json.load(open(out / "history.json"))
    print(f"mi refine {s} (sig {sig}): val {v['eval_reward']:+.2f} catch {v['catch_rate']*100:.0f}% "
          f"lastbest {hist[-1]['best']:+.1f} ({time.time()-t0:.0f}s)", flush=True)
    if best is None or v["eval_reward"] > best[1]:
        best = (s, v["eval_reward"], out)

s, val, out = best
hist = json.load(open(out / "history.json"))
if val <= GATE:
    print(f"gate: {val:+.2f} <= {GATE} - no eval spend", flush=True)
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
