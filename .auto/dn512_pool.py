"""Zero-padded top-512 refinement: capacity expansion AT the optimum.

Fresh-pool top-512 collapsed (stationary), but that was a from-scratch search.
Here the champion's 556-feature readout is embedded into the 812-feature space
(top-512 DN by variance + sensory), new feature weights zeroed: initial behavior
is EXACTLY the champion, and CEM gets 256 extra readout dimensions to climb with.
Val-gated at +15.17; frozen eval only if a draw clears.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np                                            # noqa: E402
from flybrain.brain import BrainModel as BatchedBrain         # noqa: E402
from flybrain.eval_util import eval_run                       # noqa: E402
from flybrain.game import OBS_DIM                             # noqa: E402
from flybrain.readout import make_sensory_projection          # noqa: E402
from flybrain.train import N_ACTIONS, select_features, train  # noqa: E402

CFG = json.load(open(".auto/config.json"))
CHAMP = Path("data/runs/champ_lin")
GATE = 15.17
VAL_SEEDS = list(range(8500, 8548))

t0 = time.time()
cfg = json.load(open(CHAMP / "config.json"))
graph = dict(np.load(cfg["graph"], allow_pickle=True))
brain = BatchedBrain(graph, dt=0.005, gain=cfg["gain"])
W_s = make_sensory_projection(len(brain.sensory_idx), OBS_DIM)

feat256 = np.load(CHAMP / "feat_idx.npy")
feat512 = select_features(brain, W_s, True, 1.0, 4, 512)
missing = np.setdiff1d(feat256, feat512)
print(f"feat512: {len(feat512)} feats; champion feats missing: {len(missing)}", flush=True)
assert len(missing) == 0, "champion features must be a subset of the widened set"

# embed champion weights: champion col j (feat256[j]) -> slot in feat512
pos = np.searchsorted(feat512, feat256)                       # feat512 is sorted unique
mu0 = np.zeros(N_ACTIONS * len(feat512), np.float32)
P0 = np.load(CHAMP / "mu.npy").reshape(N_ACTIONS, len(feat256))
P1 = mu0.reshape(N_ACTIONS, len(feat512))
P1[:, pos] = P0
# sanity: embedded policy must reproduce champion val exactly
np.save(CHAMP / "feat512.npy", feat512)
v_emb = eval_run(brain, P1.reshape(-1), {"n_substeps": 4, "feat_idx_path": CHAMP / "feat512.npy"},
                 seeds=VAL_SEEDS[:12], batch=12)
v_cha = eval_run(brain, P0.reshape(-1), {"n_substeps": 4, "feat_idx_path": CHAMP / "feat_idx.npy"},
                 seeds=VAL_SEEDS[:12], batch=12)
print(f"embedding sanity: embedded {v_emb['eval_reward']:+.2f} vs champion "
      f"{v_cha['eval_reward']:+.2f} (12 eps)", flush=True)
assert abs(v_emb["eval_reward"] - v_cha["eval_reward"]) < 1e-9, "embedding must be exact"

best = None
for s, sig in ((121, 0.15), (122, 0.12)):
    out = Path(f"data/runs/dn512_{s}")
    if not (out / "mu.npy").exists():
        train(seed=s, out=out, workers=6, graph_path=CFG["graph"],
              iters=24, pop=64, elites=10, eps=3, use_sensory=True, sensory_gain=1.0,
              n_substeps=4, gain=1.0, dn_topk=512, sigma_decay=0.9, sigma_floor=0.02,
              train_balls=12, quiet=True, mu_init=mu0, sigma_init=sig,
              feat_idx_override=feat512)
    c = {"n_substeps": 4, "feat_idx_path": str(out / "feat_idx.npy"), "sensory_gain": 1.0}
    v = eval_run(brain, np.load(out / "mu.npy"), c, seeds=VAL_SEEDS, batch=48)
    hist = json.load(open(out / "history.json"))
    print(f"dn512 refine {s} (sig {sig}): val {v['eval_reward']:+.2f} "
          f"catch {v['catch_rate']*100:.0f}% lastbest {hist[-1]['best']:+.1f} ({time.time()-t0:.0f}s)",
          flush=True)
    if best is None or v["eval_reward"] > best[1]:
        best = (s, v["eval_reward"], out)

s, val, out = best
hist = json.load(open(out / "history.json"))
if val <= GATE:
    print(f"gate: {val:+.2f} <= {GATE} - no eval spend; axis closed", flush=True)
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
