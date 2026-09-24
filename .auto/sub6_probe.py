"""Substeps-6 preflight: (a) does the DN top-256 set change under finer
integration? (b) zero-shot transfer of the champion mu to substeps 6 on val."""
import sys

sys.path.insert(0, ".")

import numpy as np
from flybrain.brain import BrainModel as BatchedBrain
from flybrain.eval_util import eval_run
from flybrain.train import select_features
from flybrain.game import OBS_DIM
from flybrain.readout import make_sensory_projection

CHAMP = "data/runs/champ_lin"
VAL_SEEDS = list(range(8500, 8548))

cfg = __import__("json").load(open(f"{CHAMP}/config.json"))
cfg["feat_idx_path"] = f"{CHAMP}/feat_idx.npy"
graph = dict(np.load(cfg["graph"], allow_pickle=True))
brain = BatchedBrain(graph, dt=0.005, gain=cfg["gain"])
W_s = make_sensory_projection(len(brain.sensory_idx), OBS_DIM)

feat4 = np.load(f"{CHAMP}/feat_idx.npy")
feat6 = select_features(brain, W_s, True, 1.0, 6, 256)
ov = len(np.intersect1d(feat4, feat6))
print(f"feat overlap: {ov}/{len(feat4)} identical={np.array_equal(feat4, feat6)}")

mu = np.load(f"{CHAMP}/mu.npy")
for sub in (4, 6):
    c = dict(cfg)
    c["n_substeps"] = sub
    v = eval_run(brain, mu, c, seeds=VAL_SEEDS, batch=48)
    print(f"champ mu @ substeps {sub}: val {v['eval_reward']:+.2f}  catch {v['catch_rate']*100:.1f}%")
