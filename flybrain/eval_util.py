"""Shared evaluation routine (used by eval.py and .auto/measure.sh)."""
from __future__ import annotations

import numpy as np
import torch

from flybrain.brain import BatchedBrain
from flybrain.game import EVAL_SEEDS, CatchEnv, observation
from flybrain.readout import make_sensory_projection
from flybrain.train import N_ACTIONS, batch_actions


@torch.no_grad()
def eval_run(brain: BatchedBrain, params: np.ndarray, cfg: dict, seeds=None, batch: int = 32):
    """Greedy evaluation on frozen eval seeds. Returns dict of stats."""
    from flybrain.game import OBS_DIM
    seeds = list(seeds if seeds is not None else EVAL_SEEDS)
    device = brain.device
    W_s = make_sensory_projection(int(brain.sensory_idx.numel()), OBS_DIM).to(device)
    feat_idx = torch.cat([brain.dn_idx, brain.sensory_idx]) if cfg.get("use_sensory", True) else brain.dn_idx
    n_feat = int(feat_idx.numel())
    P = params.reshape(1, N_ACTIONS, n_feat)
    total_reward, catches, n = 0.0, 0, 0
    for c0 in range(0, len(seeds), batch):
        chunk = seeds[c0:c0 + batch]
        brain.init_batch(len(chunk))
        brain.reset_batch()
        envs = [CatchEnv(s) for s in chunk]
        reps = np.repeat(P, len(chunk), axis=0)          # same greedy policy for every member
        obs = np.stack([observation(e) for e in envs])
        ep_reward = np.zeros(len(chunk))
        while True:
            O = torch.from_numpy(obs).to(device)
            S = torch.relu(W_s @ O.T) * cfg.get("sensory_gain", 1.0)
            brain.clamp_sensory_batch(S)
            brain.step_batch(cfg.get("n_substeps", 4))
            feats = brain.r[feat_idx].detach().cpu().numpy().T
            acts = batch_actions(reps, feats)
            done_all = True
            for p, e in enumerate(envs):
                r, done = e.step(int(acts[p]))
                ep_reward[p] += r
                done_all &= done
            if done_all:
                break
            obs = np.stack([observation(e) for e in envs])
        total_reward += float(ep_reward.sum())
        catches += sum(e.catches for e in envs)
        n += len(chunk)
    balls_per_ep = envs[0].balls
    return {"eval_reward": total_reward / n, "catches": catches / n,
            "catch_rate": catches / (n * balls_per_ep), "episodes": n}
