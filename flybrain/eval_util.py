"""Shared evaluation routine (used by eval.py and .auto/measure.sh)."""
from __future__ import annotations

import numpy as np

from flybrain.brain import BrainModel as BatchedBrain
from flybrain.game import EVAL_SEEDS, CatchEnv, observation
from flybrain.readout import make_sensory_projection
from flybrain.train import N_ACTIONS, batch_actions


def eval_run(brain: BatchedBrain, params: np.ndarray, cfg: dict, seeds=None, batch: int = 32,
              env_cls=CatchEnv, landing_log: list | None = None):
    """Greedy evaluation on frozen eval seeds. Returns dict of stats."""
    from flybrain.game import OBS_DIM
    seeds = list(seeds if seeds is not None else EVAL_SEEDS)
    W_s = make_sensory_projection(len(brain.sensory_idx), OBS_DIM)
    feat_idx = None
    if cfg.get("feat_idx_path"):
        saved = np.load(cfg["feat_idx_path"]).astype(np.int64)
        if saved.max() < brain.n:                     # saved on this same graph
            feat_idx = saved
    if feat_idx is None:
        # recompute the same deterministic feature selection training used
        from flybrain.train import select_features
        feat_idx = select_features(brain, W_s, cfg.get("use_sensory", True),
                                   cfg.get("sensory_gain", 1.0), cfg.get("n_substeps", 4),
                                   cfg.get("dn_topk"))
    n_feat = len(feat_idx)
    P = params.reshape(1, N_ACTIONS, n_feat)
    total_reward, catches, n = 0.0, 0, 0
    act_count = np.zeros(3, np.int64)
    for c0 in range(0, len(seeds), batch):
        chunk = seeds[c0:c0 + batch]
        brain.init_batch(len(chunk))
        brain.reset_batch()
        envs = [env_cls(s) for s in chunk]
        reps = np.repeat(P, len(chunk), axis=0)          # same greedy policy for every member
        obs = np.stack([observation(e) for e in envs])
        ep_reward = np.zeros(len(chunk))
        while True:
            S = np.maximum(W_s @ obs.T, 0.0) * cfg.get("sensory_gain", 1.0)
            brain.clamp_sensory_batch(S)
            brain.step_batch(cfg.get("n_substeps", 4))
            feats = brain.r[feat_idx].T
            acts = batch_actions(reps, feats)
            for aa in acts:
                act_count[int(aa)] += 1
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
        if landing_log is not None:
            for e in envs:
                landing_log.extend(getattr(e, "landings", []))
    balls_per_ep = envs[0].balls
    act_frac = (act_count / act_count.sum()).tolist()
    return {"eval_reward": total_reward / n, "catches": catches / n,
            "catch_rate": catches / (n * balls_per_ep), "episodes": n,
            "act_l": act_frac[0], "act_s": act_frac[1], "act_r": act_frac[2]}
