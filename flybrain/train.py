"""CEM training of the linear readout over the frozen MaleCNS subgraph circuit.

    .venv/bin/python -m flybrain.train [--iters 16] [--pop 64] [--out data/runs/cem_v0]

Population members run in parallel as columns of one batched brain state
(scipy CSR: W @ R with R (n, P)). Fitness = mean episode reward.
Elites -> Gaussian update (mu, sigma) with mild noise floor.

Anti-overfit: training seeds come from a training stream only; the frozen
EVAL_SEEDS in flybrain.game are never used here (eval.py owns them).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from flybrain import RUNS_DIR, CIRCUIT_PATH
from flybrain.brain import BrainModel as BatchedBrain
from flybrain.game import OBS_DIM, CatchEnv, observation
from flybrain.readout import make_sensory_projection

N_ACTIONS = 3


def calibration_rollout(brain: "BatchedBrain", W_s: np.ndarray, sensory_gain: float,
                        n_substeps: int, steps: int = 120, batch: int = 16,
                        seed: int = 123) -> np.ndarray:
    """Seeded random-action rollout (training-stream seeds only, NOT eval seeds).
    Returns feature matrix (n_neurons, steps*batch) of rates collected per decision."""
    from flybrain.game import CatchEnv, observation
    rng = np.random.default_rng(seed)
    brain.init_batch(batch)
    brain.reset_batch()
    envs = [CatchEnv(int(rng.integers(0, 8000))) for _ in range(batch)]
    cols = []
    obs = np.stack([observation(e) for e in envs])
    for _ in range(steps):
        S = np.maximum(W_s @ obs.T, 0.0) * sensory_gain
        brain.clamp_sensory_batch(S)
        brain.step_batch(n_substeps)
        cols.append(brain.r.copy())                              # (n, batch)
        for e in envs:
            e.step(int(rng.integers(0, 3)))
        obs = np.stack([observation(e) for e in envs])
    brain.init_batch(1)
    return np.concatenate(cols, axis=1)                      # (n, steps*batch)


def select_features(brain: "BatchedBrain", W_s: np.ndarray, use_sensory: bool,
                    sensory_gain: float, n_substeps: int, dn_topk: int | None,
                    seed: int = 123) -> np.ndarray:
    """Feature index: top-k DN by rate variance over the calibration rollout + sensory."""
    if dn_topk is not None and len(brain.dn_idx) > dn_topk:
        R = calibration_rollout(brain, W_s, sensory_gain, n_substeps, seed=seed)
        var = R.var(axis=1)
        dn = brain.dn_idx
        top = dn[np.argsort(var[dn])[::-1][:dn_topk]]
        parts = [np.sort(top)]
    else:
        parts = [brain.dn_idx.detach().cpu().numpy()]
    if use_sensory:
        parts.append(brain.sensory_idx)
    idx = np.unique(np.concatenate(parts))
    return idx.astype(np.int64)


def batch_actions(policies: np.ndarray, F: np.ndarray) -> np.ndarray:
    """policies (P, A*n_feat), features (P, n_feat) -> actions (P,)."""
    P, A, n_feat = len(policies), N_ACTIONS, F.shape[1]
    W = policies.reshape(P, A, n_feat)
    return np.argmax(np.einsum("pf,paf->pa", F, W), axis=1)


def episode_fitness(brain: BatchedBrain, W_s: np.ndarray, policies: np.ndarray,
                    seeds: list[int], feat_idx: np.ndarray, sensory_gain: float,
                    n_substeps: int, train_balls: int = 20) -> np.ndarray:
    """One episode per policy, all P in parallel. Returns per-policy total reward."""
    P = len(policies)
    envs = [CatchEnv(s, balls=train_balls) for s in seeds]
    brain.reset_batch()
    rewards = np.zeros(P, np.float64)
    obs = np.stack([observation(e) for e in envs])                      # (P, OBS)
    while True:
        S = np.maximum(W_s @ obs.T, 0.0) * sensory_gain                  # (n_sens, P)
        brain.clamp_sensory_batch(S)
        brain.step_batch(n_substeps)
        feats = brain.r[feat_idx].T                                     # (P, n_feat)
        acts = batch_actions(policies, feats)
        done_all = True
        for p, e in enumerate(envs):
            r, done = e.step(int(acts[p]))
            rewards[p] += r
            done_all &= done
        if done_all:
            return rewards
        obs = np.stack([observation(e) for e in envs])


def train(iters: int = 16, pop: int = 64, elites: int = 8, eps: int = 2, seed: int = 0,
          use_sensory: bool = True, sensory_gain: float = 1.0, n_substeps: int = 4,
          gain: float = 1.0, dn_topk: int | None = 128, sigma_decay: float = 0.9,
          sigma_floor: float = 0.02, train_balls: int = 20, elitism: bool = False,
          out: Path | None = None,
          graph_path: Path | None = None, quiet: bool = False):
    graph = dict(np.load(graph_path or CIRCUIT_PATH, allow_pickle=True))
    brain = BatchedBrain(graph, dt=0.005, gain=gain)
    brain.init_batch(pop)
    W_s = make_sensory_projection(len(brain.sensory_idx), OBS_DIM)
    feat_idx = select_features(brain, W_s, use_sensory, sensory_gain, n_substeps, dn_topk)
    brain.init_batch(pop + 1 if elitism else pop)
    n_feat = len(feat_idx)
    rng = np.random.default_rng(seed)
    K = n_feat * N_ACTIONS
    mu = np.zeros(K, np.float32)
    sigma = np.full(K, 0.5, np.float32)
    mu_tail: list[np.ndarray] = []
    tail = max(1, iters // 6)          # average mu over the last iters/6 generations
    t0 = time.time()
    history = []
    for gen in range(iters):
        samples = rng.normal(0, 1, (pop, K)).astype(np.float32) * sigma + mu
        if elitism:
            samples = np.concatenate([samples, mu[None]])      # current mean competes
        fits = np.zeros(len(samples))
        for _ in range(eps):
            seeds = [int(rng.integers(0, 8000)) for _ in range(len(samples))]
            fits += episode_fitness(brain, W_s, samples, seeds, feat_idx, sensory_gain,
                                    n_substeps, train_balls)
        fits /= eps
        order = np.argsort(fits)[::-1]
        elite = samples[order[:elites]]
        mu = elite.mean(0) if not elitism else (
            lambda w: (w[:, None] * elite).sum(0))(
                (lambda x: x / x.sum())(np.log(elites + 0.5) - np.log(np.arange(1, elites + 1))))
        sigma = elite.std(0) * sigma_decay + sigma_floor
        history.append({"gen": gen, "best": float(fits[order[0]]), "mean": float(fits.mean()),
                        "elite_mean": float(fits[order[:elites]].mean()), "sigma": float(sigma.mean()),
                        "seconds": time.time() - t0})
        mu_tail.append(mu.copy())
        if len(mu_tail) > tail:
            mu_tail.pop(0)
        if not quiet:
            print(f"gen {gen:3d} best {fits[order[0]]:7.2f} elite {history[-1]['elite_mean']:7.2f} "
                  f"mean {fits.mean():7.2f} sigma {sigma.mean():.3f} {history[-1]['seconds']:5.1f}s", flush=True)
    if out:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "mu.npy", mu)
        np.save(out / "sigma.npy", sigma)
        np.save(out / "feat_idx.npy", feat_idx)
        (out / "history.json").write_text(json.dumps(history, indent=1))
        (out / "config.json").write_text(json.dumps(
            {"use_sensory": use_sensory, "sensory_gain": sensory_gain, "n_substeps": n_substeps,
             "gain": gain, "dn_topk": dn_topk, "sigma_decay": sigma_decay,
             "sigma_floor": sigma_floor, "train_balls": train_balls,
             "pop": pop, "elites": elites, "eps": eps, "iters": iters, "seed": seed,
             "n_feat": n_feat, "n_dn": len(brain.dn_idx),
             "n_sensory": len(brain.sensory_idx), "graph": str(graph_path or CIRCUIT_PATH),
             "device": "cpu(scipy)"}, indent=1))
    mu = np.mean(mu_tail, axis=0).astype(np.float32)   # tail-averaged final policy
    return mu, sigma, history


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=16)
    p.add_argument("--pop", type=int, default=64)
    p.add_argument("--elites", type=int, default=8)
    p.add_argument("--eps", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-sensory", action="store_true")
    p.add_argument("--sensory-gain", type=float, default=1.0)
    p.add_argument("--substeps", type=int, default=4)
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--dn-topk", type=int, default=128)
    p.add_argument("--dn-all", action="store_true", help="read out from every DN")
    p.add_argument("--graph", type=Path, default=None)
    p.add_argument("--out", type=Path, default=RUNS_DIR / "cem_v0")
    a = p.parse_args()
    train(iters=a.iters, pop=a.pop, elites=a.elites, eps=a.eps, seed=a.seed,
          use_sensory=not a.no_sensory, sensory_gain=a.sensory_gain, n_substeps=a.substeps,
          gain=a.gain, dn_topk=None if a.dn_all else a.dn_topk, out=a.out, graph_path=a.graph)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
