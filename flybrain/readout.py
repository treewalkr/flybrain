"""Frozen sensory projection + linear readout, trained with CEM (train.py).

The connectome W is frozen. Two trainable parts, both linear, both CEM-fit:

  W_s: (n_sensory, OBS_DIM) fixed sparse random projection (FROZEN, seeded) maps
       the game observation onto the brain's sensory neurons (clamped rates).
  readout: (3, n_features) linear map from brain activity to action scores
       (features = descending-neuron rates, optionally + sensory rates).

Only the readout trains. The brain's job is to transform the sensory code into
rich DN activity through real fly wiring; the readout picks the action.
"""
from __future__ import annotations

import numpy as np


def make_sensory_projection(n_sensory: int, obs_dim: int, degree: int | None = None, seed: int = 7) -> np.ndarray:
    """Fixed sparse random projection obs -> sensory rates, values in {+1,-1}/degree.

    degree defaults to FLY_PROJ_DEGREE (or 4) so the projection DENSITY
    (degree/obs_dim) stays constant when the retina size changes.
    """
    import os
    if degree is None:
        degree = int(os.environ.get("FLY_PROJ_DEGREE", 4))
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n_sensory), degree)
    cols = rng.integers(0, obs_dim, size=n_sensory * degree)
    vals = rng.choice([-1.0, 1.0], size=n_sensory * degree) / np.sqrt(degree)
    W = np.zeros((n_sensory, obs_dim), np.float32)
    np.add.at(W, (rows, cols), vals)
    return W


class Policy:
    """Linear readout over brain features; params flattened for CEM."""

    def __init__(self, n_features: int, n_actions: int = 3, params: np.ndarray | None = None, seed: int = 0):
        self.n_features, self.n_actions = n_features, n_actions
        self.k = n_features * n_actions
        if params is None:
            rng = np.random.default_rng(seed)
            self.w = rng.normal(0, 0.1, self.k).astype(np.float32)
        else:
            self.w = np.asarray(params, np.float32).reshape(-1)
            assert self.w.size == self.k

    def actions(self, features: np.ndarray) -> np.ndarray:
        """features (P, n_features) -> (P,) action indices."""
        W = self.w.reshape(self.n_actions, self.n_features)   # (A, F)
        return np.argmax(features @ W.T, axis=1)
