"""Rate-based dynamics over the MaleCNS circuit (numpy/scipy, CPU).

    tau_i dv_i/dt = -v_i + g * sum_j W_ij r_j,    r = relu(v)

W signed/row-normalised from the real connectome (frozen). tau per superclass.
Sensory rows are rate-clamped to the game's sensory code each decision.
scipy CSR matmul: ~2-3 ms per substep at batch 65, deterministic, no GPU variance.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

TAU_MS = {"optic": 10.0, "central": 20.0, "descending": 20.0}


class BrainModel:
    def __init__(self, graph: dict[str, np.ndarray], dt: float = 0.005, gain: float = 1.0,
                 tau_ms: dict[str, float] | None = None):
        self.dt = float(dt)
        n = self.n = int(len(graph["body_id"]))
        rows = np.repeat(np.arange(n), np.diff(graph["indptr"]))
        self.A = sp.csr_matrix((graph["strength"].astype(np.float32),
                                (rows, graph["indices"].astype(np.int64))), shape=(n, n)).tocsr()
        self.edges = int(self.A.nnz)
        self.gain = float(gain)
        tau_ms = tau_ms or TAU_MS
        tau = np.full(n, tau_ms["central"], np.float32)
        tau[graph["optic"]] = tau_ms["optic"]
        tau[graph["descending"]] = tau_ms["descending"]
        self.decay = (dt / (tau / 1000.0)).astype(np.float32)[:, None]  # (n,1) broadcasts over batch
        self.body_id = graph["body_id"]
        self.types = graph["type"]
        self.superclass = graph["superclass"]
        self.populations = {k[len("pop/"):]: v for k, v in graph.items() if k.startswith("pop/")}
        self.sensory_idx = self.populations["sensory"]
        self.dn_idx = self.populations["descending"]
        self.v = np.zeros(n, np.float32)
        self.r = np.zeros(n, np.float32)
        self.t = 0.0

    # -- single-brain API (renderer / play) ----------------------------------
    def reset(self) -> None:
        self.v[:] = 0.0
        self.r[:] = 0.0
        self.t = 0.0

    def activity(self) -> np.ndarray:
        """Per-neuron activity in [0, 1] for the renderer (quantile-scaled relu rates)."""
        r = self.r if self.r.ndim == 1 else self.r[:, 0]   # batched use: member 0
        hi = np.quantile(r, 0.999) if r.size else 1.0
        return np.clip(r / (hi if hi > 0 else 1.0), 0.0, 1.0).astype(np.float32)

    def clamp_sensory(self, values: np.ndarray) -> None:
        values = np.asarray(values, np.float32)
        self.r[self.sensory_idx] = values
        self.v[self.sensory_idx] = values

    def step(self, n_substeps: int = 1) -> None:
        for _ in range(n_substeps):
            drive = self.A @ self.r
            self.v += self.decay[:, 0] * (self.gain * drive - self.v)
            self.r = np.maximum(self.v, 0.0)
            self.t += self.dt

    # -- batched API (CEM training / evaluation) ------------------------------
    def init_batch(self, batch: int) -> None:
        self.batch = int(batch)
        self.v = np.zeros((self.n, self.batch), np.float32)
        self.r = np.zeros_like(self.v)

    def reset_batch(self) -> None:
        self.v[:] = 0.0
        self.r[:] = 0.0
        self.t = 0.0

    def clamp_sensory_batch(self, S: np.ndarray) -> None:
        """S: (n_sensory, P) rates, written into r and v rows of sensory neurons."""
        if S.shape != (len(self.sensory_idx), self.batch):
            raise ValueError(f"expected ({len(self.sensory_idx)}, {self.batch}), got {S.shape}")
        self.r[self.sensory_idx] = S
        self.v[self.sensory_idx] = S

    def step_batch(self, n_substeps: int = 1) -> None:
        for _ in range(n_substeps):
            drive = self.A @ self.r
            self.v += self.decay * (self.gain * drive - self.v)
            self.v[self.sensory_idx] = self.r[self.sensory_idx]
            self.r = np.maximum(self.v, 0.0)
            self.t += self.dt
