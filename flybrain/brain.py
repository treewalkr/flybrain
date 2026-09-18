"""Rate-based dynamics over the MaleCNS subgraph (torch sparse CSR).

    tau_i dv_i/dt = -v_i + g * sum_j W_ij r_j + I_i,    r = relu(v)

W signed/row-normalised from the real connectome (frozen). tau per superclass
(optic 10 ms, central 20 ms, descending 20 ms). I is an external current.
`clamp` holds a population's rates at given values (they still drive their
targets through W) — used to inject the game's sensory code.
"""
from __future__ import annotations

import numpy as np
import torch

TAU_MS = {"optic": 10.0, "central": 20.0, "descending": 20.0}


def autodetect_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class BrainModel:
    def __init__(self, graph: dict[str, np.ndarray], device: str | torch.device | None = None,
                 dt: float = 0.005, gain: float = 1.0, tau_ms: dict[str, float] | None = None) -> None:
        self.device = torch.device(device) if device is not None else autodetect_device()
        self.dt = float(dt)
        n = self.n = int(len(graph["body_id"]))
        indptr = torch.from_numpy(graph["indptr"].astype(np.int64))
        indices = torch.from_numpy(graph["indices"].astype(np.int64))
        strength = torch.from_numpy(graph["strength"].astype(np.float32))
        if self.device.type == "mps":
            # MPS has no sparse CSR kernels; small circuits run dense (n^2 fp32).
            dense = torch.zeros(n, n)
            dense.index_put_((indices, torch.repeat_interleave(
                torch.arange(n), torch.from_numpy(np.diff(indptr)))), strength)
            self.w = dense.to(self.device)
            self.sparse = False
        else:
            self.w = torch.sparse_csr_tensor(indptr, indices, strength, size=(n, n)).to(self.device)
            self.sparse = True
        self.edges = int(len(indices))
        self.gain = float(gain)
        tau_ms = tau_ms or TAU_MS
        tau = np.full(n, tau_ms["central"], np.float32)
        tau[graph["optic"]] = tau_ms["optic"]
        tau[graph["descending"]] = tau_ms["descending"]
        self.tau = torch.from_numpy(tau).to(self.device) / 1000.0
        self.decay = self.dt / self.tau
        self.body_id = graph["body_id"]
        self.body_index = {int(b): i for i, b in enumerate(graph["body_id"])}
        self.types = graph["type"]
        self.superclass = graph["superclass"]
        self.populations = {k[len("pop/"):]: torch.from_numpy(v.astype(np.int64)).to(self.device)
                            for k, v in graph.items() if k.startswith("pop/")}
        self.sensory_idx = self.populations["sensory"]
        self.dn_idx = self.populations["descending"]
        self.v = torch.zeros(n, device=self.device)
        self.r = torch.zeros(n, device=self.device)
        self._clamp_dirty = False
        self._clamp_idx = torch.zeros(0, dtype=torch.int64, device=self.device)
        self._clamp_val = torch.zeros(0, device=self.device)
        self.t = 0.0

    # -- state -------------------------------------------------------------
    def reset(self) -> None:
        self.v.zero_()
        self.r.zero_()
        self.t = 0.0
        self.unclamp()

    def activity(self) -> np.ndarray:
        """Per-neuron activity in [0, 1] for the renderer (quantile-scaled relu rates)."""
        r = self.r.detach().cpu().numpy()
        hi = np.quantile(r, 0.999) if r.size else 1.0
        return np.clip(r / (hi if hi > 0 else 1.0), 0.0, 1.0).astype(np.float32)

    # -- clamping (sensory drive) ------------------------------------------
    def unclamp(self) -> None:
        self._clamp_idx = self._clamp_idx[:0]
        self._clamp_val = self._clamp_val[:0]
        self._clamp_dirty = False

    def clamp_sensory(self, values: torch.Tensor) -> None:
        """values: (n_sensory,) rates for the sensory population (game sensory code)."""
        values = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if values.shape != self.sensory_idx.shape:
            raise ValueError(f"expected ({len(self.sensory_idx)},) sensory values, got {tuple(values.shape)}")
        self._clamp_idx = self.sensory_idx
        self._clamp_val = values
        self._clamp_dirty = False

    # -- integration ---------------------------------------------------------
    @torch.no_grad()
    def step(self, n_substeps: int = 1) -> None:
        for _ in range(n_substeps):
            drive = self.w @ self.r.unsqueeze(1)
            drive = drive[:, 0] if self.sparse else drive
            self.v += self.decay * (self.gain * drive - self.v)
            self.v[self._clamp_idx] = self._clamp_val
            r = torch.relu(self.v)
            r[self._clamp_idx] = self._clamp_val
            self.r = r
            self.t += self.dt


class BatchedBrain(BrainModel):
    """P independent brains sharing the frozen W, as columns of (n, P) state tensors.

    torch.sparse.mm(W (n,n), R (n,P)) -> (n,P): one matmul per substep for the whole
    population; sensory rows are set directly (rate clamp) each decision.
    """

    def init_batch(self, batch: int) -> None:
        self.batch = int(batch)
        self.v = torch.zeros(self.n, self.batch, device=self.device)
        self.r = torch.zeros_like(self.v)

    def reset_batch(self) -> None:
        self.v.zero_()
        self.r.zero_()
        self.t = 0.0

    @torch.no_grad()
    def clamp_sensory_batch(self, S: torch.Tensor) -> None:
        """S: (n_sensory, P) rates, written into r and v rows of sensory neurons."""
        if S.shape != (len(self.sensory_idx), self.batch):
            raise ValueError(f"expected ({len(self.sensory_idx)}, {self.batch}), got {tuple(S.shape)}")
        self.r[self.sensory_idx] = S
        self.v[self.sensory_idx] = S

    @torch.no_grad()
    def step_batch(self, n_substeps: int = 1) -> None:
        for _ in range(n_substeps):
            drive = self.w @ self.r if not self.sparse else torch.sparse.mm(self.w, self.r)
            self.v += self.decay.unsqueeze(1) * (self.gain * drive - self.v)
            self.v[self.sensory_idx] = self.r[self.sensory_idx]
            r = torch.relu(self.v)
            self.r = r
            self.t += self.dt

