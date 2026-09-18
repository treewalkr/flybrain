"""Build the signed sparse connectome graph from the MaleCNS v1.0 feather tables.

    .venv/bin/python -m flybrain.graph            # full graph -> data/fly/brain_graph.npz
    .venv/bin/python -m flybrain.graph --sub      # also build the game subgraph

Full graph: every Traced neuron (~138k). Sign from consensus neurotransmitter
(gaba/glutamate/histamine inhibitory). Edge strength log1p(synapses)*sign(pre),
rows (postsynaptic) normalised to unit absolute input. Sparse CSR.

Subgraph (--sub): central-brain circuit for the game — excludes optic-lobe
superclasses but keeps a sample of sensory inputs and all descending neurons
reachable from the kept core, so the brain has inputs and outputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from flybrain import FLY_DIR, GRAPH_PATH, MALECNS_DIR, CIRCUIT_PATH

MIN_WEIGHT = 2
INHIBITORY = ("gaba", "glutamate", "histamine")
# MaleCNS v1.0 superclass vocabulary (verified against the annotations table):
OPTIC_SUPERCLASSES = ("ol_intrinsic", "ol_sensory", "visual_projection", "visual_centrifugal")
SENSORY_SUPERCLASSES = ("ol_sensory", "cb_sensory", "vnc_sensory", "vnc_sensory_tbc",
                        "cb_sensory_tbc", "sensory_ascending")
BRAIN_SUPERCLASSES = ("cb_intrinsic", "cb_sensory", "cb_motor", "cb_endocrine", "cb_efferent",
                       "cb_sensory_tbc", "sensory_ascending", "ascending_neuron",
                       "descending_neuron", "sensory_descending", "efferent_descending",
                       "descending_neuron_tbc", "efferent_ascending", "visual_projection",
                       "visual_centrifugal", "ol_intrinsic", "ol_sensory")  # brain proper + optic lobes


def combined_side(ann: pd.DataFrame) -> np.ndarray:
    side = ann.somaSide.astype(object).where(ann.somaSide.notna(), ann.rootSide.astype(object))
    side = side.where(side.isin(["L", "R", "M"]), "")
    return side.to_numpy(str)


def build_full(data: Path = MALECNS_DIR, min_weight: int = MIN_WEIGHT) -> dict[str, np.ndarray]:
    ann = feather.read_feather(data / "body-annotations-male-cns-v1.0-minconf-0.5.feather")
    nt = feather.read_feather(data / "body-neurotransmitters-male-cns-v1.0.feather")
    keep = ann[ann.status == "Traced"].sort_values("bodyId").reset_index(drop=True)
    body = keep.bodyId.to_numpy(np.int64)
    n = len(body)
    types = keep.type.fillna("").to_numpy(str)
    superclass = keep.superclass.fillna("unknown").to_numpy(str)
    side = combined_side(keep)
    consensus = nt.set_index("body").consensus_nt.reindex(body).fillna("unclear").to_numpy(str)
    sign = np.where(np.isin(consensus, INHIBITORY), -1.0, 1.0).astype(np.float32)
    optic = np.isin(superclass, OPTIC_SUPERCLASSES)
    sensory = np.isin(superclass, SENSORY_SUPERCLASSES)

    weights = feather.read_feather(data / "connectome-weights-male-cns-v1.0-minconf-0.5.feather")
    total_rows = len(weights)
    index = pd.Index(body)
    pre = index.get_indexer(weights.body_pre.to_numpy(np.int64))
    post = index.get_indexer(weights.body_post.to_numpy(np.int64))
    count = weights.weight.to_numpy(np.float32)
    del weights
    traced_pair = (pre >= 0) & (post >= 0)
    traced_edges = int(traced_pair.sum())
    keep_edge = traced_pair & (count >= min_weight)
    pre, post, count = pre[keep_edge], post[keep_edge], count[keep_edge]
    order = np.lexsort((pre, post))
    pre, post, count = pre[order].astype(np.int64), post[order].astype(np.int64), count[order].astype(np.float32)

    strength = np.log1p(count) * sign[pre]
    total = np.zeros(n, np.float64)
    np.add.at(total, post, np.abs(strength))
    strength = (strength / np.where(total[post] > 0, total[post], 1.0)).astype(np.float32)
    indptr = np.zeros(n + 1, np.int64)
    np.add.at(indptr, post + 1, 1)
    indptr = np.cumsum(indptr)

    graph = dict(
        body_id=body, type=types, superclass=superclass, side=side,
        consensus_nt=consensus, sign=sign, optic=optic, sensory=sensory,
        descending=superclass == "descending_neuron",
        indptr=indptr, indices=pre, strength=strength, synapse_count=count,
        min_weight=np.int64(min_weight), total_weight_rows=np.int64(total_rows),
        traced_edges=np.int64(traced_edges),
    )
    # population index arrays for fast lookup
    for name in ("descending", "optic", "sensory"):
        graph[f"pop/{name}"] = np.flatnonzero(graph[name]).astype(np.int64)
    graph["pop/central"] = np.flatnonzero(~graph["optic"]).astype(np.int64)
    return graph


def induced_subgraph(graph: dict[str, np.ndarray], keep_nodes: np.ndarray) -> dict[str, np.ndarray]:
    """Node-induced subgraph: keep_nodes (sorted indices), edges between kept nodes re-normalised."""
    keep_nodes = np.sort(keep_nodes)
    n_old = len(graph["body_id"])
    new_index = np.full(n_old, -1, np.int64)
    new_index[keep_nodes] = np.arange(len(keep_nodes))
    indptr, indices, strength = graph["indptr"], graph["indices"], graph["strength"]
    rows = np.repeat(np.arange(n_old), np.diff(indptr))  # postsynaptic row of each edge
    sel = new_index[rows] >= 0  # rows kept
    sel &= new_index[indices] >= 0  # presynaptic kept
    sub_post, sub_pre = new_index[rows[sel]], new_index[indices[sel]]
    sub_strength = strength[sel].astype(np.float64)
    total = np.zeros(len(keep_nodes), np.float64)
    np.add.at(total, sub_post, np.abs(sub_strength))
    sub_strength = (sub_strength / np.where(total[sub_post] > 0, total[sub_post], 1.0)).astype(np.float32)
    order = np.lexsort((sub_pre, sub_post))
    sub_post, sub_pre, sub_strength = sub_post[order], sub_pre[order], sub_strength[order]
    sub_indptr = np.zeros(len(keep_nodes) + 1, np.int64)
    np.add.at(sub_indptr, sub_post + 1, 1)
    sub_indptr = np.cumsum(sub_indptr)

    sub = {"indptr": sub_indptr, "indices": sub_pre, "strength": sub_strength}
    for key in ("body_id", "type", "superclass", "side", "consensus_nt", "sign"):
        sub[key] = graph[key][keep_nodes]
    for key in ("descending", "optic", "sensory"):
        sub[key] = graph[key][keep_nodes]
        sub[f"pop/{key}"] = np.flatnonzero(sub[key]).astype(np.int64)
    sub["pop/central"] = np.arange(len(keep_nodes))  # meaningless in sub; kept for shape compat
    sub["parent_index"] = keep_nodes  # index into the full graph
    return sub


def build_circuit(graph: dict[str, np.ndarray], n_sensory: int = 300, hops: int = 5,
                  seed: int = 0, path: Path | None = None) -> dict[str, np.ndarray]:
    """Sensorimotor core: sensory sample, all DNs, and every neuron on a path
    sensory -k-hops-> X -k-hops-> DN. Small enough (order 10k) for dense torch on GPU.

    Reachability by iterative boolean sparse matvec on the edge list. Sensory sample
    and DNs are always kept even if not on a path.
    """
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    n = len(graph["body_id"])
    brain_mask = np.isin(graph["superclass"], BRAIN_SUPERCLASSES) & ~np.isin(
        graph["superclass"], ("ol_intrinsic", "ol_sensory"))
    sensory_pool = np.flatnonzero(brain_mask & graph["sensory"])
    sensory = np.sort(rng.choice(sensory_pool, size=min(n_sensory, len(sensory_pool)), replace=False))
    dn = graph["pop/descending"]

    post, pre = graph["indptr"], graph["indices"]
    rows = np.repeat(np.arange(n), np.diff(post))
    A = sp.csr_matrix((np.ones(len(rows), np.bool_), (rows, pre)), shape=(n, n))  # A[i,j]=1: j -> i
    fwd = np.zeros(n, np.bool_); fwd[sensory] = True
    bwd = np.zeros(n, np.bool_); bwd[dn] = True
    for _ in range(hops):
        fwd = fwd | (A @ fwd)
        bwd = bwd | (A.T @ bwd)
    on_path = fwd & bwd & brain_mask
    keep = np.unique(np.concatenate([np.flatnonzero(on_path), sensory, dn]))
    sub = induced_subgraph(graph, keep)
    sub["is_sensory_sample"] = np.isin(sub["body_id"], graph["body_id"][sensory])
    sub["is_dn"] = np.isin(sub["body_id"], graph["body_id"][dn])
    sub["hops"] = np.int64(hops)
    # sensory pop = the sampled neurons only; descending pop = DN set
    sub["sensory"] = sub["is_sensory_sample"]
    sub["pop/sensory"] = np.flatnonzero(sub["sensory"]).astype(np.int64)
    sub["pop/descending"] = np.flatnonzero(sub["is_dn"]).astype(np.int64)
    if path:
        np.savez(path, **sub)
    return sub


def summarize(graph: dict[str, np.ndarray], name: str) -> None:
    n = len(graph["body_id"])
    edges = len(graph["indices"])
    inh = int((graph["sign"] < 0).sum())
    no_in = int((np.diff(graph["indptr"]) == 0).sum())
    print(f"{name}: {n:,} neurons, {edges:,} edges, inhibitory {inh:,}, no-input {no_in:,}, "
          f"descending {int(graph['descending'].sum()):,}, optic {int(graph['optic'].sum()):,}, "
          f"sensory {int(graph['sensory'].sum()):,}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit", action="store_true", help="also build the sensorimotor core circuit")
    parser.add_argument("--hops", type=int, default=5)
    args = parser.parse_args()
    graph = build_full()
    summarize(graph, "full graph")
    FLY_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(GRAPH_PATH, **graph)
    print(f"-> {GRAPH_PATH}")
    if args.circuit:
        circ = build_circuit(graph, hops=args.hops, path=CIRCUIT_PATH)
        summarize(circ, "circuit")
        print(f"-> {CIRCUIT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
