# flybrain — Plan / Handoff

> A real fruit-fly brain (MaleCNS v1.0 connectome) that trains to play a game,
> with real-time 3D brain-activity rendering. Autoresearch-optimized.
>
> **Status**: not started — blocked on plan mode. This document is the full handoff.

## 1. Objective

Build a `flybrain` project in `/Users/ralim1/Documents/personal/projects/flybrain`:

1. **Dataset**: MaleCNS v1.0 (Janelia FlyEM, https://male-cns.janelia.org/download/) — bulk feather tables from `storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`.
2. **Brain**: a rate-model brain wired from the real connectome, trained (readout only, circuit frozen) to play a game.
3. **3D viz**: real-time brain-activity rendering on neuron skeletons inside the brain shell (adapted from `ref-projects/TMNF-C/python/tmnf_fly/brain_render.py`).
4. **Training method**: **CEM on a frozen circuit's descending-neuron readout**. NO dopamine/RPE plasticity — the user's previous flappy-bird dopamine attempt failed, and FlyPong published a negative result for dopamine-style learning on frozen connectomes; CEM readouts demonstrably work (Fly Dino: frozen 80-neuron circuit + 243-param CEM readout).
5. **Git**: commit at every milestone, **no co-author trailer, no message body** (subject line only).

## 2. Game decision

**Chosen: A) pong-catch** (ball falls, paddle catches) — continuous visual tracking,
easiest for CEM; avoids flappy's failure mode.
Alternatives considered: B) dino-runner (timing/discrete events), C) lunar-lander (continuous control, hardest).

## 3. Architecture

```
flybrain/
  __init__.py          package root, paths (data/ cache layout)
  fetch_data.py        pinned-sha256 download of MaleCNS tables + brain-shell mesh
  graph.py             signed row-normalized sparse CSR connectome (Traced neurons only)
  brain.py             leaky rate model: tau dv/dt = -v + g*W*relu(v) + I; sensory current injection
  game.py              pong-catch env: sensory encoding (retina sheet + paddle proprioception)
  readout.py           action readout from descending neurons (linear, CEM-trained)
  train.py             CEM training loop (frozen circuit, train seeds only)
  eval.py              held-out evaluation (frozen eval seeds, disjoint from training)
  skeletons.py         SWC skeleton download/parse/npz cache (from TMNF-C skeletons.py)
  render.py            3D activity renderer (from TMNF-C brain_render.py): polylines
                       colored by per-neuron activity, brain shell, off-screen pyvista
  play.py              end-to-end demo: run trained brain, record game+brain MP4
  README.md
data/                  gitignored: malecns feather tables, graph cache, skeletons, meshes, videos
docs/PLAN.md           this file
.auto/                 autoresearch session (prompt.md, measure.sh, log.jsonl, ideas.md)
```

### Data details (from ref-projects/TMNF-C/python/tmnf_fly/fetch_data.py — verified pinned URLs)

| file | size | sha256 |
|---|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14,483,314 | `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` |
| `body-neurotransmitters-male-cns-v1.0.feather` | 43,282,834 | `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621` |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1,051,241,946 | `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1` |
| `JRCFIB2022M_brain.ply` (brain shell, from `rois/pointcloud-shells/`) | 1,255,587 | `13ea41a7ce2677eae00738c23b2be3c2f29ac17c6976c2209998a85efccaa427` |

Base URLs:
- connectome tables: `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`
- brain shell: `https://storage.googleapis.com/flyem-male-cns/rois/pointcloud-shells/`
- neuron skeletons (SWC, 8 nm voxel units): `https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/{body_id}.swc`

### Graph construction (from TMNF-C brain_graph.py, verified)

- Keep `status == "Traced"` neurons (~138k), sort by bodyId.
- Sign: consensus neurotransmitter — `gaba/glutamate/histamine` → −1, else +1 (incl. `unclear`).
- Edge filter: synapse count >= 2, both ends Traced.
- Strength: `log1p(count) * sign[pre]`, row-normalized (post-synaptic) by total absolute input.
- CSR: `indptr/indices/strength`, plus masks: optic (superclass in optic set), descending (`superclass == "descending_neuron"`), side (somaSide, fallback rootSide).
- Optionally build a central-brain-only subgraph (drop optic superclasses) for fast game stepping; keep DN + sensory.
- Verify against known stats: ~166,700 total neurons in release; TMNF-C prints neuron/edge counts.

### Brain model (from TMNF-C brain_model.py, verified)

- `tau_i dv_i/dt = -v_i + g * (W r)_i + I_i`, `r = relu(v)`, Euler at dt=1 ms.
- Per-superclass tau: optic 10 ms, central 20 ms, descending 20 ms.
- Sensory input = external current on chosen sensory neuron populations.
- Torch sparse CSR matmul; device mps > cuda > cpu.
- Action readout: linear layer from descending-neuron rates (3 actions: left/stay/right).

### Game: pong-catch

- Arena W×H; ball(s) fall with random x, speed; paddle at bottom, 3 actions (left/stay/right).
- Sensory encoding: coarse retina grid (e.g., 16×12, ball as bright dot, paddle row) + paddle x proprioception.
- Reward: +1 per catch, −1 per miss; episode = N balls.
- Seeds: training seeds (randomized each CEM generation) vs **frozen eval seeds** (disjoint, e.g. seeds 9000-9099) — never used for training.

### Training: CEM (frozen circuit)

- Population of readout weight vectors; elite selection on train-seed episodes; noise decay.
- The connectome W is FROZEN — only the DN readout (and optionally input projection gains) trains.
- Budget: minutes-scale per iteration of the autoresearch loop (small circuit + short CEM).

## 4. Autoresearch session

- Branch: `autoresearch/flybrain-catch-<date>`
- `.auto/prompt.md`: full playbook (objective, metrics, how to run, files in scope, off-limits, constraints, what's been tried).
- `.auto/measure.sh`:
  - Fast prechecks (py_compile on package).
  - Train on train seeds (fixed budget) → eval on frozen eval seeds.
  - Emit `METRIC eval_reward=<mean held-out episode reward>` (primary, higher better),
    `METRIC train_reward=...`, `METRIC eval_train_gap=...`, `METRIC catch_rate=...`, `METRIC train_seconds=...`.
- **Anti-overfit/anti-cheat rules** (user-mandated):
  - Eval seeds frozen in repo before baseline; never used in training or model selection besides the primary metric.
  - No eval-seed-specific branching; no reward shaping that leaks eval seeds.
  - Train/eval gap monitored; keep requires held-out improvement, not train-fit.
- `init_experiment(name="flybrain catch training", metric_name="eval_reward", direction="higher")` → baseline run → `log_experiment` → loop.
- Ideas that don't fit an iteration go to `.auto/ideas.md`.
- keep/discard per autoresearch rules; every log_experiment carries `asi` annotations.

## 5. Milestones (commit each, subject-line-only messages)

1. `chore: scaffold repo, venv, deps` — git init, uv venv (py3.12), torch/pandas/pyarrow/pyvista/imageio/ffmpeg, README stub.
2. `feat: fetch malecns data` — fetch_data.py; download + verify (1.1 GB one-time).
3. `feat: build connectome graph` — graph.py + cached npz + stats printout.
4. `feat: brain rate model` — brain.py + smoke test (activity propagates, inhibition works).
5. `feat: pong-catch game` — game.py + eval.py seeds protocol.
6. `feat: CEM training` — readout.py + train.py; baseline trained artifact.
7. `feat: skeleton + 3d activity renderer` — skeletons.py, render.py; sample MP4 of brain activity during gameplay.
8. `feat: end-to-end play demo` — play.py composites game + brain activity video; README final.
9. `chore: autoresearch session` — .auto/prompt.md, .auto/measure.sh, branch, init_experiment, baseline log.
10. Then: **loop forever** (autoresearch rules).

## 6. Environment facts (verified on this machine)

- macOS, Apple Silicon; 127 GB free disk (1.1 GB dataset fits).
- System python3 = 3.14.6 (homebrew) — NO scientific stack; use `uv` (`~/.local/bin/uv`) with python 3.12 venv.
- ref-projects/TMNF-C/python/tmnf_fly/ = proven reference implementation (fetch, graph, brain model, renderer).
- ref-projects/awesome-fly/README.md = survey of prior art; key evidence:
  - FlyPong (jonatasperaza): dopamine-inspired plasticity → documented NEGATIVE result.
  - Fly Dino (cobanov/flyjump): frozen 80-neuron MaleCNS circuit + CEM-trained readout → works, reproducible.
  - FLYT3, fly-craftax: REINFORCE/ppo-trained readouts on frozen circuit → work.
  - Doomfly: negative validation results for learned survival.
- Dataset license: CC-BY (attribution: Janelia FlyEM / MaleCNS).

## 7. Risks / fallbacks

- Full 138k-neuron model may be too slow for CEM → use central-brain subgraph (~20-40k) or DN-relevant k-hop subgraph around sensory→DN paths.
- CEM may stall → grow population, add input-projection training, or escalate to REINFORCE on readout (still frozen circuit).
- Renderer needs GL context (EGL/GLX) — macOS offscreen pyvista may need osmesa or screen capture; fallback: render with matplotlib 3D or polyscope; or render frames on-screen via pv.Plotter(show=True) capture.
- SWC downloads for rendered population (~5-10k neurons) — one-time, parallel.

## 8. Blocked on

- **Plan mode active**: all mutating commands blocked. Toggle off (shift+tab) to begin execution at Milestone 1.
