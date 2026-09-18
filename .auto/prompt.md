# Autoresearch: flybrain — MaleCNS circuit learns pong-catch

## Objective

A frozen MaleCNS v1.0 sensorimotor circuit (`data/fly/brain_circuit.npz`, 4,706 neurons:
300 sensory-sample + all 1,314 descending neurons + every neuron on a 1-hop path
sensory→DN, induced edges re-normalised) drives a leaky rate model. A linear readout
over brain activity (DN rates + clamped sensory rates by default) picks one of 3
actions in pong-catch (ball falls, paddle catches). **CEM trains the readout only;
the connectome is frozen.** Optimize held-out game performance.

## Metrics

- **Primary**: `eval_reward` — mean episode reward on the FROZEN eval seeds
  (game.EVAL_SEEDS = 9000..9095, +1/catch −1/miss, 20 balls; chance ≈ −6.7, perfect = +20). Higher is better.
- Secondary: `catch_rate`, `train_reward` (elite fitness in training), `gap`
  (eval − train: generalization), `train_seconds` (budget cost), `hist_last_best`.

## How to Run

`./.auto/measure.sh` — prechecks, trains via `.auto/config.json` settings, evaluates
on frozen eval seeds, emits `METRIC` lines. ~3 min/run at baseline budget
(iters 24 × pop 48 × eps 1).

**Tuning protocol**: edit `.auto/config.json` (committed) to change the training
hyperparameters being studied. Code changes to flybrain/*.py are also in scope.
The measure script itself must stay a fixed protocol (train on training stream →
eval on frozen seeds).

## Files in Scope

- `.auto/config.json` — training hyperparameters under study (iters/pop/elites/eps,
  use_sensory, sensory_gain, n_substeps, gain, seed, graph path)
- `flybrain/train.py` — CEM loop, fitness batching
- `flybrain/brain.py` — rate model (dt, gain, tau, substeps, clamping)
- `flybrain/readout.py` — sensory projection (frozen), policy readout
- `flybrain/game.py` — env + OBS encoding + EVAL_SEEDS (frozen!)
- `flybrain/graph.py` — circuit extraction (rebuild with `--circuit`, hops param)

## Off Limits

- `flybrain/game.py` `EVAL_SEEDS` and `observation()` — frozen evaluation protocol
- `.auto/measure.sh` evaluation path — no eval-seed leakage, no reward shaping toward eval
- `data/` derived caches can be rebuilt but never hand-edited

## Constraints

- The connectome W stays FROZEN (no training of synapse weights — that's the project's
  scientific constraint; readout/input-gain only)
- No eval seed may influence training, selection, or early stopping (user's
  no-overfit/no-cheat mandate; watch `gap`)
- Training budget ≤ ~5 min per run (measure wall time `train_seconds`)
- Deps fixed (torch/pandas/pyarrow/pyvista/scipy installed in .venv)

## What's Been Tried

- **Baseline (run 1)**: iters 24, pop 48, eps 1, substeps 4, gain 1.0,
  use_sensory=True, sensory_gain 1.0, hops=1 circuit.
  eval_reward −6.67, catch 33% (≈chance), train elite −2.5, gap −4.2, 180 s.
  Training clearly fits (best gen policies reach +2) but held-out doesn't follow yet —
  readout overfits the sampled sensory projection. Next: more CEM generations/eps
  averaging, noise decay, sensory_gain sweep, feature scaling.
- Environment notes: MPS dense (no sparse CSR on MPS); hops=2 circuit exists
  (39.9k neurons, `data/fly/brain_circuit_h2.npz`) but dense MPS would need ~6.4 GB —
  try only with subsampling or fp16.
