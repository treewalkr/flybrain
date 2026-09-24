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

Best recipe (run 23): **iters 48, pop 64, elites 10, eps 3, seed 0, dn_topk 256,
sigma_decay 0.9, sigma_floor 0.02, train_balls 12, substeps 4, gain 1.0,
tail-averaged mu (last iters/4 gens) → eval +2.31, catch 55.8%** (baseline −6.67, chance 33%).

Wins, in order of impact:
- **dn_topk 256** (top-256 DN by variance) vs 128: +1.69 → biggest single lever
- **48 CEM generations** vs 24: −1.19 → +1.69 line of improvements
- **tail-averaged mu** (last 12 gens): +2.31, gap turned positive
- **12-ball training episodes + eps 3**: first positive results
- scipy/numpy brain core: determinism (bit-identical reruns, zero noise floor)

Dead ends (do not revisit without changed assumptions):
- Elitism + rank-weighted recombination: amplifies winner's curse (−5.79 at full budget)
- Cross-seed mu ensembling: averaging policies from independent trajectories destroys both (−6.58)
- Feature standardization (run 2): −13.1
- Shrink circuit below ~3k neurons (mc800): degenerate stuck-left policy (−13.85); middle-path richness is essential
- substeps < 4: coarser integration degrades policy (−2.10 at 2, −3.98 at 3)
- pop 32 (width matters more than gens): −6.85; eps 5 + fewer gens: −5.54
- sigma_decay 0.85: premature convergence (−7.38); dn_topk 512: stationary collapse (−7.85)
- brain gain 1.4: destabilized dynamics (−5.71); sensory_gain 1.5: bit-identical (linear regime)
- Tail selection (best-of-tail on fresh train episodes): +2.21 < averaging's +2.31

Caveats:
- Trajectory (seed) variance is ±2–3 reward — single A/B runs are weak evidence for
  small deltas; strong effects (≥3) are trustworthy. Across seeds the champion
  recipe scores −3.6…+2.3 reward, 41–56% catch (chance 33%): catch rate is the
  stable metric; seed replicas logged as informational discards (runs 19, 33)
- Machine load varies wildly (load avg 4–28); wall time 12–50 min, results unaffected
- use_sensory=True keeps a direct sensory→readout path; **DN-only ablation (run 43): connectome
  alone reaches 46% catch (chance 33%) — the real fly wiring genuinely carries the task;
  the sensory shortcut adds the top end (69% champion)**
- Closed directions (do not reopen): elitism, cross-seed averaging, feature scaling,
  circuit shrink, substeps<4, substeps>4 (run 72: transfer flat +0.04, brain settles
  within 4 substeps), pop shrink, sigma_decay 0.85, dn_topk 512, zero-padded top-512
  expansion at the optimum (run 73: both draws below gate), gain 1.4, sensory_gain,
  tau 10ms, n_sensory 600, min-fitness, elites 16, iters 60, balls-24 episodes,
  margin-argmax/hysteresis decoding (run 74: monotone val decay with margin —
  the 2%-stay bang-bang dithering is FUNCTIONAL, not pathology),
  retina 24/32 (runs 75/44/45/55: 10 fresh draws across widths cap ~50% from
  scratch; NOT the 387>300 squeeze — retina-24 fits 291<=300 — and NOT density;
  something intrinsic to wide retinas breaks from-scratch CEM)
- Miss-mode diagnosis (.auto/miss_probe.py, .auto/landing_probe.py, .auto/osc_bias_probe.py):
  misses are diffuse — uniform across step-quartiles (86-88%), bounce counts, speeds;
  52% of misses are MARGINAL (landing 0.10-0.15 from paddle vs 0.10 window); on missed
  balls the paddle's 10-step MEDIAN position is 0.156 off the true landing (only 6/156
  within 0.05) → ESTIMATE BIAS, not oscillation phase: the readout settles at a wrong
  target. Action-feedback/hysteresis mechanisms are dead; finer input untrainable (see
  retina closures). The ~0.15 target error on ~16% of balls is the frozen linear
  readout's estimate precision — final attribution
- DN-only fair fight (run 46): val-selected connectome-only policy = 54.2% catch
  vs 86.2% full — the sensory skip path adds ~32pts; connectome alone beats chance
  (33%) by 21pts
- FINAL STATE: converged at champ_lin (+14.44/86.1%) with champ_lin2 as statistical
  twin (+14.48/86.2%); val noise floor ±1.1 (48-ep scores); every axis closed;
  only parked low-EV idea is a fully retuned wide-retina projection (fresh axis)
