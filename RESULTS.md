# Results — MaleCNS circuit plays pong-catch

A frozen MaleCNS v1.0 connectome circuit (4,706 neurons: 300 sensory + 1,314
descending + all 1-hop interneurons between them, induced edges re-normalised)
drives a leaky rate model. Only a linear readout over brain activity is trained
(CEM); **the connectome is never modified**.

## Headline

| policy | eval reward | catch rate |
|---|---|---|
| chance (random actions) | −6.7 | 33% |
| connectome only, no sensory skip path (val-selected) | — | 54.2% |
| **full model — champion (`data/runs/champ_lin`)** | **+14.44** | **86.1%** |
| statistical twin (`champ_lin2`) | +14.48 | 86.2% |
| perfect play | +20 | 100% |

Evaluated on 96 frozen seeds (9000–9095), 20 balls each, never seen in training.

**The real fly wiring carries the task**: with the direct sensory→readout shortcut
removed and the readout restricted to descending-neuron rates, the circuit alone
reaches 54.2% catch — 21 points above chance. The sensory skip path contributes
the remaining ~32 points.

## The champion recipe

Found by a 73-experiment autoresearch loop (`.auto/log.jsonl`). Ladder of wins,
in order of impact:

1. **top-256 DN features** (variance-ranked descending neurons) — +1.7 vs top-128
2. **48+ CEM generations** (iters 96 pilot protocol, pop 64, elites 10, eps 6)
3. **champion-restart refinement ladder** (4 rounds: restart CEM at the best mu
   with sigma 0.10–0.15, fresh training episodes, val-gated) — 68.8% → 84.9%
4. **lineage averaging** (mean of refinement-ladder mus, weights 0.5/0.3/0.2) —
   84.9% → 86.1%, and it eliminated the hemifield cold-zone pathology: zone
   coverage is now uniform (worst zone 0.79, no dead cells)

Training: 12-ball episodes, 4 substeps × 5 ms per decision, 6 workers,
scipy/numpy core — bit-identical reruns (zero noise floor on the sim itself).

## Measurement honesty

- Validation noise floor is **±1.1 reward** on 48-episode scores (measured on
  disjoint blocks); gate discipline was audited against it — recent sub-gate
  margins (round 4, extended averages) were within noise and correctly not kept.
- `gap` (eval − val) ≈ −0.7 to −1.1: validation slightly over-predicts, no
  overfitting signature; fresh-never-used val block matches frozen eval.
- Trajectory (seed) variance across independent trainings is ±2–3 reward; the
  reported champion is a val-selected point, its twin confirms stability.

## What did not work (closed axes)

Elitism + rank weighting (winner's curse), cross-seed mu ensembling, feature
standardization, circuit shrink (<3k neurons → stuck-left degenerate policy),
substeps <4 and >4 (the rate dynamics settle to their input-driven fixed points
within 4 substeps — transfer to 6 is flat at +0.04), pop 32, sigma_decay 0.85,
dn_topk 512 from scratch, zero-padded top-512 capacity expansion at the optimum
(ranks 257–512 carry no climbable signal), brain gain 1.4, sensory gain (linear
regime — bit-identical), tau 10 ms, retina 32 (encoding dilution), 24-ball
episodes, coverage/maximin selection (≡ aggregate selection), ensembles beyond
the lineage average, margin-argmax decoding, landing-correlation feature
selection (augmentation and from-scratch replacement both fail: the criterion
is immaterial, from-scratch MI +8.25/71% vs variance +8.42/68.8%),
per-channel input gains (relu clipping is active — 40.8% of DN voltages clip —
so gains genuinely reshape the computation, and CEM actively uses them; yet
from-scratch (+8.21/71%), ladder (rung-1 degrades −0.71), and champion-anchored
(+13.54 < +15.17) all fail to beat the readout-only family — only the readout
family's basin supports the restart-refinement ladder that produced the
champion).

## Failure-mode analysis (final)

Misses (13%) are **diffuse**: uniform across decision-step quartiles (86–88%
catch), wall-bounce counts (86%/86%/96% for 0/1/2 bounces), and ball speeds.
No regime to target. Exact landing reconstruction shows **52% of misses are
marginal** (0.10–0.15 from the paddle vs the 0.10 catch window) and the policy
reaches within 0.05 of the true landing spot for 91% of balls — a *precision*
ceiling, not a directional error. Final adjudication (osc-vs-bias probe): on
missed balls the paddle's 10-step median position is **0.156** from the true
landing — it **settles at a biased target**, ruling out oscillation-phase
mechanisms (action feedback, hysteresis). The policy's signature behaviour is
**functional bang-bang control** — it holds still on only 2% of steps,
switching left↔right every ~2.8 steps; suppressing the dither via margin
decoding degrades performance monotonically. Fixing the precision via a wider
retina fails at training time: 10 fresh draws across retina-24/32 (incl. the
no-bottleneck 291≤300 control and density-preserving projections) all cap at
~50% catch from scratch. The residual gap is the frozen linear readout's
~0.15 landing-estimate error on ~16% of balls.

## The closure matrix (final)

| family | from-scratch | ladder rung-1 | near-optimum restart |
|---|---|---|---|
| readout-only (variance feats) | **+8.42** | **+2.6 ✓ (→+15.17 over 4 rungs)** | below init |
| MI-features | +8.25 | +0.75 (below alive gate) | below init (augment) |
| input-gains | +8.21 | −0.71 ✗ | +13.54 < init |

Only the readout family's basin supports the ladder+averaging pipeline that
produced the champion — a finding about *why* that lineage worked, not just
that it did. Basin climb-rate is family-specific and not predictable from
from-scratch equality.

## Reproduce

```bash
./.auto/measure.sh          # protocol dispatcher (config.json "protocol")
.venv/bin/python flybrain/eval.py data/runs/champ_lin   # champion on frozen eval seeds
```

Artifacts: `data/runs/champ_lin/` (mu, feat_idx, config). Full experiment log:
`.auto/log.jsonl`. Methods & constraints: `.auto/prompt.md`.
