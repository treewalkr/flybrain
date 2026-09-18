# flybrain

A real fruit-fly brain — wired from the **MaleCNS v1.0 connectome**
([male-cns.janelia.org](https://male-cns.janelia.org/download/), Janelia FlyEM, CC-BY) —
plays **pong-catch**. A ball drops; the fly's brain, simulated with its true synaptic
wiring, moves the paddle. Only a linear readout over descending-neuron activity is
trained (cross-entropy method); **every synapse in the brain is frozen real anatomy**.

## What runs where

```
.venv/bin/python -m flybrain.fetch_data        # download MaleCNS (1.1 GB, sha-pinned)
.venv/bin/python -m flybrain.graph --circuit   # signed sparse graph + sensorimotor circuit
.venv/bin/python -m flybrain.skeletons         # SWC skeletons for rendering
.venv/bin/python -m flybrain.train ...         # CEM training of the DN readout
.venv/bin/python -m flybrain.eval data/runs/best
.venv/bin/python -m flybrain.plot data/runs/best
.venv/bin/python -m flybrain.play data/runs/best   # gameplay + 3D brain activity video
```

## The brain

- **Circuit**: every neuron on a sensory→(1 hop)→DN motor path in the central brain —
  4,706 neurons (300 sensory sample, 3,092 intermediates, all 1,314 descending
  neurons), 306k edges, sign from consensus neurotransmitter (GABA/glutamate/histamine
  inhibitory), strength log1p(synapse count), row-normalised.
- **Dynamics**: leaky rates, `tau dv/dt = -v + g*W*relu(v)`, tau 10 ms (optic) /
  20 ms (central, descending), Euler dt = 5 ms, 4 substeps per 50 ms decision.
- **Senses**: the game observation (16x12 retina + paddle/velocity) drives a frozen
  random sparse projection onto 300 real sensory neurons (rate clamp).
- **Action**: linear readout over the top-256 variance-ranked DN rates + sensory
  rates; argmax of 3 actions (left/stay/right).

## Learning

CEM (population 64, elites 10, 48 generations, ~4,600 training episodes) on the
readout only. The connectome is never modified — it is a fixed, biologically real
nonlinear feature map from senses to descending neurons.

| policy | eval reward (frozen seeds) | catch rate |
|---|---:|---:|
| random | ≈ −6.7 | 33% |
| CEM readout (this repo's recipe) | **+2.3** | **56%** |

Failure modes that did **not** work (documented in `.auto/prompt.md`): dopamine/RPE
plasticity (also the user's earlier flappy-bird attempt and FlyPong's published
negative result), elitist CEM, cross-seed weight averaging, subsampling the circuit
below ~3k neurons, coarser integration.

## Videos

`flybrain.play` renders each decision: the game on the left, and on the right the
same 4,706 neurons as skeletons inside the JRCFIB2022M brain shell, coloured by
live activity — you can watch the sensory wave propagate to descending neurons as
the paddle tracks the ball.

## Data & credit

- MaleCNS v1.0 flat connectome, neurotransmitters, annotations and skeletons:
  Janelia Research Campus, FlyEM project (CC-BY).
- Brain shell mesh: JRCFIB2022M point-cloud shells from the MaleCNS ROI set.
- Renderer adapted from the TMNF-C project's `brain_render.py`.
