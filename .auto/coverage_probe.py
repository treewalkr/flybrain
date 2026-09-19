"""Per-position coverage screening over the existing artifact pool.

Hypothesis: aggregate reward hides patchy spatial coverage (hemifield
specialists, hot/cold cells). Selecting by worst-landing-zone catch (maximin)
should find a better-rounded artifact — or prove the champion is already
well-covered.

Protocol (overfit-safe):
  - score every pool artifact on VALIDATION seeds 8500..8595 (never CEM-sampled,
    never eval);
  - per ball, predict landing x in closed form (specular wall bounces);
  - bucket into 6 landing zones; per-zone catch rate per artifact;
  - PRE-REGISTERED rule: eligible = overall val catch >= champion - 0.08;
    winner = argmax min-zone catch among eligible;
  - confirm the winner ONCE on frozen eval seeds.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from flybrain.brain import BrainModel as BatchedBrain          # noqa: E402
from flybrain.eval_util import eval_run                        # noqa: E402
from flybrain.game import CatchEnv, observation                # noqa: E402
from flybrain.human import load_policy                         # noqa: E402

VAL_SEEDS = list(range(8500, 8596))
ZONES = np.linspace(0.0, 1.0, 7)                                # 6 zones

def landing_x(env: CatchEnv) -> float:
    """Closed-form landing x: constant vy, specular bounces at 0.02/0.98."""
    t = env.ball_y / abs(env.ball_vy)
    u = (env.ball_x - 0.02) + env.ball_vx * t
    p = 0.96
    s = np.mod(u, 2 * p)
    return 0.02 + np.where(s < p, s, 2 * p - s)

def coverage(path: Path) -> dict:
    policy = load_policy(path)
    zone = np.zeros((6, 2), np.int64)                           # caught, total
    overall = np.zeros(2, np.int64)
    for s in VAL_SEEDS:
        env = CatchEnv(s)
        while True:
            a = int(policy(observation(env)[None, :])[0])
            lx = float(landing_x(env))
            z = int(np.clip(np.searchsorted(ZONES, lx) - 1, 0, 5))
            r, done = env.step(a)
            if r != 0:                         # ball resolved this step
                zone[z] += (int(r > 0), 1)
                overall += (int(r > 0), 1)
            if done:
                break
    zc = zone[:, 0] / np.maximum(zone[:, 1], 1)
    return {"catch": overall[0] / overall[1], "min_zone": zc.min(),
            "zones": zc.round(2).tolist()}

pool = sorted(Path("data/runs").glob("ms_*"), key=lambda p: int(p.name.split("_")[1]))
rows = []
for p in pool:
    if not (p / "mu.npy").exists():
        continue
    c = coverage(p)
    rows.append((p.name, c))
    print(f"{p.name}: catch {c['catch']*100:4.1f}%  min-zone {c['min_zone']*100:4.1f}%  {c['zones']}", flush=True)

champ = max(rows, key=lambda r: r[1]["catch"])
elig = [r for r in rows if r[1]["catch"] >= champ[1]["catch"] - 0.08]
win = max(elig, key=lambda r: r[1]["min_zone"])
print(f"\nchampion-by-catch: {champ[0]} ({champ[1]['catch']*100:.1f}%)")
print(f"maximin winner: {win[0]} (min-zone {win[1]['min_zone']*100:.1f}%, catch {win[1]['catch']*100:.1f}%)")

if win[0] != champ[0]:
    print(f"confirming {win[0]} once on frozen eval", flush=True)
    graph = dict(np.load("data/fly/brain_circuit.npz", allow_pickle=True))
    cfg = json.load(open(win[0].replace("ms_", "data/runs/ms_") + "/config.json"))
    wp = Path("data/runs") / win[0]
    cfg["feat_idx_path"] = str(wp / "feat_idx.npy")
    brain = BatchedBrain(graph, dt=0.005, gain=cfg["gain"])
    r = eval_run(brain, np.load(wp / "mu.npy"), cfg)
    print(f"METRIC eval_reward={r['eval_reward']:.4f}")
    print(f"METRIC catch_rate={r['catch_rate']:.4f}")
    print(f"METRIC train_reward={win[1]['catch']*20-20*(1-win[1]['catch']):.4f}")
    print(f"METRIC gap={r['eval_reward']-(win[1]['catch']*2-1)*10:.4f}")
    print(f"METRIC hist_last_best=0")
    print(f"METRIC train_seconds=0")
    print(f"METRIC act_l={r['act_l']:.3f} act_s={r['act_s']:.3f} act_r={r['act_r']:.3f}")
else:
    print("RESULT maximin selects the same artifact as aggregate - champion already best-covered; no eval spend")
