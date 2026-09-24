"""Champion failure-mode diagnosis (train-stream seeds only, never eval seeds).

Per-ball telemetry across 48 fresh episodes: was it caught, ball speed multiplier,
|vx|, wall-bounce count, landing x, paddle miss distance, decisions elapsed,
action switches, paddle travel. Aggregates point at the mechanism to target.
"""
import sys

sys.path.insert(0, ".")

import json

import numpy as np
from flybrain.brain import BrainModel
from flybrain.eval_util import eval_run
from flybrain.game import OBS_DIM, CatchEnv, observation
from flybrain.readout import make_sensory_projection
from flybrain.train import N_ACTIONS, batch_actions

CHAMP = "data/runs/champ_lin"
SEEDS = list(range(7000, 7048))          # fresh train-stream range, never used

cfg = json.load(open(f"{CHAMP}/config.json"))
cfg["feat_idx_path"] = f"{CHAMP}/feat_idx.npy"
graph = dict(np.load(cfg["graph"], allow_pickle=True))
brain = BrainModel(graph, dt=0.005, gain=cfg["gain"])
mu = np.load(f"{CHAMP}/mu.npy")
feat_idx = np.load(f"{CHAMP}/feat_idx.npy")
W_s = make_sensory_projection(len(brain.sensory_idx), OBS_DIM)
P = mu.reshape(1, N_ACTIONS, len(feat_idx))

rows = []
for s in SEEDS:
    brain.init_batch(1)
    brain.reset_batch()
    env = CatchEnv(s)
    obs = observation(env)
    prev_act, switches, travel = 1, 0, 0.0
    ball = {"speed": None, "vx0": abs(env.ball_vx), "bounces": 0, "steps": 0,
            "land": None, "miss_d": None, "caught": None, "sw": 0, "trav": 0.0}
    prev_bx, prev_vx = env.ball_x, env.ball_vx
    while True:
        S = np.maximum(W_s @ obs[None].T, 0.0)
        brain.clamp_sensory_batch(S)
        brain.step_batch(cfg.get("n_substeps", 4))
        feats = brain.r[feat_idx].T
        act = int(batch_actions(P, feats)[0])
        if act != prev_act:
            switches += 1
            ball["sw"] += 1
        travel += abs(act - 1) * 0.0425
        ball["trav"] += abs(act - 1) * 0.0425
        prev_act = act
        _, done = env.step(act)
        ball["steps"] += 1
        if env.ball_x < prev_bx - 1e-9 and prev_vx > 0 or env.ball_x > prev_bx + 1e-9 and prev_vx < 0:
            pass
        if np.sign(env.ball_vx) != np.sign(prev_vx) and abs(prev_vx) > 1e-9:
            ball["bounces"] += 1
        prev_bx, prev_vx = env.ball_x, env.ball_vx
        # ball resolved? catches count increments exactly at resolution
        if env.catches + (env.balls - (1 if not env.done and env.ball_y < 1.0 else 0)) > len([r for r in rows if r["seed"] == s]):
            pass
        if env.ball_y >= 0.999 and ball["land"] is None and ball["steps"] > 1:
            # a new ball just spawned -> previous ball resolved
            caught = env.score > ball.get("prev_score", -1)
            ball["caught"] = None  # filled below
        if done:
            break
        obs = observation(env)
    rows.append({"seed": s, "total": env.score})

# simpler: instrument via env internals in a second pass is messy; use aggregate stats
r = eval_run(brain, mu, cfg, seeds=SEEDS, batch=48)
print(f"aggregate on fresh seeds: reward {r['eval_reward']:+.2f} catch {r['catch_rate']*100:.1f}% "
      f"act l/s/r {r['act_l']:.2f}/{r['act_s']:.2f}/{r['act_r']:.2f}")

# per-ball telemetry with a clean loop (re-run with explicit ball bookkeeping)
rows = []
for s in SEEDS:
    brain.init_batch(1)
    brain.reset_batch()
    env = CatchEnv(s)
    obs = observation(env)
    prev_act = 1
    cur = {"speed": None, "avx": abs(env.ball_vx), "bounces": 0, "steps": 0,
           "sw": 0, "trav": 0.0, "caught": None, "land": None, "d": None}
    resolved = 0
    catches_before = 0
    while True:
        S = np.maximum(W_s @ obs[None].T, 0.0)
        brain.clamp_sensory_batch(S)
        brain.step_batch(cfg.get("n_substeps", 4))
        feats = brain.r[feat_idx].T
        act = int(batch_actions(P, feats)[0])
        if act != prev_act:
            cur["sw"] += 1
        cur["trav"] += abs(act - 1) * 0.0425
        prev_act = act
        vx_before = env.ball_vx
        _, done = env.step(act)
        cur["steps"] += 1
        if np.sign(env.ball_vx) != np.sign(vx_before):
            cur["bounces"] += 1
        if env.balls > resolved:                    # a ball just resolved
            cur["caught"] = env.catches > catches_before
            catches_before = env.catches
            cur["land"] = env.ball_x if env.ball_y <= 0.0 else None
            cur["d"] = abs(env.ball_x - env.paddle_x) if env.ball_y <= 0.0 else None
            rows.append(cur)
            resolved = env.balls
            cur = {"speed": None, "avx": abs(env.ball_vx), "bounces": 0, "steps": 0,
                   "sw": 0, "trav": 0.0, "caught": None, "land": None, "d": None}
        if done:
            break
        obs = observation(env)

import statistics as st
ca = [r for r in rows if r["caught"]]
mi = [r for r in rows if not r["caught"]]
print(f"balls: {len(rows)}  caught {len(ca)} ({len(ca)/len(rows)*100:.1f}%)  missed {len(mi)}")
for name, grp in (("caught", ca), ("missed", mi)):
    if not grp:
        continue
    print(f"{name}: bounces {st.mean(g['bounces'] for g in grp):.2f}  steps {st.mean(g['steps'] for g in grp):.1f}  "
          f"|vx| {st.mean(g['avx'] for g in grp):.3f}  switches {st.mean(g['sw'] for g in grp):.1f}  "
          f"travel {st.mean(g['trav'] for g in grp):.2f}")
bn = {}
for r in rows:
    bn[r["bounces"]] = bn.get(r["bounces"], [0, 0])
    bn[r["bounces"]][0] += 1
    bn[r["bounces"]][1] += int(r["caught"])
print("catch rate by bounce count:", {k: f"{v[1]}/{v[0]}" for k, v in sorted(bn.items())})
swt = [r["sw"] / max(r["steps"], 1) for r in rows]
print(f"switch rate: mean {st.mean(swt):.3f}  p90 {sorted(swt)[int(len(swt)*0.9)]:.3f}")
d = [r["d"] for r in mi if r["d"] is not None]
if d:
    print(f"miss distances: mean {st.mean(d):.3f} median {st.median(d):.3f}  "
          f"within 0.10: {sum(x <= 0.10 for x in d)}/{len(d)}  within 0.17: {sum(x <= 0.17 for x in d)}/{len(d)}")
land_c = [abs(r["land"] - 0.5) if r["land"] is not None else None for r in mi]
land_c = [x for x in land_c if x is not None]
if land_c:
    print(f"missed-landing |x-0.5|: mean {st.mean(land_c):.3f}  >0.35 (near wall): {sum(x > 0.35 for x in land_c)}/{len(land_c)}")
stp = sorted(rows, key=lambda r: r["steps"])
q = len(stp) // 4
print("catch by step-count quartile:", [f"{sum(r['caught'] for r in stp[i*q:(i+1)*q])}/{q}" for i in range(4)])
