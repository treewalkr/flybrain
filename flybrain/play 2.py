"""Run a trained fly brain playing pong-catch; record game + 3D brain activity video.

    .venv/bin/python -m flybrain.play data/runs/best --out data/out/play.mp4

Left panel: the game. Right panel: the MaleCNS circuit's skeletons inside the brain
shell, coloured by per-decision activity (same brain state that produces the actions).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from flybrain.brain import BrainModel
from flybrain.game import CatchEnv, observation
from flybrain.readout import make_sensory_projection
from flybrain.render import BRAIN_CENTER, BrainRenderer, Camera
from flybrain.skeletons import load as load_skeletons
from flybrain.train import N_ACTIONS, batch_actions

GAME_W, GAME_H = 320, 320
GAP = 8


def draw_game(env: CatchEnv) -> np.ndarray:
    """Simple solid-colour game frame (H, W, 3) uint8."""
    img = np.full((GAME_H, GAME_W, 3), (16, 18, 28), np.uint8)
    # arena border
    img[0, :, :] = (60, 70, 100); img[-1, :, :] = (60, 70, 100)
    img[:, 0, :] = (60, 70, 100); img[:, -1, :] = (60, 70, 100)
    # ball
    bx = int(env.ball_x * (GAME_W - 1))
    by = int((1.0 - env.ball_y) * (GAME_H - 1))
    r = 6
    img[max(by - r, 0):by + r, max(bx - r, 0):bx + r] = (255, 190, 60)
    # paddle
    px = env.paddle_x
    w = env.PADDLE_W if hasattr(env, "PADDLE_W") else 0.14
    x0 = int((px - w / 2) * (GAME_W - 1)); x1 = int((px + w / 2) * (GAME_W - 1))
    img[GAME_H - 18:GAME_H - 6, max(x0, 0):max(x1, 1)] = (90, 220, 120)
    # score line
    for i in range(min(env.catches, 40)):
        img[4:10, 6 + i * 7:10 + i * 7] = (90, 220, 120)
    return img


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path, help="training run dir with mu.npy + config.json")
    p.add_argument("--out", type=Path, default=Path("data/out/play.mp4"))
    p.add_argument("--seed", type=int, default=9000, help="episode seed (an eval seed by default)")
    p.add_argument("--every", type=int, default=4, help="render brain every N decisions")
    p.add_argument("--fps", type=int, default=12)
    a = p.parse_args()

    cfg = json.loads((a.run / "config.json").read_text())
    graph = dict(np.load(cfg["graph"], allow_pickle=True))
    brain = BrainModel(graph, dt=0.005, gain=cfg.get("gain", 1.0))
    from flybrain.game import OBS_DIM
    W_s = make_sensory_projection(len(brain.sensory_idx), OBS_DIM)
    from flybrain.train import select_features
    feat_idx = select_features(brain, W_s, cfg.get("use_sensory", True),
                               cfg.get("sensory_gain", 1.0), cfg.get("n_substeps", 4),
                               cfg.get("dn_topk"))
    params = np.load(a.run / "mu.npy").reshape(1, N_ACTIONS, len(feat_idx))
    brain.init_batch(1)

    skel = load_skeletons()
    assert np.array_equal(skel["body_ids"], graph["body_id"]), "skeleton cache must match the circuit"
    renderer = BrainRenderer(skel, Camera.frontal(orbit_deg_per_s=6.0))

    env = CatchEnv(a.seed)
    frames = []
    t = 0.0
    while not env.done:
        obs = observation(env)[None]
        S = np.maximum(W_s @ obs.T, 0.0) * cfg.get("sensory_gain", 1.0)
        brain.clamp_sensory_batch(S)
        brain.step_batch(cfg.get("n_substeps", 4))
        feats = brain.r[feat_idx].T
        act = int(batch_actions(np.repeat(params, 1, axis=0), feats)[0])
        env.step(act)
        t += 1
        if t % a.every == 0 or env.done:
            bimg = renderer.frame(brain.activity(), t / a.fps)
            gimg = draw_game(env)
            h = min(bimg.shape[0], GAME_H)
            combo = np.full((max(bimg.shape[0], GAME_H), bimg.shape[1] + GAME_W + GAP, 3), 12, np.uint8)
            combo[:bimg.shape[0], :bimg.shape[1]] = bimg
            combo[:GAME_H, bimg.shape[1] + GAP:] = gimg
            frames.append(combo)
    renderer.close()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(a.out, frames, fps=a.fps, quality=8)
    print(f"{a.out}: {len(frames)} frames, catches {env.catches}/{env.balls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
