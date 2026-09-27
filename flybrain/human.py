"""Play pong-catch yourself, or race the trained fly brain on identical balls.

  .venv/bin/python -m flybrain.human                # you alone
  .venv/bin/python -m flybrain.human --brain        # you (left) vs the brain (right)
  .venv/bin/python -m flybrain.human --seed 42      # replay a specific episode

Controls: <- / -> move, Esc quit. 20 balls per episode, same physics as training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pygame

from flybrain.game import (BALLS_PER_EPISODE, DT, FIELD_H, FIELD_W, PADDLE_W,
                           CatchEnv, observation)

CELL = 560, 620
BG, FG, BALL, PADDLE, YOU, TXT = (12, 12, 16), (200, 200, 210), (255, 180, 60), (90, 220, 140), (90, 160, 255), (230, 230, 235)


def draw(screen: pygame.Surface, font, env: CatchEnv, ox: int, label: str, color, done: bool):
    pygame.draw.rect(screen, (28, 28, 36), (ox, 0, CELL[0], CELL[1]))
    for y in (0.0, 1.0):
        pygame.draw.line(screen, FG, (ox, y * (CELL[1] - 60)), (ox + CELL[0], y * (CELL[1] - 60)), 2)
    bx = ox + env.ball_x / FIELD_W * CELL[0]
    by = (1.0 - env.ball_y / FIELD_H) * (CELL[1] - 60)
    pygame.draw.circle(screen, BALL, (int(bx), int(by)), 12)
    pw = PADDLE_W / FIELD_W * CELL[0]
    px = ox + env.paddle_x / FIELD_W * CELL[0] - pw / 2
    pygame.draw.rect(screen, color, (px, CELL[1] - 76, pw, 14), border_radius=7)
    surf = font.render(f"{label}  {env.catches}/{env.balls_total}  score {env.score:+d}", True, TXT if not done else (120, 255, 120))
    screen.blit(surf, (ox + 14, CELL[1] - 46))


def brain_policy(policy):
    return lambda env: int(policy(observation(env)[None, :])[0])


def load_policy(run: Path):
    """Rebuild the greedy trained policy from a run dir (mirrors eval_util.eval_run)."""
    from flybrain.brain import BrainModel as BatchedBrain
    from flybrain.game import OBS_DIM
    from flybrain.readout import make_sensory_projection
    from flybrain.train import N_ACTIONS, batch_actions
    if not (run / "mu.npy").exists():
        return None
    cfg = __import__("json").load(open(run / "config.json"))
    graph = dict(np.load(cfg.get("graph", "data/fly/brain_circuit.npz"), allow_pickle=True))
    brain = BatchedBrain(graph, dt=0.005, gain=cfg.get("gain", 1.0))
    W_s = make_sensory_projection(len(brain.sensory_idx), OBS_DIM)
    feat_idx = np.load(run / "feat_idx.npy").astype(np.int64)
    P = np.load(run / "mu.npy").reshape(1, N_ACTIONS, len(feat_idx))
    brain.init_batch(1)

    def policy(obs: np.ndarray) -> np.ndarray:
        S = np.maximum(W_s @ obs.T, 0.0) * cfg.get("sensory_gain", 1.0)
        brain.clamp_sensory_batch(S)
        brain.step_batch(cfg.get("n_substeps", 4))
        return batch_actions(P, brain.r[feat_idx].T)

    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--brain", action="store_true", help="race the trained brain on identical balls")
    ap.add_argument("--run", default=None, help="trained run dir for --brain (default: strongest artifact)")
    args = ap.parse_args()
    if args.run is None:
        for cand in ("data/runs/champ_lin", "data/runs/champ_lin2", "data/runs/best"):
            if Path(cand + "/mu.npy").exists():
                args.run = cand
                break

    you = CatchEnv(args.seed)
    fly = CatchEnv(args.seed)
    policy = None
    if args.brain:
        from flybrain.human import load_policy
        policy = load_policy(Path(args.run))
        if policy is None:
            sys.exit(f"no trained run at {args.run} (.venv/bin/python -m flybrain.train first)")
        fly_act = brain_policy(policy)

    pygame.init()
    w = CELL[0] * (2 if args.brain else 1)
    screen = pygame.display.set_mode((w, CELL[1]))
    pygame.display.set_caption("flybrain pong-catch" + (" — you vs the brain" if args.brain else ""))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo,monospace", 22)

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit(); return
        keys = pygame.key.get_pressed()
        a = 0 if keys[pygame.K_LEFT] else 2 if keys[pygame.K_RIGHT] else 1
        screen.fill(BG)
        draw(screen, font, you, 0, "YOU", YOU, you.done)
        you.step(a)
        if args.brain:
            draw(screen, font, fly, CELL[0], "FLY", PADDLE, fly.done)
            fly.step(fly_act(fly))
            if you.done and fly.done:
                res = "YOU WIN" if you.score > fly.score else "BRAIN WINS" if fly.score > you.score else "TIE"
                screen.blit(font.render(f"{res}  ({you.score:+d} vs {fly.score:+d}) — Esc to quit, R to rematch",
                                        True, (255, 220, 90)), (20, 16))
                if keys[pygame.K_r]:
                    you, fly = CatchEnv(args.seed), CatchEnv(args.seed)
        elif you.done:
            screen.blit(font.render(f"final {you.catches}/{BALLS_PER_EPISODE} ({you.score:+d}) — Esc to quit, R to rematch",
                                    True, (255, 220, 90)), (20, 16))
            if keys[pygame.K_r]:
                you = CatchEnv(args.seed)
        pygame.display.flip()
        clock.tick(int(1 / DT))


if __name__ == "__main__":
    main()
