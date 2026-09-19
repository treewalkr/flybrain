"""Pong-catch: the game the fly brain plays.

A ball drops from a random x with a random angle; the paddle (3 actions:
left / stay / right) must be under it when it lands. Episodes are seeded.
The observation is encoded as a sensory code for the brain's sensory neurons:

  - a coarse retina: `retina_w x retina_h` grid, ball cell bright + paddle row
    bright (bottom row, paddle position),
  - paddle x proprioception channel,
  - ball dx/dy velocity channels.

`SensoryCodec` maps the observation vector -> (n_sensory,) rates, deterministic
given the observation. The brain is clamped with these rates each decision.

Seeds protocol (anti-overfit):
  - training samples random seeds from a stream;
  - EVAL_SEEDS below are frozen forever and only ever used by eval.py / measure.sh.
"""
from __future__ import annotations

import numpy as np

RETINA_W, RETINA_H = 16, 12
FIELD_W, FIELD_H = 1.0, 1.0
PADDLE_W = 0.14
BALL_SPEED = 0.55          # field heights per second
DT = 0.05                  # decision interval, seconds
BALLS_PER_EPISODE = 20
ACTIONS = (0, 1, 2)        # left, stay, right
PADDLE_V = 0.85            # field widths per second

# frozen, public: evaluation episodes (disjoint from any training seed range)
EVAL_SEEDS = tuple(range(9000, 9096))

OBS_DIM = RETINA_W * RETINA_H + 1 + 2   # retina + paddle_x + ball vx,vy


def observation(env: "CatchEnv") -> np.ndarray:
    """Flat float32 observation: retina grid + [paddle_x, ball_vx, ball_vy]."""
    obs = np.zeros(OBS_DIM, np.float32)
    retina = obs[: RETINA_W * RETINA_H].reshape(RETINA_H, RETINA_W)
    bx, by = env.ball_x, env.ball_y
    cx = int(np.clip(bx / FIELD_W * (RETINA_W - 1), 0, RETINA_W - 1))
    cy = int(np.clip(by / FIELD_H * (RETINA_H - 1), 0, RETINA_H - 1))
    retina[cy, cx] = 1.0
    px = int(np.clip(env.paddle_x / FIELD_W * (RETINA_W - 1), 0, RETINA_W - 1))
    retina[RETINA_H - 1, :] = 0.0
    retina[RETINA_H - 1, max(px - 1, 0): px + 2] = 0.6
    obs[RETINA_W * RETINA_H] = env.paddle_x / FIELD_W
    obs[-2] = env.ball_vx * 2.0
    obs[-1] = -env.ball_vy * 2.0
    return obs


class CatchEnv:
    """One ball at a time falls; +1 catch, -1 miss; episode = `balls` balls (default 20)."""

    def __init__(self, seed: int, balls: int = BALLS_PER_EPISODE):
        self.balls_total = balls
        self.rng = np.random.default_rng(seed)
        self.score = 0
        self.catches = 0
        self.balls = 0
        self.done = False
        self._new_ball(first=True)

    def _new_ball(self, first: bool = False) -> None:
        self.ball_x = self.rng.uniform(0.1, 0.9)
        self.ball_y = 1.0 if not first else self.rng.uniform(0.35, 0.8)
        ang = self.rng.uniform(-0.5, 0.5)
        speed = BALL_SPEED * self.rng.uniform(0.85, 1.15)
        self.ball_vx = speed * np.sin(ang) / FIELD_H * FIELD_H
        self.ball_vy = -speed
        if first:
            self.paddle_x = self.rng.uniform(0.3, 0.7)
        # ball starts when previous resolved; paddle persists

    def step(self, action: int) -> tuple[float, bool]:
        """Advance one decision interval. Returns (reward, done)."""
        if self.done:
            return 0.0, True
        dv = (action - 1) * PADDLE_V * DT
        self.paddle_x = float(np.clip(self.paddle_x + dv, PADDLE_W / 2, FIELD_W - PADDLE_W / 2))
        # integrate ball; catch/miss when it reaches the bottom
        steps = 1
        reward = 0.0
        for _ in range(steps):
            self.ball_x += self.ball_vx * DT
            self.ball_y += self.ball_vy * DT
            if self.ball_x < 0.02 or self.ball_x > 0.98:
                self.ball_vx = -self.ball_vx
                self.ball_x = float(np.clip(self.ball_x, 0.02, 0.98))
            if self.ball_y <= 0.0:
                caught = abs(self.ball_x - self.paddle_x) <= PADDLE_W / 2 + 0.03
                reward += 1.0 if caught else -1.0
                self.catches += int(caught)
                self.balls += 1
                if self.balls >= self.balls_total:
                    self.done = True
                    self.score = self.catches
                    return reward, True
                self._new_ball()
        return reward, False


def play_episode(policy, seed: int) -> tuple[float, int, int]:
    """policy: callable(obs_vec) -> action. Returns (total_reward, catches, balls)."""
    env = CatchEnv(seed)
    total = 0.0
    while True:
        a = policy(observation(env))
        r, done = env.step(a)
        total += r
        if done:
            return total, env.catches, env.balls
