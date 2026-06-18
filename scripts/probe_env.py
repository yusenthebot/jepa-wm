"""Introspect the real PointMaze API before committing the env wrapper.

Confirms: offscreen rgb rendering works, info/obs structure, success signal,
and how to place the ball at an arbitrary xy (needed for a goal image)."""
from __future__ import annotations

import sys

import numpy as np
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

ENV_ID = sys.argv[1] if len(sys.argv) > 1 else "PointMaze_Open-v3"

env = gym.make(ENV_ID, render_mode="rgb_array", continuing_task=False,
               max_episode_steps=200)
print("env:", ENV_ID)
print("obs space:", env.observation_space)
print("act space:", env.action_space)

obs, info = env.reset(seed=0)
print("obs keys:", list(obs.keys()))
print("observation:", np.asarray(obs["observation"]))
print("achieved_goal:", np.asarray(obs["achieved_goal"]))
print("desired_goal:", np.asarray(obs["desired_goal"]))
print("reset info keys:", list(info.keys()))

img = env.render()
print("render shape/dtype:", None if img is None else (img.shape, img.dtype))
try:
    import imageio.v2 as imageio
    imageio.imwrite("runs/probe_reset.png", img)
    print("wrote runs/probe_reset.png")
except Exception as e:  # noqa: BLE001
    print("imwrite failed:", e)

a = env.action_space.sample()
obs2, rew, term, trunc, info2 = env.step(a)
print("step reward:", rew, "term:", term, "trunc:", trunc)
print("step info:", {k: info2[k] for k in info2})

# --- introspect how to place the ball (for a goal image) ---
u = env.unwrapped
print("unwrapped type:", type(u).__name__)
attrs = [a for a in dir(u) if not a.startswith("__")]
interesting = [a for a in attrs if any(k in a.lower() for k in
              ("point", "maze", "set_state", "data", "model", "goal", "reset", "ball", "agent"))]
print("interesting unwrapped attrs:", interesting)

for cand in ("point_env", "ball_env", "maze"):
    if hasattr(u, cand):
        sub = getattr(u, cand)
        print(f"  {cand} -> {type(sub).__name__}; has set_state={hasattr(sub, 'set_state')} "
              f"has data={hasattr(sub, 'data')} has model={hasattr(sub, 'model')}")

# Try to render the ball at the goal: set qpos to desired_goal, render.
goal = np.asarray(obs["desired_goal"], dtype=np.float64)
placed = False
try:
    pe = getattr(u, "point_env", None)
    if pe is not None and hasattr(pe, "set_state"):
        qpos = pe.data.qpos.copy()
        qvel = pe.data.qvel.copy()
        print("point_env qpos shape:", qpos.shape, "qvel shape:", qvel.shape)
        qpos[:2] = goal
        pe.set_state(qpos, np.zeros_like(qvel))
        gimg = env.render()
        import imageio.v2 as imageio
        imageio.imwrite("runs/probe_goal.png", gimg)
        print("wrote runs/probe_goal.png (ball placed at goal)")
        placed = True
except Exception as e:  # noqa: BLE001
    print("place-ball-at-goal failed:", repr(e))
print("PLACED_BALL_AT_GOAL:", placed)
env.close()
print("OK")
