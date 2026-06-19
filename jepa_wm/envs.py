"""PointMaze wrapped for pixel-JEPA: render -> small RGB, plus a goal image
(ball placed at the goal) so we can encode a goal latent for planning.

True xy is exposed ONLY for evaluation/diagnostics (success check, linear probe).
The world model never receives it.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
import gymnasium_robotics
import mujoco
from PIL import Image

gym.register_envs(gymnasium_robotics)


def _resize_chw(img: np.ndarray, size: int) -> np.ndarray:
    """RGB uint8 HxWx3 -> float32 CHW in [0,1], resized to size x size."""
    pil = Image.fromarray(img).resize((size, size), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


class PointMazePixels:
    def __init__(self, env_id: str, img_size: int = 64, max_episode_steps: int = 200,
                 seed: int = 0, cam_distance: float = 6.5, enlarge_agent: bool = True,
                 agent_size: float = 0.5):
        self.env = gym.make(env_id, render_mode="rgb_array",
                            continuing_task=False, max_episode_steps=max_episode_steps)
        self.img_size = img_size
        self.action_dim = int(self.env.action_space.shape[0])
        self._seed = seed
        self._last_obs = None
        if enlarge_agent:
            self._enlarge_agent(size=agent_size)
        self._set_topdown_camera(cam_distance)

    def _enlarge_agent(self, size: float = 0.5, rgba=(1.0, 0.9, 0.1, 1.0)):
        """Make the agent a large, bright ball. The default agent is a ~0.1-radius
        site in a ~5-wide arena (~2% of pixels) -> no encoder can isolate it from the
        static background. A prominent ball makes its position the dominant signal so
        ordinary objectives (VICReg / reconstruction) recover it. Dynamics unchanged."""
        self.env.reset(seed=self._seed)
        m = getattr(self.env.unwrapped, "point_env", None)
        m = getattr(m, "model", None)
        if m is None:
            return
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "particle_site")
        if sid >= 0:
            m.site_size[sid] = [size, size, size]
            m.site_rgba[sid] = list(rgba)

    def _set_topdown_camera(self, distance: float):
        """Top-down camera: the ball becomes a large, prominent blob and position
        maps ~linearly to image coordinates. Without this the ball is ~3 px in a
        sea of static background (cross-frame pixel std ~1.8/255) and no encoder
        can localize it."""
        self.env.reset(seed=self._seed)
        pe = getattr(self.env.unwrapped, "point_env", None)
        mr = getattr(pe, "mujoco_renderer", None)
        if mr is not None:
            mr.default_cam_config = {
                "distance": distance, "elevation": -90.0, "azimuth": 90.0,
                "lookat": np.array([0.0, 0.0, 0.0]),
            }
            mr.close()  # drop cached viewer so the new camera config takes effect

    # --- low-level ---
    def _render_small(self) -> np.ndarray:
        return _resize_chw(self.env.render(), self.img_size)

    def _xy(self, obs) -> np.ndarray:
        return np.asarray(obs["achieved_goal"], dtype=np.float32)

    def _goal_xy(self, obs) -> np.ndarray:
        return np.asarray(obs["desired_goal"], dtype=np.float32)

    # --- episode API ---
    def reset(self, seed: int | None = None):
        obs, info = self.env.reset(seed=seed)
        self._last_obs = obs
        img = self._render_small()
        return img, self._xy(obs), self._goal_xy(obs), info

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        obs, rew, term, trunc, info = self.env.step(action)
        self._last_obs = obs
        img = self._render_small()
        xy = self._xy(obs)
        gxy = self._goal_xy(obs)
        success = bool(info.get("success", np.linalg.norm(xy - gxy) < 0.45))
        return img, xy, success, term, trunc, info

    def render_goal_image(self, goal_xy: np.ndarray) -> np.ndarray:
        """Place the ball at goal_xy, render, restore. Returns CHW float image.

        Within one episode the goal marker is fixed, so a current frame and this
        goal frame differ (in latent) mainly by ball position — exactly the
        signal CEM should minimize."""
        u = self.env.unwrapped
        pe = getattr(u, "point_env", None)
        if pe is None or not hasattr(pe, "set_state"):
            # Fallback: just render current frame (degrades planning, logged upstream).
            return self._render_small()
        qpos = pe.data.qpos.copy()
        qvel = pe.data.qvel.copy()
        saved_qpos, saved_qvel = qpos.copy(), qvel.copy()
        qpos[:2] = np.asarray(goal_xy, dtype=qpos.dtype)
        pe.set_state(qpos, np.zeros_like(qvel))
        img = self._render_small()
        pe.set_state(saved_qpos, saved_qvel)  # restore so the episode is untouched
        return img

    def close(self):
        self.env.close()
