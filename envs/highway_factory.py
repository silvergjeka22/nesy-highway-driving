"""highway-env construction for Part 1 (and the reward-wrapper hook for Part 2).

Public entry point: ``make_env(cfg, render, seed, fast)``. Everything the agent
sees (observation, discrete meta-actions, native reward) is configured from the
YAML; the only Part-1 additions are two thin wrappers:

  * ``OvertakeCounter`` — instrumentation only; writes ``info['overtakes']`` and
    ``info['is_offroad']`` so the evaluation harness can report them. Does NOT
    change the reward or the action.
  * ``RewardShapingWrapper`` — the deliberately *light* shaping the plan calls
    for (small bonus per car passed, small off-road penalty). In Part 2 this is
    where the logic-shaped reward penalties get added, so the hook lives here.

No top-level execution — the notebook calls ``make_env``.
"""

import gymnasium as gym
import numpy as np

# highway_env must be imported so its envs register with gymnasium.
import highway_env  # noqa: F401


def make_env(cfg, render=False, seed=None, fast=False):
    """Build a configured, wrapped highway-env instance.

    Args:
        cfg: full project config dict (see ``configs/highway.yaml``).
        render: if True, create the env with ``render_mode='rgb_array'`` (for
            ``record_video``); otherwise no rendering (faster training/eval).
        seed: optional seed applied via ``reset(seed=...)`` on first reset.
        fast: if True, use ``env.fast_id`` (``highway-fast-v0``) for quick smoke
            tests. Algorithms/observation are identical.

    Returns:
        A Gymnasium env exposing a constant-shape Kinematics observation and a
        ``DiscreteMetaAction`` space.
    """
    env_cfg = cfg["env"]
    env_id = env_cfg["fast_id"] if fast else env_cfg["id"]

    env = gym.make(
        env_id,
        render_mode="rgb_array" if render else None,
        config=env_cfg["config"],
    )

    # Instrumentation first (reads the raw road state), then optional shaping.
    env = OvertakeCounter(env)

    shaping = cfg.get("shaping", {})
    if shaping.get("enabled", False):
        env = RewardShapingWrapper(
            env,
            overtake_bonus=shaping.get("overtake_bonus", 0.0),
            offroad_penalty=shaping.get("offroad_penalty", 0.0),
        )

    if seed is not None:
        env.reset(seed=int(seed))

    return env


class OvertakeCounter(gym.Wrapper):
    """Count distinct cars the ego passes, and flag off-road, via ``info``.

    Overtake heuristic: a neighbour that was *ahead* of the ego (greater
    longitudinal position, same direction) and is now *behind* it counts once.
    Vehicles are tracked by identity so none is double-counted; vehicles that
    appear ahead later are added to the watch set. This is pure instrumentation
    — it never alters reward or actions.
    """

    def __init__(self, env):
        super().__init__(env)
        self._ahead_ids = set()
        self._overtakes = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._overtakes = 0
        self._ahead_ids = self._currently_ahead()
        info = dict(info)
        info["overtakes"] = 0
        info["is_offroad"] = self._is_offroad()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_overtakes()
        info = dict(info)
        info["overtakes"] = self._overtakes
        info["is_offroad"] = self._is_offroad()
        return obs, reward, terminated, truncated, info

    # --- helpers reaching into the unwrapped highway-env state ---------------
    def _ego(self):
        return self.env.unwrapped.vehicle

    def _others(self):
        ego = self._ego()
        return [v for v in self.env.unwrapped.road.vehicles if v is not ego]

    def _currently_ahead(self):
        ego = self._ego()
        if ego is None:
            return set()
        return {id(v) for v in self._others() if v.position[0] > ego.position[0]}

    def _update_overtakes(self):
        ego = self._ego()
        if ego is None:
            return
        ex = ego.position[0]
        still_ahead = set()
        for v in self._others():
            vx = v.position[0]
            vid = id(v)
            if vid in self._ahead_ids:
                if vx < ex:                 # was ahead, now behind -> overtaken
                    self._overtakes += 1
                else:
                    still_ahead.add(vid)
            elif vx > ex:                   # newly ahead -> start tracking
                still_ahead.add(vid)
        self._ahead_ids = still_ahead

    def _is_offroad(self):
        ego = self._ego()
        if ego is None:
            return False
        on_road = getattr(ego, "on_road", True)
        return not bool(on_road)


class RewardShapingWrapper(gym.Wrapper):
    """Add a small, config-driven shaping term to the native reward.

    Part 1 keeps this minimal on purpose. The same wrapper is where Part 2's
    logic-shaped reward penalties (``- Σ λ_i · violation_i``) will be added, so
    the reward-augmentation surface lives in exactly one place.
    """

    def __init__(self, env, overtake_bonus=0.0, offroad_penalty=0.0):
        super().__init__(env)
        self.overtake_bonus = float(overtake_bonus)
        self.offroad_penalty = float(offroad_penalty)
        self._prev_overtakes = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_overtakes = int(info.get("overtakes", 0))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        overtakes = int(info.get("overtakes", self._prev_overtakes))
        passed = max(0, overtakes - self._prev_overtakes)
        self._prev_overtakes = overtakes

        shaped = reward + self.overtake_bonus * passed
        if info.get("is_offroad", False):
            shaped -= self.offroad_penalty

        info = dict(info)
        info["native_reward"] = float(reward)
        info["shaped_reward"] = float(shaped)
        return obs, float(shaped), terminated, truncated, info
