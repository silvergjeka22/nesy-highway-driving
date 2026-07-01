"""highway-env construction + scene adapter + reward wrappers (Parts 1-2).

Public functions:
  * ``create_environment(cfg, render, seed, logic_reward)`` — build the wrapped env.
  * ``read_scene(env)`` — SI-unit scene dict used by the NeSy predicates (Part 2).

``make_env`` is kept as an alias of ``create_environment`` for Parts 2-4.

Wrappers (in order):
  * ``OvertakeCounter``      — instrumentation: info['overtakes'], info['is_offroad'].
  * ``RewardShapingWrapper`` — Part-1 light shaping (overtake bonus, off-road pen).
  * ``LogicRewardWrapper``   — Part-2 Step C: subtract ``logic_penalty(preds)``.

No top-level execution — the notebooks call the functions.
"""

import os

import numpy as np
import gymnasium as gym

# highway_env must be imported so its envs register with gymnasium.
import highway_env  # noqa: F401

def _ensure_render_backend():
    """Set up a render backend for rgb_array frames — used inside the video subprocess.

    Videos are rendered in a subprocess (never in the notebook kernel), so this only
    configures that child process:

      * Always disable audio (``SDL_AUDIODRIVER=dummy``): highway-env calls
        ``pygame.init()``, which opens the audio device, and on a headless machine
        (no sound card) that segfaults the process with ALSA errors. Rendering never
        needs audio.
      * Linux (Colab/Kaggle) has no screen — start a headless virtual display (xvfb
        via pyvirtualdisplay) and keep it alive. We must NOT use ``SDL_VIDEODRIVER=
        dummy`` for video: highway-env disables all drawing when it sees the dummy
        video driver, which would produce blank frames.
      * macOS/Windows — use the native backend, which yields real frames. That is
        safe here because the crash it can cause only happens in the Jupyter kernel.

    An already-configured display or driver is respected.
    """
    import sys
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    if os.environ.get("DISPLAY") or os.environ.get("SDL_VIDEODRIVER"):
        return
    if sys.platform == "linux":
        try:
            from pyvirtualdisplay import Display

            display = Display(visible=0, size=(1400, 900))
            display.start()
            _ensure_render_backend._display = display   # keep the xvfb process alive
        except Exception:
            os.environ["SDL_VIDEODRIVER"] = "dummy"


def create_environment(cfg, render=False, seed=None, logic_reward=False):
    """Build a configured, wrapped highway-env instance.

    Args:
        cfg: full project config dict.
        render: if True, create with ``render_mode='rgb_array'`` (for video).
        seed: optional seed applied on the first reset.
        logic_reward: if True, add the Part-2 ``LogicRewardWrapper``. Off for Part 1.

    Returns:
        A Gymnasium env (constant-shape Kinematics obs, DiscreteMetaAction).
    """
    env_cfg = cfg["env"]

    if render:
        _ensure_render_backend()

    env = gym.make(
        env_cfg["id"],
        render_mode="rgb_array" if render else None,
        config=env_cfg["config"],
    )

    env = OvertakeCounter(env)

    shaping = cfg.get("shaping", {})
    if shaping.get("enabled", False):
        env = RewardShapingWrapper(
            env,
            overtake_bonus=shaping.get("overtake_bonus", 0.0),
            offroad_penalty=shaping.get("offroad_penalty", 0.0),
        )

    if logic_reward:
        env = LogicRewardWrapper(env, cfg)

    if seed is not None:
        env.reset(seed=int(seed))
    return env


# Back-compat alias for Parts 2-4 (which import ``make_env``).
make_env = create_environment


def read_scene(env):
    """Extract an SI-unit scene dict from a (wrapped) highway-env.

    Schema matches ``nesy.roadmap`` (ego + others in metres / m/s, lane as int).
    Reaches through wrappers via ``env.unwrapped``.
    """
    u = env.unwrapped
    ego = u.vehicle
    lanes = u.config.get("lanes_count", 4)

    def vinfo(v):
        vx, vy = float(v.velocity[0]), float(v.velocity[1])
        lane = v.lane_index[2] if getattr(v, "lane_index", None) else 0
        return {
            "x": float(v.position[0]),
            "y": float(v.position[1]),
            "vx": vx,
            "vy": vy,
            "v": float(getattr(v, "speed", (vx ** 2 + vy ** 2) ** 0.5)),
            "lane": int(lane),
        }

    ego_d = vinfo(ego)
    ego_d["on_road"] = bool(getattr(ego, "on_road", True))
    ego_d["heading"] = float(getattr(ego, "heading", 0.0))
    action = getattr(ego, "action", None)
    if isinstance(action, dict) and "acceleration" in action:
        ego_d["a"] = float(action["acceleration"])

    others = [vinfo(v) for v in u.road.vehicles if v is not ego]
    return {"ego": ego_d, "others": others, "lanes_count": int(lanes)}


class OvertakeCounter(gym.Wrapper):
    """Count distinct cars the ego passes, and flag off-road, via ``info``.

    Pure instrumentation — never alters reward or actions. A neighbour that was
    ahead of the ego and is now behind it counts once (tracked by identity).
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
                if vx < ex:
                    self._overtakes += 1
                else:
                    still_ahead.add(vid)
            elif vx > ex:
                still_ahead.add(vid)
        self._ahead_ids = still_ahead

    def _is_offroad(self):
        ego = self._ego()
        if ego is None:
            return False
        return not bool(getattr(ego, "on_road", True))


class RewardShapingWrapper(gym.Wrapper):
    """Add a small, config-driven shaping term to the native reward (Part 1)."""

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


class LogicRewardWrapper(gym.Wrapper):
    """Part-2 Step C: subtract the logic penalty ``Σ λ_i · violation_i``.

    Computes the NeSy predicates from the SI scene each step and subtracts the
    weighted soft-rule penalty (``nesy.roadmap.logic_penalty``). This is the
    single reward-augmentation surface; the hard rules stay in the shield.
    """

    def __init__(self, env, cfg):
        super().__init__(env)
        self.cfg = cfg

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        from nesy.roadmap import predicates, logic_penalty

        scene = read_scene(self.env)
        preds = predicates(scene, self.cfg)
        pen = logic_penalty(preds, self.cfg)

        info = dict(info)
        info["logic_penalty"] = float(pen)
        return obs, float(reward - pen), terminated, truncated, info
