import os
import sys

# block pkg_resources before gymnasium imports highway_env -> pygame
sys.modules.setdefault("pkg_resources", None)

import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401


def ensure_render_backend():
    import sys
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    if os.environ.get("DISPLAY") or os.environ.get("SDL_VIDEODRIVER"):
        return
    if sys.platform == "linux":
        try:
            from pyvirtualdisplay import Display
            display = Display(visible=0, size=(1400, 900), color_depth=24)
            display.start()
            ensure_render_backend._display = display
        except Exception:
            os.environ["SDL_VIDEODRIVER"] = "dummy"


def create_environment(cfg, render=False, seed=None, logic_reward=False):
    env_cfg = cfg["env"]
    render_config = dict(env_cfg["config"])

    if render:
        ensure_render_backend()
        render_config["offscreen_rendering"] = True

    env = gym.make(
        env_cfg["id"],
        render_mode="rgb_array" if render else None,
        config=render_config,
    )

    env = OvertakeCounter(env, blocked_gap=cfg["fsm"]["follow_gap"])

    shaping = cfg.get("shaping", {})
    if shaping.get("enabled", False):
        env = RewardShapingWrapper(
            env,
            overtake_bonus=shaping.get("overtake_bonus", 0.0),
            offroad_penalty=shaping.get("offroad_penalty", 0.0),
            collision_penalty=shaping.get("collision_penalty", 0.0),
            lane_change_bonus=shaping.get("lane_change_bonus", 0.0),
            survival_bonus=shaping.get("survival_bonus", 0.0),
            blocked_penalty=shaping.get("blocked_penalty", 0.0),
        )

    if logic_reward:
        env = LogicRewardWrapper(env, cfg)

    if seed is not None:
        env.reset(seed=int(seed))
    return env


make_env = create_environment


def read_scene(env):
    u = env.unwrapped
    ego = u.vehicle
    lanes = u.config.get("lanes_count", 4)

    def vinfo(v):
        vx, vy = float(v.velocity[0]), float(v.velocity[1])
        lane = v.lane_index[2] if getattr(v, "lane_index", None) else 0
        return {
            "x": float(v.position[0]),
            "y": float(v.position[1]),
            "vx": vx, "vy": vy,
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
    def __init__(self, env, blocked_gap=25.0):
        super().__init__(env)
        self.blocked_gap = float(blocked_gap)
        self.ahead_ids = set()
        self.overtakes = 0
        self.lane_changes_count = 0
        self.lane = None
        self.blocked = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.overtakes = 0
        self.lane_changes_count = 0
        self.ahead_ids = self.currently_ahead()
        self.lane = self.current_lane()
        self.blocked = self.is_blocked()
        info = dict(info)
        info["overtakes"] = 0
        info["lane_changed"] = False
        info["lane_changes"] = 0
        info["escape_lane_change"] = False
        info["is_offroad"] = self.is_offroad()
        return obs, info

    def step(self, action):
        was_blocked = self.blocked
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.update_overtakes()
        lane = self.current_lane()
        info = dict(info)
        info["overtakes"] = self.overtakes
        info["lane_changed"] = bool(lane is not None and self.lane is not None
                                    and lane != self.lane)
        self.lane_changes_count += int(info["lane_changed"])
        info["lane_changes"] = self.lane_changes_count
        info["escape_lane_change"] = bool(was_blocked and lane != self.lane)
        self.lane = lane
        self.blocked = self.is_blocked()
        info["blocked"] = self.blocked
        info["is_offroad"] = self.is_offroad()
        return obs, reward, terminated, truncated, info

    def current_lane(self):
        ego = self.get_ego()
        idx = getattr(ego, "lane_index", None) if ego is not None else None
        return idx[2] if idx else None

    def is_blocked(self):
        ego = self.get_ego()
        if ego is None:
            return False
        lane = self.current_lane()
        for v in self.get_others():
            vidx = getattr(v, "lane_index", None)
            if vidx and vidx[2] == lane and 0 < (v.position[0] - ego.position[0]) < self.blocked_gap:
                return True
        return False

    def get_ego(self):
        return self.env.unwrapped.vehicle

    def get_others(self):
        ego = self.get_ego()
        return [v for v in self.env.unwrapped.road.vehicles if v is not ego]

    def currently_ahead(self):
        ego = self.get_ego()
        if ego is None:
            return set()
        return {id(v) for v in self.get_others() if v.position[0] > ego.position[0]}

    def update_overtakes(self):
        ego = self.get_ego()
        if ego is None:
            return
        ex = ego.position[0]
        still_ahead = set()
        for v in self.get_others():
            vx = v.position[0]
            vid = id(v)
            if vid in self.ahead_ids:
                if vx < ex:
                    self.overtakes += 1
                else:
                    still_ahead.add(vid)
            elif vx > ex:
                still_ahead.add(vid)
        self.ahead_ids = still_ahead

    def is_offroad(self):
        ego = self.get_ego()
        if ego is None:
            return False
        return not bool(getattr(ego, "on_road", True))


class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env, overtake_bonus=0.0, offroad_penalty=0.0, collision_penalty=0.0,
                 lane_change_bonus=0.0, survival_bonus=0.0, blocked_penalty=0.0):
        super().__init__(env)
        self.overtake_bonus = float(overtake_bonus)
        self.offroad_penalty = float(offroad_penalty)
        self.collision_penalty = float(collision_penalty)
        self.lane_change_bonus = float(lane_change_bonus)
        self.survival_bonus = float(survival_bonus)
        self.blocked_penalty = float(blocked_penalty)
        self.prev_overtakes = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_overtakes = int(info.get("overtakes", 0))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        overtakes = int(info.get("overtakes", self.prev_overtakes))
        passed = max(0, overtakes - self.prev_overtakes)
        self.prev_overtakes = overtakes

        shaped = reward + self.overtake_bonus * passed + self.survival_bonus
        if info.get("escape_lane_change", False):
            shaped += self.lane_change_bonus
        if info.get("blocked", False):
            shaped -= self.blocked_penalty
        if info.get("is_offroad", False):
            shaped -= self.offroad_penalty
        if info.get("crashed", False):
            shaped -= self.collision_penalty

        info = dict(info)
        info["native_reward"] = float(reward)
        info["shaped_reward"] = float(shaped)
        return obs, float(shaped), terminated, truncated, info


class LogicRewardWrapper(gym.Wrapper):
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
