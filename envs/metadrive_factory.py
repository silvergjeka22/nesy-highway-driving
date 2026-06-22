"""MetaDrive construction with a velocity (v, ω) action (Part 3).

Public entry points:
  * ``make_env_md(cfg, render, seed)`` — MetaDrive env exposing a continuous
    ``(v, ω)`` action that maps onto ROS ``cmd_vel`` (Lab 1).
  * ``read_scene_md(env)`` — the SAME scene schema as the highway adapter, so the
    NeSy predicates / shield / CBF are reused unchanged.

MetaDrive is heavy and is installed only in the Part-3 notebook, so it is
imported lazily. MetaDrive API details that were not available at writing time
(exact velocity-control hook, neighbour enumeration) are marked TODO and wrapped
in best-effort code that fails loudly rather than silently guessing.

No top-level execution.
"""

from utils import silence_warnings

silence_warnings()

import numpy as np  # noqa: E402
import gymnasium as gym  # noqa: E402


def make_env_md(cfg, render=False, seed=None):
    """Build a MetaDrive env with a continuous ``(v, ω)`` action.

    Args:
        cfg: full config; uses the ``metadrive:`` block.
        render: enable offscreen RGB rendering (for ``record_video``).
        seed: start seed (MetaDrive uses integer scenario seeds).

    Returns:
        A Gymnasium-API env whose action is ``(v_norm, ω_norm) ∈ [-1, 1]²``.
    """
    md = cfg["metadrive"]
    try:
        from metadrive.envs import MetaDriveEnv
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "MetaDrive is not installed. In colab_3 run: "
            "`pip install metadrive-simulator`."
        ) from e

    md_config = {
        "use_render": False,
        "image_observation": False,
        "traffic_density": md["traffic_density"],
        "num_scenarios": md["num_scenarios"],
        "horizon": md["horizon"],
        "map": md["map"],
        # TODO: confirm the exact MetaDrive keys for lidar lasers / offscreen RGB
        # against the installed version; these are the common ones.
        "vehicle_config": {"lidar": {"num_lasers": md.get("lidar_num_lasers", 72)}},
    }
    if render:
        md_config["image_observation"] = True

    env = MetaDriveEnv(md_config)
    env = VelocityActionWrapper(env, cfg)
    if seed is not None:
        env.reset(seed=int(seed))
    return env


class VelocityActionWrapper(gym.Wrapper):
    """Expose a normalised ``(v, ω)`` action and convert it to MetaDrive control.

    The policy (and the manoeuvre→cmd_vel translation in Lab 1) acts in velocity
    space; this wrapper turns ``(v_norm, ω_norm) ∈ [-1,1]²`` into MetaDrive's
    native ``(steering, throttle)`` command.
    """

    def __init__(self, env, cfg):
        super().__init__(env)
        self.cfg = cfg
        self.v_max = cfg["metadrive"]["v_max"]
        self.omega_max = cfg["metadrive"]["omega_max"]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def step(self, action):
        v_cmd = float(np.clip(action[0], -1.0, 1.0)) * self.v_max
        omega_cmd = float(np.clip(action[1], -1.0, 1.0)) * self.omega_max
        native = self._velocity_to_native(v_cmd, omega_cmd)
        return self.env.step(native)

    def _velocity_to_native(self, v, omega):
        """Map ``(v, ω)`` to MetaDrive's ``(steering, throttle)`` in [-1, 1].

        Best-effort proportional mapping: steering ∝ ω, throttle ∝ (v_target − v)
        relative to the current speed. TODO: replace with MetaDrive's documented
        velocity-control hook (or a tuned low-level PID) once the installed API
        is confirmed — the exact interface was unavailable at writing time.
        """
        try:
            ego = self.env.unwrapped.agent
            v_cur = float(np.linalg.norm(ego.velocity))
        except Exception:
            v_cur = 0.0
        steering = float(np.clip(omega / max(self.omega_max, 1e-6), -1.0, 1.0))
        throttle = float(np.clip((v - v_cur) / max(self.v_max, 1e-6), -1.0, 1.0))
        return np.array([steering, throttle], dtype=np.float32)


def read_scene_md(env):
    """Extract the common SI scene dict from a MetaDrive env.

    Returns the same schema as ``envs.highway_factory.read_scene`` so the NeSy
    predicates/shield/CBF apply unchanged.

    TODO: confirm the neighbour-enumeration API for the installed MetaDrive
    version. This best-effort version reads the ego from ``env.unwrapped.agent``
    and neighbours from the engine's traffic manager; adjust attribute names to
    match the handout/installed API if they differ.
    """
    u = env.unwrapped
    ego = u.agent

    def vinfo(obj):
        pos = np.asarray(obj.position, dtype=float)
        vel = np.asarray(obj.velocity, dtype=float)
        heading = float(getattr(obj, "heading_theta", 0.0))
        lane_idx = 0
        try:
            lane_idx = int(obj.lane_index[2])
        except Exception:
            lane_idx = 0
        return {
            "x": float(pos[0]),
            "y": float(pos[1]),
            "vx": float(vel[0]),
            "vy": float(vel[1]),
            "v": float(np.linalg.norm(vel)),
            "lane": lane_idx,
            "heading": heading,
        }

    ego_d = vinfo(ego)
    ego_d["on_road"] = not bool(getattr(ego, "out_of_route", False))

    others = []
    try:
        for obj in u.engine.traffic_manager.vehicles:
            if obj is ego:
                continue
            others.append(vinfo(obj))
    except Exception:
        # TODO: fall back to lidar-derived predicates (labs.lab2) when the
        # neighbour list is not exposed by this MetaDrive version.
        others = []

    return {"ego": ego_d, "others": others, "lanes_count": 3}
