"""Verify the overtake counter (envs.highway_factory.OvertakeCounter) counts the
RIGHT number in scenarios where the correct answer is known exactly.

Motivation: in the 30k RPPO run the overtakes/ep FELL as the crash rate fell, and
we needed to know whether that is a counting bug or real behaviour. These tests
drive a fake env through scripted vehicle x-positions (so every overtake is
hand-countable) and assert the wrapper's ``info['overtakes']`` matches.

Run:  python tests/test_overtake_counter.py     (or: pytest tests/test_overtake_counter.py)
"""
import os
import sys
import time
from types import SimpleNamespace

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import numpy as np
import gymnasium as gym

from envs.highway_factory import OvertakeCounter


class _Veh:
    def __init__(self, x, lane=1):
        self.position = np.array([float(x), 0.0])
        self.lane_index = ("a", "b", lane)
        self.on_road = True
        self.speed = 20.0


class _ScriptEnv(gym.Env):
    """Minimal env whose ego + other vehicles follow scripted x-positions.

    ``ego_xs[t]`` and ``others_xs[t][i]`` give the longitudinal position of the
    ego and each neighbour at frame ``t``. Everything else the wrapper reads
    (lane_index, on_road, road.vehicles, config) is faked."""

    metadata = {"render_modes": []}

    def __init__(self, ego_xs, others_xs, lanes=None):
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (5, 5), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(5)
        self.render_mode = None
        self._ego_xs = ego_xs
        self._others_xs = others_xs
        n = len(others_xs[0])
        lanes = lanes or [1] * n
        self.vehicle = _Veh(ego_xs[0], lane=0)
        self._others = [_Veh(others_xs[0][i], lanes[i]) for i in range(n)]
        self.road = SimpleNamespace(vehicles=[self.vehicle] + self._others)
        self.config = {"lanes_count": 4}
        self._t = 0

    def _apply(self, t):
        t = min(t, len(self._ego_xs) - 1)
        self.vehicle.position[0] = float(self._ego_xs[t])
        for i, v in enumerate(self._others):
            v.position[0] = float(self._others_xs[t][i])

    def reset(self, **kwargs):
        self._t = 0
        self._apply(0)
        return np.zeros((5, 5), np.float32), {}

    def step(self, action):
        self._t += 1
        self._apply(self._t)
        term = self._t >= len(self._ego_xs) - 1
        return np.zeros((5, 5), np.float32), 0.0, term, False, {}


def _run(ego_xs, others_xs, lanes=None):
    """Drive the scripted env through the wrapper; return final overtake count."""
    env = OvertakeCounter(_ScriptEnv(ego_xs, others_xs, lanes))
    _, info = env.reset()
    last = info["overtakes"]
    for _ in range(len(ego_xs) - 1):
        _, _, term, trunc, info = env.step(0)
        last = info["overtakes"]
        if term or trunc:
            break
    return last


def test_ego_passes_one_car():
    """Ego (fast) passes one slow car ahead -> exactly 1."""
    assert _run(ego_xs=[0, 10, 20, 30], others_xs=[[15], [17], [19], [21]]) == 1


def test_car_passes_ego_not_counted():
    """A faster car overtakes the SLOW ego (behind -> ahead) -> 0 (not the ego's pass)."""
    assert _run(ego_xs=[0, 2, 4, 6], others_xs=[[-10], [-2], [6], [14]]) == 0


def test_repass_counts_twice():
    """Ego passes a car, the car re-passes, the ego passes again -> 2 genuine passes."""
    assert _run(ego_xs=[0, 20, 20, 40], others_xs=[[10], [15], [30], [35]]) == 2


def test_passes_three_of_five():
    """Five cars ahead; ego passes three, two pull away and stay ahead -> 3."""
    assert _run(ego_xs=[0, 50],
                others_xs=[[10, 20, 30, 40, 45], [11, 21, 31, 100, 100]]) == 3


def test_pass_is_lane_independent():
    """Overtake is on longitudinal x — a car two lanes over still counts when passed."""
    assert _run(ego_xs=[0, 20], others_xs=[[10], [15]], lanes=[2]) == 1


def test_camping_same_speed_zero():
    """Ego cruises at traffic speed (a leader stays 10 m ahead all episode) -> 0.

    THIS is the 30k-run situation: long survival, ~0 overtakes is CORRECT, because
    a car that never advances past anyone overtakes no one. Not a counting bug."""
    assert _run(ego_xs=[0, 2, 4, 6], others_xs=[[10], [12], [14], [16]]) == 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t0 = time.time()
        t()
        print(f"PASS {t.__name__} ({time.time() - t0:.2f}s)", flush=True)
    print(f"all {len(tests)} overtake-counter tests passed")
