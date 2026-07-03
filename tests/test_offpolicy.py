"""Small, fast smoke tests for the Part-1 off-policy stack (SAC vs MACURA).

Micro budgets only — these verify the MACHINERY (hybrid env, SAC update, MACURA's
ensemble -> adaptive-κ rollouts -> Eq. 22 updates, dual stopping, the evaluation
harness and the training-curve CSV) in ~1-2 min on CPU. They never do real
training; the 3k head-to-head runs in the notebook on Colab.

Run:  python tests/test_offpolicy.py        (or: pytest tests/test_offpolicy.py)
"""
import os
import sys
import time
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import numpy as np
import torch

from utils import load_config


def micro_cfg():
    """The real YAML shrunk to seconds-scale quantities (algorithm logic untouched)."""
    cfg = load_config(os.path.join(_REPO, "configs", "highway.yaml"))
    cfg["device"] = "cpu"
    cfg["print_freq"] = 60
    cfg["paths"]["drive_root"] = tempfile.mkdtemp(prefix="nesy_test_")
    cfg["eval_seeds"] = [1000]
    cfg["eval"]["episodes_per_seed"] = 1
    cfg["env"]["config"]["vehicles_count"] = 10
    cfg["env"]["config"]["duration"] = 15                    # 30-step episodes
    cfg["offpolicy"]["shared"].update(
        total_steps=120, warmup_steps=40, epoch_length=40,
        max_episode_steps=30, batch_size=64, buffer_size=2000)
    cfg["offpolicy"]["ensemble"].update(
        num_models=2, num_elites=2, hidden_dim=32, hidden_layers=2,
        max_epochs=2, batch_size=64, patience=2)
    cfg["offpolicy"]["model_based"].update(
        freq_train_model=40, effective_rollouts_per_step=5,
        max_rollout_length=4, term_max_epochs=2)
    cfg["offpolicy"]["curriculum"]["real_ratio_knots"] = [
        {"step": 0, "real_ratio": 1.0}, {"step": 80, "real_ratio": 0.5}]
    return cfg


def test_hybrid_env():
    """Hybrid (lane_cmd, speed_cmd) env: spaces, info keys, speed mapping."""
    from envs.highway_factory import create_hybrid_environment

    cfg = micro_cfg()
    env = create_hybrid_environment(cfg)
    assert env.action_space.shape == (2,)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (25,) and obs.dtype == np.float32
    for _ in range(5):
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
    assert {"overtakes", "crashed", "speed"} <= set(info)
    env.reset(seed=1)
    env.step(np.array([0.0, 1.0], dtype=np.float32))         # speed_cmd = +1 -> v_hi
    v_hi = float(cfg["offpolicy"]["env"]["v_range"][1])
    assert abs(env.unwrapped.vehicle.target_speed - v_hi) < 1e-6
    env.close()


def test_sac_update():
    """One SAC gradient step on random data: finite losses, deterministic eval action."""
    from agents.offpolicy.trainers import build_offpolicy

    cfg = micro_cfg()
    agent = build_offpolicy(cfg, "sac")
    rng = np.random.default_rng(0)
    for _ in range(200):
        agent.store(rng.standard_normal(25).astype(np.float32),
                    rng.uniform(-1, 1, 2).astype(np.float32),
                    float(rng.normal()),
                    rng.standard_normal(25).astype(np.float32), 0.0)
    m = agent.update_step(1)
    assert np.isfinite(m["critic_loss"]) and np.isfinite(m["actor_loss"])
    a1 = agent.act(np.zeros(25, np.float32), explore=False)
    a2 = agent.act(np.zeros(25, np.float32), explore=False)
    assert a1.shape == (2,) and np.allclose(a1, a2)           # tanh(μ) is deterministic


def test_dual_stopping_mask():
    """TL mask on constructed states: over-limit and unsafe-gap fire, safe gap doesn't."""
    from agents.offpolicy.dual_stopping import tl_violation_mask

    rules = micro_cfg()["rules"]

    def state(v_ego, leader_gap=None):
        # rows of [presence, x, y, vx, vy]; ego vx normalised by 2*40, x by 5*40
        s = torch.zeros(25)
        s[0], s[3] = 1.0, v_ego / 80.0
        if leader_gap is not None:
            s[5], s[6] = 1.0, leader_gap / 200.0              # same lane, same speed
        return s

    states = torch.stack([
        state(30.0),          # RG3: 30 m/s > v_max 25            -> violation
        state(20.0, 100.0),   # legal speed, leader 100 m ahead   -> safe
        state(20.0, 8.0),     # leader 8 m ahead < RSS safe gap   -> violation
    ])
    mask = tl_violation_mask(states, rules, lanes_count=4, features_per_vehicle=5)
    assert mask.tolist() == [True, False, True]


def test_train_loop_sac():
    """train_offpolicy end to end (micro): checkpoint + training-curve CSV written."""
    from agents.offpolicy.trainers import train_offpolicy

    cfg = micro_cfg()
    train_offpolicy(cfg, "sac")
    root = cfg["paths"]["drive_root"]
    assert os.path.exists(os.path.join(root, "checkpoints", "sac_policy.pt"))
    csv = os.path.join(root, "metrics", "curves", "sac", "progress.csv")
    with open(csv) as f:
        lines = f.read().strip().split("\n")
    assert lines[0].startswith("time/total_timesteps") and len(lines) >= 2


def test_train_loop_macura():
    """MACURA's full Dyna loop (micro): ensemble trains, κ is set, rollouts stored."""
    from agents.offpolicy.trainers import train_offpolicy

    cfg = micro_cfg()
    agent = train_offpolicy(cfg, "macura")
    assert agent.kappa > 0.0                                  # adaptive threshold calibrated
    assert len(agent.model_buffer) > 0                        # imagined transitions stored
    root = cfg["paths"]["drive_root"]
    assert os.path.exists(os.path.join(root, "checkpoints", "macura_policy.pt"))
    assert os.path.exists(os.path.join(root, "checkpoints", "macura_ensemble.pt"))


def test_evaluate_hybrid():
    """Shared evaluate harness on the hybrid env via as_predictor, incl. distance."""
    from agents.offpolicy.trainers import build_offpolicy, as_predictor
    from envs.highway_factory import create_hybrid_environment
    from eval.evaluate import evaluate

    cfg = micro_cfg()
    agent = build_offpolicy(cfg, "sac")
    m = evaluate(as_predictor(agent), cfg,
                 env_fn=lambda c, render: create_hybrid_environment(c))
    s = m["summary"]
    assert s["n_episodes"] == 1
    assert {"crash_rate", "overtakes", "distance", "length", "on_road_pct"} <= set(s)
    assert np.isfinite(s["distance"]["mean"])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t0 = time.time()
        t()
        print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)", flush=True)
    print(f"all {len(tests)} tests passed")
