"""Small, fast smoke tests for the Part-1 discrete stack (RecurrentPPO / DQN / QR-DQN).

Micro budgets only — these verify the MACHINERY (discrete env + lane-change flag,
the three trainers end to end with checkpoint + training-curve CSV, the stateful
RecurrentPPO facade, and the evaluation harness) in ~1-2 min on CPU. They never do
real training; the shared-budget comparison runs in the notebook on Colab.

Run:  python tests/test_baselines.py        (or: pytest tests/test_baselines.py)
"""
import os
import sys
import time
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import numpy as np

from utils import load_config


def micro_cfg():
    """The real YAML shrunk to seconds-scale quantities (hyperparameter logic untouched)."""
    cfg = load_config(os.path.join(_REPO, "configs", "highway.yaml"))
    cfg["device"] = "cpu"
    cfg["print_freq"] = 60
    cfg["paths"]["drive_root"] = tempfile.mkdtemp(prefix="nesy_test_")
    cfg["eval_seeds"] = [1000]
    cfg["eval"]["episodes_per_seed"] = 1
    cfg["env"]["config"]["vehicles_count"] = 10
    cfg["env"]["config"]["duration"] = 15                    # 30-step episodes
    cfg["train"]["total_timesteps"] = 150
    cfg["rppo"].update(n_steps=64, batch_size=64, n_epochs=2)
    for q in ("dqn", "qrdqn"):
        cfg[q].update(learning_starts=50, buffer_size=2000, target_update_interval=100)
    return cfg


def test_discrete_env():
    """Discrete meta-action env: spaces, info keys, the lane_changed flag fires."""
    from envs.highway_factory import create_environment

    cfg = micro_cfg()
    env = create_environment(cfg)
    assert env.action_space.n == 5                            # the 5 meta-actions
    obs, info = env.reset(seed=0)
    assert obs.shape == (5, 5) and info["lane_changed"] is False
    saw_lane_change = False
    for i in range(20):                                       # steer right then left
        obs, r, terminated, truncated, info = env.step(2 if i < 10 else 0)
        assert {"overtakes", "lane_changed", "crashed"} <= set(info)
        saw_lane_change = saw_lane_change or info["lane_changed"]
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    assert saw_lane_change                                    # commanded changes are counted


def test_train_loop_rppo():
    """train_rppo end to end (micro): best checkpoint + training-curve CSV written."""
    from agents.baselines import train_rppo

    cfg = micro_cfg()
    model = train_rppo(cfg)
    assert model is not None
    _assert_artifacts(cfg, "rppo")


def test_train_loop_dqn():
    from agents.baselines import train_dqn

    cfg = micro_cfg()
    train_dqn(cfg)
    _assert_artifacts(cfg, "dqn")


def test_train_loop_qrdqn():
    from agents.baselines import train_qrdqn

    cfg = micro_cfg()
    train_qrdqn(cfg)
    _assert_artifacts(cfg, "qrdqn")


def _assert_artifacts(cfg, tag):
    root = cfg["paths"]["drive_root"]
    assert os.path.exists(os.path.join(root, "checkpoints", f"{tag}.zip"))
    csv = os.path.join(root, "metrics", "curves", tag, "progress.csv")
    with open(csv) as f:
        header = f.readline()
    assert "ep_rew_mean" in header


def test_recurrent_predictor():
    """The stateful LSTM facade: carries hidden state, resets per episode."""
    from envs.highway_factory import create_environment
    from agents.baselines import build_rppo, as_predictor

    cfg = micro_cfg()
    env = create_environment(cfg)
    model = build_rppo(cfg, env)
    pred = as_predictor(model, "rppo")
    obs, _ = env.reset(seed=0)
    a1, _ = pred.predict(obs, deterministic=True)
    assert int(a1) in range(5)
    assert pred._state is not None and pred._episode_start is False   # state carried
    pred.reset_states()
    assert pred._state is None and pred._episode_start is True        # fresh episode
    assert as_predictor(model, "dqn") is model                        # non-recurrent: pass-through
    env.close()


def test_evaluate_discrete():
    """Shared evaluate harness: summary includes distance and lane_changes."""
    from agents.baselines import build_dqn
    from envs.highway_factory import create_environment
    from eval.evaluate import evaluate

    cfg = micro_cfg()
    env = create_environment(cfg)
    model = build_dqn(cfg, env)
    env.close()
    m = evaluate(model, cfg)
    s = m["summary"]
    assert s["n_episodes"] == 1
    assert {"crash_rate", "overtakes", "lane_changes", "distance", "length",
            "on_road_pct"} <= set(s)
    assert np.isfinite(s["distance"]["mean"]) and np.isfinite(s["lane_changes"]["mean"])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t0 = time.time()
        t()
        print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)", flush=True)
    print(f"all {len(tests)} tests passed")
