"""Evaluation harness for Part 1.

Runs a trained model over a fixed set of held-out seeds (identical for PPO and
DQN) and reports the Part-1 metrics: crash rate, on-road %, overtakes/episode,
return, and episode length, as mean ± std over all evaluated episodes. Also
records an MP4 rollout for qualitative inspection.

The per-episode rows are returned too, so the Part-1 "study" can catalogue
failure modes (and so Part 2 can diff against these exact numbers).

No top-level execution — called from the notebook.
"""

import numpy as np
import gymnasium as gym

from envs.highway_factory import make_env


def evaluate(model, cfg, seeds=None):
    """Evaluate ``model`` over held-out seeds and aggregate metrics.

    Args:
        model: a trained SB3 model (PPO or DQN).
        cfg: full project config dict.
        seeds: iterable of eval seeds; defaults to ``cfg['eval_seeds']``.

    Returns:
        dict with ``summary`` (mean/std per metric) and ``episodes`` (per-episode
        rows). ``summary`` is the table the study reports.
    """
    seeds = list(seeds) if seeds is not None else list(cfg["eval_seeds"])
    eval_cfg = cfg["eval"]
    episodes_per_seed = eval_cfg["episodes_per_seed"]
    deterministic = eval_cfg["deterministic"]

    env = make_env(cfg, render=False)
    rows = []
    try:
        for seed in seeds:
            for ep in range(episodes_per_seed):
                # Distinct, reproducible seed per episode.
                ep_seed = int(seed) * 100 + ep
                rows.append(_run_episode(model, env, ep_seed, deterministic))
    finally:
        env.close()

    return {"summary": _summarise(rows), "episodes": rows, "seeds": seeds}


def _run_episode(model, env, seed, deterministic):
    """Roll out one episode and return its metric row."""
    obs, info = env.reset(seed=seed)
    done = False
    ret = 0.0
    native_ret = 0.0
    steps = 0
    offroad_steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        ret += float(reward)
        native_ret += float(info.get("native_reward", reward))
        steps += 1
        if info.get("is_offroad", False):
            offroad_steps += 1
        done = terminated or truncated

    crashed = bool(info.get("crashed", False))
    overtakes = int(info.get("overtakes", 0))
    on_road_pct = 100.0 * (1.0 - offroad_steps / steps) if steps else 0.0

    return {
        "seed": seed,
        "crashed": crashed,
        "on_road_pct": on_road_pct,
        "overtakes": overtakes,
        "return": ret,
        "native_return": native_ret,
        "length": steps,
    }


def _summarise(rows):
    """Aggregate per-episode rows into mean/std (and crash rate)."""
    def ms(key):
        vals = np.array([r[key] for r in rows], dtype=float)
        return {"mean": float(vals.mean()), "std": float(vals.std())}

    n = len(rows)
    crashes = sum(1 for r in rows if r["crashed"])
    return {
        "n_episodes": n,
        "crash_rate": crashes / n if n else 0.0,
        "on_road_pct": ms("on_road_pct"),
        "overtakes": ms("overtakes"),
        "return": ms("return"),
        "native_return": ms("native_return"),
        "length": ms("length"),
    }


def record_video(model, cfg, path, n_episodes=None, seed=None):
    """Record an MP4 of the policy driving, saved to ``path``.

    Uses Gymnasium's ``RecordVideo`` over a render-mode env. Returns the output
    path so the notebook can mirror it to Drive.
    """
    import os

    n_episodes = n_episodes if n_episodes is not None else cfg["eval"].get("video_episodes", 1)
    seed = seed if seed is not None else cfg["eval_seeds"][0]
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    name_prefix = os.path.splitext(os.path.basename(path))[0]

    env = make_env(cfg, render=True)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=out_dir,
        name_prefix=name_prefix,
        episode_trigger=lambda e: e < n_episodes,
    )
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=int(seed) + ep)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=cfg["eval"]["deterministic"])
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
    finally:
        env.close()

    return out_dir
