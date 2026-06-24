"""Evaluation harness for all four parts.

  * ``evaluate``           — metrics over held-out seeds, with optional safety
    shield and per-rule violation counting (independent MTL monitor).
  * ``record_video``       — MP4 of one model driving (Parts 1-3).
  (Part 4's head-to-head race lives in ``eval/race.py``.)

Per-rule violations are counted by ``nesy.roadmap.rule_violations`` on the SI
scene, NOT by the shield/reward the agent sees, so "fewer violations" is not
circular. The scene adapter is pluggable via ``scene_fn`` so the same harness
serves highway-env and MetaDrive.

No top-level execution — the notebooks call these.
"""

import os

import numpy as np

from envs.highway_factory import make_env, read_scene


# =============================================================================
# Single-model evaluation (Parts 1-3)
# =============================================================================
def evaluate(model, cfg, seeds=None, apply_shield=False, count_violations=False,
             env_fn=None, scene_fn=None):
    """Evaluate ``model`` over held-out seeds and aggregate metrics.

    Args:
        model: trained SB3 model.
        cfg: full config.
        seeds: eval seeds; defaults to ``cfg['eval_seeds']``.
        apply_shield: wrap each action in the NeSy safety shield (Part 2/3).
        count_violations: report per-rule violation rates (independent monitor).
        env_fn: builder ``(cfg, render) -> env``; defaults to highway ``make_env``.
        scene_fn: ``env -> scene dict``; defaults to highway ``read_scene``.

    Returns:
        dict with ``summary`` (incl. ``rule_violation_rate``) and ``episodes``.
    """
    seeds = list(seeds) if seeds is not None else list(cfg["eval_seeds"])
    env_fn = env_fn or (lambda c, render: make_env(c, render=render))
    scene_fn = scene_fn or read_scene
    ec = cfg["eval"]

    env = env_fn(cfg, False)
    rows = []
    try:
        for seed in seeds:
            for ep in range(ec["episodes_per_seed"]):
                rows.append(_run_episode(
                    model, env, int(seed) * 100 + ep, cfg,
                    deterministic=ec["deterministic"],
                    apply_shield=apply_shield,
                    count_violations=count_violations,
                    scene_fn=scene_fn,
                ))
    finally:
        env.close()

    return {"summary": _summarise(rows, count_violations), "episodes": rows, "seeds": seeds}


def _run_episode(model, env, seed, cfg, deterministic, apply_shield,
                 count_violations, scene_fn):
    from nesy.roadmap import predicates, safety_shield, rule_violations

    obs, info = env.reset(seed=seed)
    fsm_state = cfg["fsm"]["initial_state"]
    done = False
    ret = native_ret = 0.0
    steps = offroad_steps = 0
    viol = {k: 0 for k in ("RG1", "RG2", "RG3", "RG4", "RI1", "RI2")}

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)

        if apply_shield or count_violations:
            scene = scene_fn(env)
            preds = predicates(scene, cfg)
            if apply_shield:
                action, fsm_state = safety_shield(action, preds, fsm_state, cfg)
            if count_violations:
                for k, v in rule_violations(preds, cfg).items():
                    viol[k] += int(v)

        obs, reward, terminated, truncated, info = env.step(action)
        ret += float(reward)
        native_ret += float(info.get("native_reward", reward))
        steps += 1
        if info.get("is_offroad", info.get("out_of_road", False)):
            offroad_steps += 1
        done = terminated or truncated

    # crash key differs by simulator (highway: 'crashed'; MetaDrive: 'crash').
    crashed = bool(info.get("crashed", info.get("crash", info.get("crash_vehicle", False))))
    row = {
        "seed": seed,
        "crashed": crashed,
        "on_road_pct": 100.0 * (1.0 - offroad_steps / steps) if steps else 0.0,
        "overtakes": int(info.get("overtakes", 0)),
        "return": ret,
        "native_return": native_ret,
        "length": steps,
    }
    if count_violations:
        row["violations"] = viol
        row["viol_steps"] = steps
    return row


def _summarise(rows, count_violations):
    def ms(key):
        vals = np.array([r[key] for r in rows], dtype=float)
        return {"mean": float(vals.mean()), "std": float(vals.std())}

    n = len(rows)
    summary = {
        "n_episodes": n,
        "crash_rate": sum(1 for r in rows if r["crashed"]) / n if n else 0.0,
        "on_road_pct": ms("on_road_pct"),
        "overtakes": ms("overtakes"),
        "return": ms("return"),
        "native_return": ms("native_return"),
        "length": ms("length"),
    }
    if count_violations:
        total_steps = sum(r["viol_steps"] for r in rows) or 1
        rates = {}
        for k in ("RG1", "RG2", "RG3", "RG4", "RI1", "RI2"):
            rates[k] = sum(r["violations"][k] for r in rows) / total_steps
        summary["rule_violation_rate"] = rates  # fraction of steps violating each rule
    return summary


# =============================================================================
# Single-model video (Parts 1-3)
# =============================================================================
def record_video(model, cfg, path, n_episodes=None, seed=None,
                 apply_shield=False, env_fn=None, scene_fn=None):
    """Record an MP4 of the policy driving, saved to ``path``. Returns ``path``.

    Captures ``env.render()`` frames manually and writes them with imageio — this
    works uniformly for highway-env and MetaDrive (no dependence on
    ``RecordVideo``'s render-mode handling, which differs across simulators).
    """
    import imageio
    from nesy.roadmap import predicates, safety_shield

    n_episodes = n_episodes if n_episodes is not None else cfg["eval"].get("video_episodes", 1)
    seed = seed if seed is not None else cfg["eval_seeds"][0]
    env_fn = env_fn or (lambda c, render: make_env(c, render=render))
    scene_fn = scene_fn or read_scene
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    env = env_fn(cfg, True)
    frames = []
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=int(seed) + ep)
            fsm_state = cfg["fsm"]["initial_state"]
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=cfg["eval"]["deterministic"])
                if apply_shield:
                    preds = predicates(scene_fn(env), cfg)
                    action, fsm_state = safety_shield(action, preds, fsm_state, cfg)
                obs, _, terminated, truncated, _ = env.step(action)
                frame = env.render()
                if frame is not None:
                    frames.append(np.asarray(frame))
                done = terminated or truncated
    finally:
        env.close()

    if frames:
        imageio.mimsave(path, frames, fps=10)
    return path


def record_random_video(cfg, path, n_steps=None, seed=None, env_fn=None):
    """Record a short random-policy rollout (Part-1 environment study).

    Lets the reader see the task before any learning. Returns ``path``.
    """
    import imageio

    seed = seed if seed is not None else cfg["eval_seeds"][0]
    n_steps = n_steps if n_steps is not None else cfg["env"]["config"].get("duration", 40)
    env_fn = env_fn or (lambda c, render: make_env(c, render=render))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    env = env_fn(cfg, True)
    frames = []
    try:
        env.reset(seed=int(seed))
        for _ in range(int(n_steps)):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))
            if terminated or truncated:
                env.reset()
    finally:
        env.close()

    if frames:
        imageio.mimsave(path, frames, fps=10)
    return path

# Part 4 (head-to-head race) lives in eval/race.py: race(), record_race_video().
