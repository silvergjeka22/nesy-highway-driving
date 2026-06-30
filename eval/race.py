"""Part 4 — head-to-head race: baseline (A) vs NeSy (B) in ONE shared scene.

Tier 1 of the plan: both agents are already trained (Parts 1-2) and are deployed
*together* in a multi-agent highway-env (``controlled_vehicles = 2``). Each agent
runs its own policy on its own local observation; agent B additionally runs the
NeSy safety shield. We score not just who is ahead, but who stays safe and
rule-compliant under competitive pressure — so the scorecard always pairs
finishing progress with crash + per-rule-violation metrics.

Function-only. Multi-agent MetaDrive (MARL) is the Tier-2 stretch and is left as
a TODO in ``make_race_env``.

Public API (see README.md, Part 4):
    make_race_env(cfg, n_agents=2, render=False) -> env
    race(model_a, model_b, cfg, seeds=None)      -> race_metrics
    record_race_video(model_a, model_b, cfg, path) -> path
"""

import os
import copy

import numpy as np

from utils import silence_warnings

silence_warnings()

import gymnasium as gym  # noqa: E402
import highway_env  # noqa: F401,E402


def make_race_env(cfg, n_agents=2, render=False):
    """Build a multi-agent highway-env with ``n_agents`` controlled vehicles.

    Each agent gets its own Kinematics observation and DiscreteMetaAction; the
    other controlled car appears in each agent's view as a nearby vehicle. Same
    road/traffic config as Parts 1-2 so the race is on the trained distribution.

    TODO (Tier 2): a MetaDrive MARL variant (native multi-agent dict env) riding
    the Part-3 velocity action + CBF/VO stack.
    """
    env_cfg = copy.deepcopy(cfg["env"]["config"])
    obs_cfg = copy.deepcopy(env_cfg["observation"])
    act_cfg = copy.deepcopy(env_cfg["action"])
    env_cfg["controlled_vehicles"] = n_agents
    env_cfg["observation"] = {"type": "MultiAgentObservation", "observation_config": obs_cfg}
    env_cfg["action"] = {"type": "MultiAgentAction", "action_config": act_cfg}

    env = gym.make(
        cfg["env"]["id"],
        render_mode="rgb_array" if render else None,
        config=env_cfg,
    )
    return env


def race(model_a, model_b, cfg, seeds=None):
    """Run N races on fixed seeds; report per-agent progress + safety.

    A = baseline (no shield), B = NeSy (shield on). Winner per race by
    ``cfg['race']['winner_metric']`` ("distance" progress or "overtakes").
    """
    rc = cfg["race"]
    seeds = list(seeds) if seeds is not None else list(rc["seeds"])
    metric = rc["winner_metric"]

    env = make_race_env(cfg, n_agents=2)
    rows = []
    try:
        for seed in seeds:
            rows.append(_run_race(model_a, model_b, env, int(seed), cfg))
    finally:
        env.close()

    wins = {"A": 0, "B": 0, "tie": 0}
    for r in rows:
        a, b = r["A"][metric], r["B"][metric]
        r["winner"] = "B" if b > a else ("A" if a > b else "tie")
        wins[r["winner"]] += 1
    return {"races": rows, "wins": wins, "metric": metric}


def _run_race(model_a, model_b, env, seed, cfg):
    from nesy.roadmap import predicates, safety_shield

    obs, info = env.reset(seed=seed)
    fsm_b = cfg["fsm"]["initial_state"]
    interventions_b = 0
    steps = 0
    done = False
    while not done and steps < cfg["race"]["horizon"]:
        a_act, _ = model_a.predict(obs[0], deterministic=True)
        b_act, _ = model_b.predict(obs[1], deterministic=True)

        # Agent B runs the NeSy shield on its own local scene.
        preds_b = predicates(_scene_for(env, 1, cfg), cfg)
        b_shielded, fsm_b = safety_shield(b_act, preds_b, fsm_b, cfg)
        if int(b_shielded) != int(b_act):
            interventions_b += 1

        obs, reward, terminated, truncated, info = env.step((int(a_act), int(b_shielded)))
        done = _is_done(terminated) or _is_done(truncated)
        steps += 1

    a = _agent_outcome(env, 0, cfg)
    b = _agent_outcome(env, 1, cfg)
    b["intervention_rate"] = interventions_b / steps if steps else 0.0
    return {"seed": seed, "steps": steps, "A": a, "B": b}


def record_race_video(model_a, model_b, cfg, path, seed=None):
    """Save an MP4 of one shared-scene race (both agents racing). Returns ``path``.

    Both controlled cars are in the same frame; A is the baseline, B runs the
    shield. (For a two-pane comparison instead, render two single-agent envs and
    hstack the frames.)
    """
    from utils import save_mp4
    from nesy.roadmap import predicates, safety_shield

    rc = cfg["race"]
    seed = seed if seed is not None else rc["seeds"][0]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    env = make_race_env(cfg, n_agents=2, render=True)
    frames = []
    try:
        obs, _ = env.reset(seed=int(seed))
        fsm_b = cfg["fsm"]["initial_state"]
        done = False
        steps = 0
        while not done and steps < rc["horizon"]:
            a_act, _ = model_a.predict(obs[0], deterministic=True)
            b_act, _ = model_b.predict(obs[1], deterministic=True)
            preds_b = predicates(_scene_for(env, 1, cfg), cfg)
            b_act, fsm_b = safety_shield(b_act, preds_b, fsm_b, cfg)

            obs, _, terminated, truncated, _ = env.step((int(a_act), int(b_act)))
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))
            done = _is_done(terminated) or _is_done(truncated)
            steps += 1
    finally:
        env.close()

    return save_mp4(frames, path, fps=cfg["eval"].get("video_fps", 10))


# ---- helpers ----------------------------------------------------------------
def _scene_for(env, idx, cfg):
    """Build the SI scene dict for controlled vehicle ``idx`` (ego = that car)."""
    u = env.unwrapped
    ego = u.controlled_vehicles[idx]
    lanes = u.config.get("lanes_count", 4)

    def vinfo(v):
        vx, vy = float(v.velocity[0]), float(v.velocity[1])
        lane = v.lane_index[2] if getattr(v, "lane_index", None) else 0
        return {"x": float(v.position[0]), "y": float(v.position[1]),
                "vx": vx, "vy": vy,
                "v": float(getattr(v, "speed", (vx ** 2 + vy ** 2) ** 0.5)),
                "lane": int(lane)}

    ego_d = vinfo(ego)
    ego_d["on_road"] = bool(getattr(ego, "on_road", True))
    ego_d["heading"] = float(getattr(ego, "heading", 0.0))
    others = [vinfo(v) for v in u.road.vehicles if v is not ego]
    return {"ego": ego_d, "others": others, "lanes_count": int(lanes)}


def _agent_outcome(env, idx, cfg):
    """Per-agent race outcome: progress, crash, and per-rule violations."""
    from nesy.roadmap import predicates, rule_violations

    veh = env.unwrapped.controlled_vehicles[idx]
    scene = _scene_for(env, idx, cfg)
    viol = rule_violations(predicates(scene, cfg), cfg)
    return {
        "distance": float(veh.position[0]),
        "overtakes": 0,  # TODO: per-agent overtake counter in the multi-agent env
        "crashed": bool(getattr(veh, "crashed", False)),
        "violations_final": viol,
    }


def _is_done(flag):
    """Normalise a scalar/tuple terminated|truncated flag to a single bool."""
    if isinstance(flag, (tuple, list, np.ndarray)):
        return bool(np.any(flag))
    return bool(flag)
