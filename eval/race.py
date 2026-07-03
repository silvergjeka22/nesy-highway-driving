"""Part 4 — head-to-head race: baseline (A) vs NeSy (B) in ONE shared scene.

Both agents are already trained (Parts 1-2) and are deployed *together* in a
multi-agent highway-env (``controlled_vehicles = 2``). Each agent runs its own
policy on its own local observation; agent B additionally runs the NeSy safety
shield. The scorecard never reports the winner alone: every race pairs progress
(distance, overtakes) with crashes and per-rule violation rates, because a
"be ahead" incentive rewards aggression.

To cancel positional bias, the agents swap start slots on alternate seeds.

Public API (see README.md, Part 4):
    make_race_env(cfg, n_agents=2, render=False) -> env
    race(model_a, model_b, cfg, seeds=None)      -> race_metrics
    record_race_video(model_a, model_b, cfg, path) -> path
"""

import os
import copy

import numpy as np

# The factory import must come first: it blocks pygame's pkg_resources import
# (fixing the deprecation warning at the source) and registers highway-env.
from envs.highway_factory import _ensure_render_backend

import gymnasium as gym  # noqa: E402


def make_race_env(cfg, n_agents=2, render=False):
    """Build a multi-agent highway-env with ``n_agents`` controlled vehicles.

    Each agent gets its own Kinematics observation and DiscreteMetaAction; the
    other controlled car appears in each agent's view as a nearby vehicle. Same
    road/traffic config as Parts 1-2 (so the race is on the trained
    distribution), except a longer ``race.duration`` so there is room to pass.
    """
    env_cfg = copy.deepcopy(cfg["env"]["config"])
    obs_cfg = copy.deepcopy(env_cfg["observation"])
    act_cfg = copy.deepcopy(env_cfg["action"])
    env_cfg["controlled_vehicles"] = n_agents
    env_cfg["observation"] = {"type": "MultiAgentObservation", "observation_config": obs_cfg}
    env_cfg["action"] = {"type": "MultiAgentAction", "action_config": act_cfg}
    env_cfg["duration"] = cfg["race"].get("duration", env_cfg["duration"])

    if render:
        _ensure_render_backend()

    env = gym.make(
        cfg["env"]["id"],
        render_mode="rgb_array" if render else None,
        config=env_cfg,
    )
    return env


def race(model_a, model_b, cfg, seeds=None):
    """Run N races on fixed seeds; report per-agent progress + safety.

    A = baseline (no shield), B = NeSy (shield on). The start slots swap on
    alternate seeds. Winner per race by ``cfg['race']['winner_metric']``
    ("distance" progress or "overtakes").
    """
    rc = cfg["race"]
    seeds = list(seeds) if seeds is not None else list(rc["seeds"])
    metric = rc["winner_metric"]

    env = make_race_env(cfg, n_agents=2)
    rows = []
    try:
        for i, seed in enumerate(seeds):
            r = _run_race(model_a, model_b, env, int(seed), cfg, swap=bool(i % 2))
            print(f"[race] seed {seed} (swap={r['swapped_start']}): "
                  f"A {metric}={r['A'][metric]:.1f} ot={r['A']['overtakes']} crash={r['A']['crashed']} | "
                  f"B {metric}={r['B'][metric]:.1f} ot={r['B']['overtakes']} crash={r['B']['crashed']}",
                  flush=True)
            rows.append(r)
    finally:
        env.close()

    wins = {"A": 0, "B": 0, "tie": 0}
    for r in rows:
        a, b = r["A"][metric], r["B"][metric]
        r["winner"] = "B" if b > a else ("A" if a > b else "tie")
        wins[r["winner"]] += 1
    return {"races": rows, "wins": wins, "metric": metric}


def _run_race(model_a, model_b, env, seed, cfg, swap=False):
    from nesy.roadmap import predicates, safety_shield, rule_violations

    slot_b = 0 if swap else 1              # the shielded NeSy agent's start slot
    models = {slot_b: model_b, 1 - slot_b: model_a}

    obs, info = env.reset(seed=seed)
    u = env.unwrapped
    start_x = [float(u.controlled_vehicles[i].position[0]) for i in (0, 1)]
    ahead = [_ahead_ids(env, i) for i in (0, 1)]
    overtakes = [0, 0]
    viol = [{}, {}]
    fsm_b = cfg["fsm"]["initial_state"]
    interventions = 0
    steps = 0
    done = False

    while not done and steps < cfg["race"]["horizon"]:
        acts = []
        for i in (0, 1):
            a, _ = models[i].predict(obs[i], deterministic=True)
            a = int(a)
            if i == slot_b:                # agent B runs the NeSy shield
                preds = predicates(_scene_for(env, i, cfg), cfg)
                shielded, fsm_b = safety_shield(a, preds, fsm_b, cfg)
                if int(shielded) != a:
                    interventions += 1
                a = int(shielded)
            acts.append(a)

        obs, reward, terminated, truncated, info = env.step(tuple(acts))
        steps += 1
        for i in (0, 1):                   # post-step: overtakes + independent monitor
            passed, ahead[i] = _count_passes(env, i, ahead[i])
            overtakes[i] += passed
            for k, v in rule_violations(predicates(_scene_for(env, i, cfg), cfg), cfg).items():
                viol[i][k] = viol[i].get(k, 0) + int(v)
        done = _is_done(terminated) or _is_done(truncated)

    def outcome(i):
        veh = u.controlled_vehicles[i]
        return {
            "distance": float(veh.position[0] - start_x[i]),
            "overtakes": int(overtakes[i]),
            "crashed": bool(getattr(veh, "crashed", False)),
            "violation_rate": {k: v / steps for k, v in viol[i].items()} if steps else viol[i],
        }

    a_out, b_out = outcome(1 - slot_b), outcome(slot_b)
    b_out["intervention_rate"] = interventions / steps if steps else 0.0
    return {"seed": seed, "steps": steps, "swapped_start": swap, "A": a_out, "B": b_out}


def record_race_video(model_a, model_b, cfg, path, seed=None):
    """Save an MP4 of one shared-scene race (both agents racing). Returns ``path``.

    Both controlled cars are in the same frame; A (start slot 0) is the baseline,
    B (slot 1) runs the shield.
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


def _ahead_ids(env, idx):
    """Identities of the vehicles currently ahead of controlled vehicle ``idx``."""
    u = env.unwrapped
    ego = u.controlled_vehicles[idx]
    return {id(v) for v in u.road.vehicles if v is not ego and v.position[0] > ego.position[0]}


def _count_passes(env, idx, ahead_ids):
    """How many tracked vehicles agent ``idx`` passed this step (and the new ahead-set)."""
    u = env.unwrapped
    ego = u.controlled_vehicles[idx]
    ex = ego.position[0]
    passed = 0
    still = set()
    for v in u.road.vehicles:
        if v is ego:
            continue
        vid = id(v)
        if vid in ahead_ids:
            if v.position[0] < ex:
                passed += 1
            else:
                still.add(vid)
        elif v.position[0] > ex:
            still.add(vid)
    return passed, still


def _is_done(flag):
    """Normalise a scalar/tuple terminated|truncated flag to a single bool."""
    if isinstance(flag, (tuple, list, np.ndarray)):
        return bool(np.any(flag))
    return bool(flag)
