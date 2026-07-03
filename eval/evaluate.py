"""Evaluation harness (metrics only — no rendering).

  * ``evaluate``            — metrics over held-out seeds, with an optional safety
    shield and per-rule violation counting (independent monitor).
  * ``select_nesy_method``  — rank the Part-2 NeSy configs (shield vs reward).

Runs entirely with ``render=False`` (no pygame), so it is fast and never touches a
kernel. Videos are produced separately by the standalone ``demo/demo.py`` script.

Per-rule violations are counted by ``nesy.roadmap.rule_violations`` on the SI
scene, NOT by the shield/reward the agent sees, so "fewer violations" is not
circular. The scene adapter is pluggable via ``scene_fn`` so the same harness
serves highway-env and MetaDrive.

No top-level execution — the notebooks call these.
"""

import numpy as np

from envs.highway_factory import create_environment, read_scene

# Rules audited by the independent monitor (order used everywhere we tabulate).
_RULES = ("RG1", "RG2", "RG3", "RG4", "RI1", "RI2")


# =============================================================================
# Single-model evaluation (Parts 1-3)
# =============================================================================
def evaluate(model, cfg, seeds=None, apply_shield=False, count_violations=False,
             env_fn=None, scene_fn=None):
    """Evaluate ``model`` over held-out seeds and aggregate metrics (no rendering).

    Args:
        model: trained SB3 model.
        cfg: full config.
        seeds: eval seeds; defaults to ``cfg['eval_seeds']``.
        apply_shield: wrap each action in the discrete NeSy safety shield (Part 2).
        count_violations: report per-rule violation rates (independent monitor).
        env_fn: builder ``(cfg, render) -> env``; defaults to highway.
        scene_fn: ``env -> scene dict``; defaults to highway ``read_scene``.

    Returns:
        dict with ``summary`` (headline metrics + overtaking diagnostics, and
        ``rule_violation_rate`` when requested), ``episodes`` and ``seeds``.
    """
    seeds = list(seeds) if seeds is not None else list(cfg["eval_seeds"])
    env_fn = env_fn or (lambda c, render: create_environment(c, render=render))
    scene_fn = scene_fn or read_scene
    ec = cfg["eval"]

    env = env_fn(cfg, False)
    rows = []
    try:
        for i, seed in enumerate(seeds):
            for ep in range(ec["episodes_per_seed"]):
                rows.append(_run_episode(
                    model, env, int(seed) * 100 + ep, cfg,
                    deterministic=ec["deterministic"],
                    apply_shield=apply_shield,
                    count_violations=count_violations,
                    scene_fn=scene_fn,
                ))
            _print_eval_progress(rows, i + 1, len(seeds))
    finally:
        env.close()

    return {"summary": _summarise(rows, count_violations), "episodes": rows, "seeds": seeds}


def _print_eval_progress(rows, seeds_done, seeds_total):
    """Running one-liner so a long evaluation shows the overtaking/crash picture live."""
    n = len(rows)
    ot = sum(r["overtakes"] for r in rows)
    steps = sum(r["length"] for r in rows)
    crashes = sum(1 for r in rows if r["crashed"])
    print(f"[eval] seed {seeds_done:>2}/{seeds_total} | {n:>3} episodes | "
          f"overtakes/ep {ot / n:4.2f} ({100.0 * ot / max(1, steps):4.1f}/100 steps) | "
          f"crash {crashes / n:4.0%}", flush=True)


def _run_episode(model, env, seed, cfg, deterministic, apply_shield,
                 count_violations, scene_fn):
    from nesy.roadmap import predicates, safety_shield, rule_violations

    obs, info = env.reset(seed=seed)
    fsm_state = cfg["fsm"]["initial_state"]
    done = False
    ret = native_ret = 0.0
    steps = offroad_steps = 0
    viol = {k: 0 for k in _RULES}
    speed_sum = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)

        # The shield decides from the CURRENT (pre-action) state.
        if apply_shield:
            preds = predicates(scene_fn(env), cfg)
            action, fsm_state = safety_shield(action, preds, fsm_state, cfg)

        obs, reward, terminated, truncated, info = env.step(action)

        # The independent monitor audits the state the action LEADS TO — including the
        # terminal/crash state, where abrupt braking (RG2) and near-stall (RI1)
        # concentrate. Counting the pre-action state instead would miss them. The same
        # post-step scene also gives the ego speed for the mean-velocity metric.
        if count_violations:
            sc = scene_fn(env)
            for k, v in rule_violations(predicates(sc, cfg), cfg).items():
                viol[k] += int(v)
            speed_sum += float(sc["ego"].get("v", 0.0))
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
        row["mean_speed"] = speed_sum / steps if steps else 0.0
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

    # Overtaking diagnostics — episodes are cut short by crashes, so the raw
    # overtake count understates the policy. Normalising by episode length (rate
    # per 100 steps) and reporting how *often* it overtakes at all gives a fairer
    # read on whether the car actually passes traffic.
    overtakes = [r["overtakes"] for r in rows]
    per100 = [100.0 * o / max(1, r["length"]) for o, r in zip(overtakes, rows)]
    summary["overtake_rate_per_100steps"] = {
        "mean": float(np.mean(per100)) if n else 0.0,
        "std": float(np.std(per100)) if n else 0.0,
    }
    summary["episodes_with_overtake"] = (
        sum(1 for o in overtakes if o > 0) / n if n else 0.0
    )
    summary["max_overtakes"] = int(max(overtakes)) if overtakes else 0

    if count_violations:
        total_steps = sum(r["viol_steps"] for r in rows) or 1
        summary["rule_violation_rate"] = {
            k: sum(r["violations"][k] for r in rows) / total_steps for k in _RULES
        }  # fraction of steps violating each rule
        summary["mean_speed"] = float(np.mean([r["mean_speed"] for r in rows])) if rows else 0.0
    return summary


# =============================================================================
# Part 3 — run the Part-2 DISCRETE policy on MetaDrive via the Lab-1 cmd_vel bridge
# =============================================================================
def evaluate_nesy_md(part2_model, cfg, seeds=None, shield=True):
    """Evaluate the Part-2 discrete policy on MetaDrive, aggregated (no rendering).

    The bridge (one step): reconstruct highway Kinematics obs from MetaDrive
    (``read_kin_obs_md``) -> the Part-2 model picks a manoeuvre -> [optional discrete
    FSM shield] -> Lab-1 ``manoeuvre_to_cmd_vel`` -> ``(v, ω)`` -> [optional CBF/VO
    continuous filter] -> step. Same summary shape as ``evaluate`` (rule_violation_rate
    + mean_speed). This is how Part 2's model + all the labs run in MetaDrive.
    """
    from envs.metadrive_factory import make_env_md

    seeds = list(seeds) if seeds is not None else list(cfg["eval_seeds"])
    ec = cfg["eval"]
    env = make_env_md(cfg, render=False)
    rows = []
    try:
        for seed in seeds:
            for ep in range(ec["episodes_per_seed"]):
                rows.append(_run_nesy_md_episode(part2_model, env, int(seed) * 100 + ep, cfg, shield))
    finally:
        env.close()
    return {"summary": _summarise(rows, True), "episodes": rows, "seeds": seeds}


def _run_nesy_md_episode(model, env, seed, cfg, shield):
    from envs.metadrive_factory import read_scene_md, nesy_md_action, count_passes_md
    from nesy.roadmap import predicates, rule_violations

    obs, info = env.reset(seed=seed)
    fsm = cfg["fsm"]["initial_state"]
    _, ahead = count_passes_md(env, set())     # prime the overtake tracker
    done = False
    ret = 0.0
    steps = offroad_steps = overtakes = 0
    viol = {k: 0 for k in _RULES}
    speed_sum = 0.0

    while not done:
        action, fsm = nesy_md_action(model, env, cfg, fsm, shield)   # Lab-1 bridge step
        obs, reward, terminated, truncated, info = env.step(action)
        passed, ahead = count_passes_md(env, ahead)
        overtakes += passed
        sc = read_scene_md(env)
        for k, val in rule_violations(predicates(sc, cfg), cfg).items():
            viol[k] += int(val)
        speed_sum += float(sc["ego"].get("v", 0.0))
        ret += float(reward)
        steps += 1
        if info.get("out_of_road", info.get("is_offroad", False)):
            offroad_steps += 1
        done = terminated or truncated

    crashed = bool(info.get("crash", info.get("crashed", info.get("crash_vehicle", False))))
    return {
        "seed": seed, "crashed": crashed,
        "on_road_pct": 100.0 * (1.0 - offroad_steps / steps) if steps else 0.0,
        "overtakes": int(overtakes), "return": ret, "native_return": ret, "length": steps,
        "violations": viol, "viol_steps": steps,
        "mean_speed": speed_sum / steps if steps else 0.0,
    }


# =============================================================================
# Part 2 (XAI) — rank the NeSy methods: shield vs reward-shaping vs both
# =============================================================================
def total_violation_rate(metrics):
    """Sum of the independent monitor's per-rule violation rates (lower better)."""
    rv = metrics["summary"].get("rule_violation_rate", {})
    return float(sum(rv.values()))


def select_nesy_method(metrics_by_name, cfg, baseline_key=None):
    """Pick the best logic-integration method, the XAI headline of Part 2.

    Mirrors the lecture's *shielding vs reward-shaping* trade-off (slides 92-96):
    shielding guarantees safety by pruning actions but can cost performance;
    reward-shaping keeps performance but only *softly* enforces rules. So we do
    not pick on violations alone — among configs that **keep driving**
    (overtakes within ``select.within_return_pct`` of the baseline **and** crash
    rate no worse than the baseline) we choose the **lowest total violation
    rate**; ties break on higher overtakes.

    Args:
        metrics_by_name: ``{label: evaluate(...) dict}`` (must include the baseline).
        baseline_key: the baseline label; defaults to the first key.

    Returns:
        ``(best_label, table)`` where ``table`` is a list of per-config rows
        (overtakes, crash_rate, total_violation_rate, eligible) for display.
    """
    names = list(metrics_by_name)
    baseline_key = baseline_key or names[0]
    base = metrics_by_name[baseline_key]["summary"]
    band = cfg.get("select", {}).get("within_return_pct", 0.10)
    floor = base["overtakes"]["mean"] - abs(base["overtakes"]["mean"]) * band

    table = []
    for n in names:
        s = metrics_by_name[n]["summary"]
        eligible = (s["overtakes"]["mean"] >= floor) and (s["crash_rate"] <= base["crash_rate"] + 1e-9)
        table.append({
            "config": n,
            "overtakes": round(s["overtakes"]["mean"], 3),
            "crash_rate": round(s["crash_rate"], 3),
            "total_violation_rate": round(total_violation_rate(metrics_by_name[n]), 4),
            "eligible": bool(eligible),
        })

    pool = [r for r in table if r["eligible"]] or table
    best = min(pool, key=lambda r: (r["total_violation_rate"], -r["overtakes"]))
    return best["config"], table
