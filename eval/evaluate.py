import numpy as np

from envs.highway_factory import create_environment, read_scene

RULES = ("RG1", "RG2", "RG3", "RG4", "RI1", "RI2")


def evaluate(model, cfg, seeds=None, apply_shield=False, count_violations=False,
             env_fn=None, scene_fn=None):
    """Run the model over multiple seeds and return summary metrics + per-episode rows."""
    seeds = list(seeds) if seeds is not None else list(cfg["eval_seeds"])
    env_fn = env_fn or (lambda c, render: create_environment(c, render=render))
    scene_fn = scene_fn or read_scene
    ec = cfg["eval"]

    env = env_fn(cfg, False)
    if hasattr(model, "set_eval_env"):
        model.set_eval_env(env)
    rows = []
    try:
        for i, seed in enumerate(seeds):
            for ep in range(ec["episodes_per_seed"]):
                rows.append(run_episode(
                    model, env, int(seed) * 100 + ep, cfg,
                    deterministic=ec["deterministic"],
                    apply_shield=apply_shield,
                    count_violations=count_violations,
                    scene_fn=scene_fn,
                ))
            print_eval_progress(rows, i + 1, len(seeds))
    finally:
        env.close()

    return {"summary": summarise(rows, count_violations), "episodes": rows, "seeds": seeds}


def sanity_rollout(model, cfg, tag="model", max_steps=1000):
    """Quick rollout to verify the model drives sensibly. Prints overtakes and crashes."""
    from collections import Counter

    names = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
    env = create_environment(cfg)
    acts = Counter()
    speeds = []
    lane_changes = overtakes = steps = crashes = eps = 0
    obs, _ = env.reset(seed=cfg["seed"])
    prev = read_scene(env)["ego"]["lane"]
    while steps < max_steps:
        a, _ = model.predict(obs, deterministic=True)
        acts[names[int(a)]] += 1
        obs, _, terminated, truncated, info = env.step(a)
        sc = read_scene(env)
        speeds.append(sc["ego"]["v"])
        lane_changes += int(sc["ego"]["lane"] != prev)
        prev = sc["ego"]["lane"]
        steps += 1
        if terminated or truncated:
            crashes += int(info["crashed"])
            overtakes += info["overtakes"]
            eps += 1
            obs, _ = env.reset()
            prev = read_scene(env)["ego"]["lane"]
    env.close()
    print(f"[sanity {tag}] {steps} steps, {eps} episodes | "
          f"overtakes {overtakes} | crashes {crashes} | actions {dict(acts)}", flush=True)


def print_eval_progress(rows, seeds_done, seeds_total):
    """Print a one-line progress update during evaluation."""
    n = len(rows)
    ot = sum(r["overtakes"] for r in rows)
    crashes = sum(1 for r in rows if r["crashed"])
    print(f"[eval] seed {seeds_done}/{seeds_total} | "
          f"overtakes/ep {ot / n:.2f} | crash {crashes / n:.0%}", flush=True)


def run_episode(model, env, seed, cfg, deterministic, apply_shield,
                count_violations, scene_fn):
    """Run one episode and return a dict with all metrics."""
    from nesy.roadmap import predicates, safety_shield, rule_violations

    obs, info = env.reset(seed=seed)
    fsm_state = cfg["fsm"]["initial_state"]
    done = False
    ret = native_ret = 0.0
    steps = offroad_steps = lane_changes = 0
    viol = {k: 0 for k in RULES}
    speed_sum = 0.0
    x_start = float(scene_fn(env)["ego"]["x"])

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)

        if apply_shield:
            preds = predicates(scene_fn(env), cfg)
            action, fsm_state = safety_shield(action, preds, fsm_state, cfg)

        obs, reward, terminated, truncated, info = env.step(action)
        lane_changes += int(info.get("lane_changed", False))

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

    crashed = bool(info.get("crashed", info.get("crash", info.get("crash_vehicle", False))))
    row = {
        "seed": seed,
        "crashed": crashed,
        "on_road_pct": 100.0 * (1.0 - offroad_steps / steps) if steps else 0.0,
        "overtakes": int(info.get("overtakes", 0)),
        "lane_changes": lane_changes,
        "return": ret,
        "native_return": native_ret,
        "length": steps,
        "distance": float(scene_fn(env)["ego"]["x"]) - x_start,
    }
    if count_violations:
        row["violations"] = viol
        row["viol_steps"] = steps
        row["mean_speed"] = speed_sum / steps if steps else 0.0
    return row


def summarise(rows, count_violations):
    """Aggregate per-episode rows into mean/std summary statistics."""
    def ms(key):
        vals = np.array([r[key] for r in rows], dtype=float)
        return {"mean": float(vals.mean()), "std": float(vals.std())}

    n = len(rows)
    summary = {
        "n_episodes": n,
        "crash_rate": sum(1 for r in rows if r["crashed"]) / n if n else 0.0,
        "on_road_pct": ms("on_road_pct"),
        "overtakes": ms("overtakes"),
        "lane_changes": ms("lane_changes"),
        "return": ms("return"),
        "native_return": ms("native_return"),
        "length": ms("length"),
        "distance": ms("distance"),
    }

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
            k: sum(r["violations"][k] for r in rows) / total_steps for k in RULES
        }
        summary["mean_speed"] = float(np.mean([r["mean_speed"] for r in rows])) if rows else 0.0
    return summary


# ---- Part 3: MetaDrive evaluation ------------------------------------------

def evaluate_nesy_md(part2_model, cfg, seeds=None, shield=True, video_out=None):
    """Evaluate a discrete model on MetaDrive via the Lab-1 bridge.

    With ``video_out``, every episode is also rendered and the BEST episode
    (most overtakes, then lane changes, then length) is saved as an MP4 —
    the video is the eval episode itself, so they can never disagree."""
    from envs.metadrive_factory import make_env_md

    seeds = list(seeds) if seeds is not None else list(cfg["eval_seeds"])
    # MetaDrive episodes run to the horizon (~1000 steps), so use a smaller
    # per-seed count than the fast highway eval unless configured otherwise.
    eps_per_seed = int(cfg["metadrive"].get("episodes_per_seed",
                                            cfg["eval"]["episodes_per_seed"]))
    env = make_env_md(cfg, render=bool(video_out))
    grab = None
    if video_out:
        from demo.demo_md import make_topdown_grab
        grab = make_topdown_grab(env, cfg)
    rows = []
    best = None
    try:
        for i, seed in enumerate(seeds):
            for ep in range(eps_per_seed):
                # seed*100+ep collides mod num_scenarios (100 % 50 == 0): every eval
                # seed would replay the same two scenarios. Space episodes instead.
                ep_seed = int(cfg["seed"]) + i * eps_per_seed + ep
                row = run_nesy_md_episode(part2_model, env, ep_seed, cfg, shield, grab=grab)
                frames = row.pop("_frames", None)
                rows.append(row)
                if frames is not None:
                    score = (row["overtakes"], row["lane_changes"], row["length"])
                    if best is None or score > best[0]:
                        best = (score, dict(row), frames)
            print_eval_progress(rows, i + 1, len(seeds))
    finally:
        env.close()

    if video_out and best is not None:
        from utils import save_mp4
        save_mp4(best[2], video_out, fps=cfg["metadrive"].get("video_fps", 20))
        r = best[1]
        print(f"[eval] video: best episode saved -> {video_out} | "
              f"{r['overtakes']} overtakes, {r['lane_changes']} lane changes, "
              f"{r['length']} steps, crashed={r['crashed']} (seed {r['seed']})", flush=True)
    return {"summary": summarise(rows, True), "episodes": rows, "seeds": seeds}


def run_nesy_md_episode(model, env, seed, cfg, shield, grab=None):
    """Run one MetaDrive episode through the bridge (manoeuvre -> cmd_vel -> CBF/VO).

    With ``grab``, overlaid video frames are collected into row["_frames"]."""
    import random
    from envs.metadrive_factory import read_scene_md, nesy_md_action, count_passes_md
    from nesy.roadmap import predicates, rule_violations

    # Pin the global RNGs so the episode is identical regardless of what ran
    # before in this process — required for exact video replay of eval episodes.
    random.seed(int(seed))
    np.random.seed(int(seed) % 2**31)
    obs, info = env.reset(seed=seed)
    fsm = cfg["fsm"]["initial_state"]
    _, ahead = count_passes_md(env, set())
    x_start = float(read_scene_md(env)["ego"]["x"])
    prev_lane = read_scene_md(env)["ego"].get("lane")
    done = False
    ret = 0.0
    steps = offroad_steps = overtakes = lane_changes = 0
    viol = {k: 0 for k in RULES}
    speed_sum = 0.0
    bridge = {}

    frames = [] if grab is not None else None

    while not done:
        action, fsm = nesy_md_action(model, env, cfg, fsm, shield, bridge=bridge)
        obs, reward, terminated, truncated, info = env.step(action)
        passed, ahead = count_passes_md(env, ahead)
        overtakes += passed
        sc = read_scene_md(env)
        lane_changes += int(sc["ego"].get("lane") != prev_lane)
        prev_lane = sc["ego"].get("lane")
        for k, val in rule_violations(predicates(sc, cfg), cfg).items():
            viol[k] += int(val)
        speed_sum += float(sc["ego"].get("v", 0.0))
        ret += float(reward)
        steps += 1
        if info.get("out_of_road", info.get("is_offroad", False)):
            offroad_steps += 1
        done = terminated or truncated
        if frames is not None:
            from demo.demo_md import draw_telemetry
            frame = grab(env, obs)
            if frame is not None:
                frame = draw_telemetry(
                    frame, sc["ego"].get("v", 0.0),
                    float(action[1]) * cfg["metadrive"]["omega_max"],
                    fsm_state=fsm, overtakes=overtakes, lane_changes=lane_changes,
                    crashed=done and bool(info.get("crash", False)))
                frames.append(np.asarray(frame))

    crashed = bool(info.get("crash", info.get("crashed", info.get("crash_vehicle", False))))
    row = {
        "seed": seed, "crashed": crashed,
        "on_road_pct": 100.0 * (1.0 - offroad_steps / steps) if steps else 0.0,
        "overtakes": int(overtakes), "lane_changes": lane_changes,
        "return": ret, "native_return": ret, "length": steps,
        "distance": float(read_scene_md(env)["ego"]["x"]) - x_start,
        "violations": viol, "viol_steps": steps,
        "mean_speed": speed_sum / steps if steps else 0.0,
    }
    if frames is not None:
        row["_frames"] = frames
    return row


# ---- Part 2: rank NeSy methods ---------------------------------------------

def total_violation_rate(metrics):
    """Sum of all per-rule violation rates."""
    rv = metrics["summary"].get("rule_violation_rate", {})
    return float(sum(rv.values()))


def select_nesy_method(metrics_by_name, cfg, baseline_key=None):
    """Pick the NeSy config with fewest total violations that is still a safe
    overtaker: crashes no more than the baseline and keeps at least
    ``select.min_overtake_frac`` of the baseline's overtakes.
    Returns (best_name, comparison_table)."""
    names = list(metrics_by_name)
    baseline_key = baseline_key or names[0]
    base = metrics_by_name[baseline_key]["summary"]
    frac = cfg.get("select", {}).get("min_overtake_frac", 0.5)
    floor = base["overtakes"]["mean"] * frac

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
