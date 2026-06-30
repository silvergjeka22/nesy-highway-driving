"""Evaluation harness for all four parts.

  * ``evaluate``            — metrics over held-out seeds, with optional safety
    shield, per-rule violation counting (independent MTL monitor), and an
    optional ≥``video_seconds`` clip of the same policy driving.
  * ``record_video``        — MP4 of one model driving, in-process (Parts 1-3).
  * ``record_video_safe``   — same, but rendered in a SUBPROCESS so a headless-
    Colab pygame/SDL segfault can't crash the notebook kernel. ``evaluate`` uses
    this when given the model's checkpoint path (the Colab-safe path).
  * ``record_random_video`` — short random-policy clip (the Part-1 env study).
  * ``show_video``          — embed an MP4 inline in the notebook.
  * ``select_nesy_method``  — rank the Part-2 NeSy configs (shield vs reward).

Per-rule violations are counted by ``nesy.roadmap.rule_violations`` on the SI
scene, NOT by the shield/reward the agent sees, so "fewer violations" is not
circular. The scene adapter is pluggable via ``scene_fn`` so the same harness
serves highway-env and MetaDrive.

No top-level execution — the notebooks call these.
"""

import os

import numpy as np

from envs.highway_factory import make_env, read_scene

# Rules audited by the independent monitor (order used everywhere we tabulate).
_RULES = ("RG1", "RG2", "RG3", "RG4", "RI1", "RI2")


# =============================================================================
# Single-model evaluation (Parts 1-3)
# =============================================================================
def evaluate(model, cfg, seeds=None, apply_shield=False, count_violations=False,
             env_fn=None, scene_fn=None, video_path=None, video_seconds=None,
             video_model_path=None, video_algo=None, cfg_path="configs/highway.yaml"):
    """Evaluate ``model`` over held-out seeds and aggregate metrics.

    Args:
        model: trained SB3 model.
        cfg: full config.
        seeds: eval seeds; defaults to ``cfg['eval_seeds']``.
        apply_shield: wrap each action in the NeSy safety shield (Part 2/3).
        count_violations: report per-rule violation rates (independent monitor).
        env_fn: builder ``(cfg, render) -> env``; defaults to highway ``make_env``.
        scene_fn: ``env -> scene dict``; defaults to highway ``read_scene``.
        video_path: if set, also save a clip of this policy driving (same shield
            setting) to this path, at least ``video_seconds`` long, and return it
            under ``video``. Lets one call both score *and* show a model.
        video_seconds: minimum video length; defaults to ``cfg['eval']['video_seconds']``.
        video_model_path, video_algo: if given (with ``video_path``), the clip is
            rendered in a SUBPROCESS from this checkpoint — kernel-safe on headless
            Colab (the in-process pygame renderer can segfault the notebook). This
            is the path the notebooks use. Without them, falls back to in-process
            ``record_video`` (fine locally).
        cfg_path: config path the render subprocess reloads (default repo YAML).

    Returns:
        dict with ``summary`` (headline metrics + overtaking diagnostics, and
        ``rule_violation_rate`` when requested), ``episodes``, ``seeds`` and,
        when ``video_path`` is set, ``video``.
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

    result = {"summary": _summarise(rows, count_violations), "episodes": rows, "seeds": seeds}

    if video_path is not None:
        secs = video_seconds if video_seconds is not None else ec.get("video_seconds", 30)
        if video_model_path is not None:
            # Kernel-safe: render from the checkpoint in a child process so a
            # headless-Colab pygame/SDL segfault can't kill the notebook kernel.
            result["video"] = record_video_safe(
                video_model_path, video_algo, cfg_path, video_path,
                apply_shield=apply_shield, min_seconds=secs,
            )
        else:
            result["video"] = record_video(
                model, cfg, video_path, apply_shield=apply_shield,
                env_fn=env_fn, scene_fn=scene_fn, min_seconds=secs,
            )
    return result


def _run_episode(model, env, seed, cfg, deterministic, apply_shield,
                 count_violations, scene_fn):
    from nesy.roadmap import predicates, safety_shield, rule_violations

    obs, info = env.reset(seed=seed)
    fsm_state = cfg["fsm"]["initial_state"]
    done = False
    ret = native_ret = 0.0
    steps = offroad_steps = 0
    viol = {k: 0 for k in _RULES}

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
    return summary


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


# =============================================================================
# Kernel-safe video: render in a SUBPROCESS so a pygame/SDL segfault on headless
# Colab cannot crash the notebook kernel ("Canceled future…"). Each call gets a
# fresh process + fresh display, which also dodges the second-pygame-init crash.
# =============================================================================
def record_video_safe(model_path, algo, cfg_path, out_path,
                      apply_shield=False, min_seconds=30, timeout=900):
    """Render ``model_path``'s ~``min_seconds`` video in a child process.

    Returns the path on success, else None. Rendering in a subprocess means a
    pygame/SDL segfault on headless Colab cannot crash the notebook kernel
    ("Canceled future…"). The model must already be saved to ``model_path`` and
    the config readable at ``cfg_path`` (both true in the notebooks).
    """
    import sys
    import subprocess

    # Anchor the child to the repo root so it never depends on the caller's cwd.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_abs = cfg_path if os.path.isabs(cfg_path) else os.path.join(repo_root, cfg_path)

    code = (
        f"import sys; sys.path.insert(0, {repo_root!r})\n"
        "from utils import load_config\n"
        "from agents.baselines import load_model\n"
        "from eval.evaluate import record_video\n"
        f"cfg = load_config({cfg_abs!r})\n"
        f"m = load_model({model_path!r}, {algo!r})\n"
        f"p = record_video(m, cfg, {out_path!r}, apply_shield={bool(apply_shield)}, "
        f"min_seconds={min_seconds!r})\n"
        "print('VIDEO_OK', p)\n"
    )
    try:
        r = subprocess.run([sys.executable, "-c", code], cwd=repo_root, timeout=timeout,
                           capture_output=True, text=True)
    except Exception as e:
        print("record_video_safe: could not start the render process:", repr(e))
        return None
    if r.returncode == 0:
        return out_path
    # Surface why the child failed (instead of a silent None).
    print(f"record_video_safe: render failed (returncode={r.returncode}). Last output:")
    tail = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-20:]
    for line in tail:
        print("   ", line)
    return None


# =============================================================================
# In-process video (used by the subprocess above, and locally)
# =============================================================================
def record_video(model, cfg, path, n_episodes=None, seed=None,
                 apply_shield=False, env_fn=None, scene_fn=None, min_seconds=None):
    """Record an MP4 of the policy driving, saved to ``path``. Returns ``path``.

    Captures ``env.render()`` frames manually and writes them with imageio — this
    works uniformly for highway-env and MetaDrive.

    ``min_seconds``: if set, keep playing episodes until the clip is at least this
    long (≈ ``min_seconds × fps`` frames), capped at ``video_max_episodes`` — so a
    short (crashed) episode doesn't give a 2-second video. Otherwise plays
    ``n_episodes``.
    """
    from utils import save_mp4
    from nesy.roadmap import predicates, safety_shield

    n_episodes = n_episodes if n_episodes is not None else cfg["eval"].get("video_episodes", 1)
    seed = seed if seed is not None else cfg["eval_seeds"][0]
    fps = cfg["eval"].get("video_fps", 10)
    env_fn = env_fn or (lambda c, render: make_env(c, render=render))
    scene_fn = scene_fn or read_scene
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    target_frames = int(min_seconds * fps) if min_seconds else None
    max_episodes = int(cfg["eval"].get("video_max_episodes", 40))

    env = env_fn(cfg, True)
    frames = []
    try:
        ep = 0
        while True:
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
            ep += 1
            if target_frames is not None:
                if len(frames) >= target_frames or ep >= max_episodes:
                    break
            elif ep >= n_episodes:
                break
    finally:
        env.close()

    return save_mp4(frames, path, fps=fps)


def record_random_video(cfg, path, n_steps=None, seed=None, env_fn=None):
    """Record a short random-policy rollout (Part-1 environment study).

    Lets the reader see the task before any learning. Returns ``path``.
    """
    from utils import save_mp4

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

    return save_mp4(frames, path, fps=cfg["eval"].get("video_fps", 10))


def show_video(path, width=720):
    """Return an IPython HTML5 ``<video>`` that embeds the MP4 inline (for Colab).

    Base64-embeds the file so it plays in the notebook regardless of the Drive
    path. Use in the last cell: ``show_video(drive_path(cfg,'videos','part1_best.mp4'))``.
    """
    import base64
    from IPython.display import HTML

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return HTML(
        f'<video controls autoplay loop width="{width}">'
        f'<source src="data:video/mp4;base64,{b64}" type="video/mp4">'
        "</video>"
    )

# Part 4 (head-to-head race) lives in eval/race.py: race(), record_race_video().
