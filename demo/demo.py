"""Record an MP4 of a saved policy driving highway-v0 (Parts 1-2).

Usage:
    python demo/demo.py --model .../checkpoints/ppo.zip --algo ppo
    python demo/demo.py --model .../part2_nesy.zip --shield
"""

import os
import sys
import argparse

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/xdg-nesy")
try:
    os.makedirs("/tmp/xdg-nesy", mode=0o700, exist_ok=True)
except OSError:
    pass
if sys.platform == "linux" and not os.environ.get("DISPLAY"):
    try:
        from pyvirtualdisplay import Display
        _XVFB = Display(visible=0, size=(1400, 900), color_depth=24)
        _XVFB.start()
    except Exception:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

from utils import load_config, save_mp4  # noqa: E402
from envs.highway_factory import create_environment, read_scene  # noqa: E402
from agents.baselines import load_model, build_ppo, build_dqn, build_qrdqn  # noqa: E402
from nesy.roadmap import predicates, safety_shield  # noqa: E402

_BUILDERS = {"ppo": build_ppo, "dqn": build_dqn, "qrdqn": build_qrdqn}


def load_policy_weights(weights_path, algo, cfg):
    """Rebuild a model from a bare torch state_dict (.pt) for cross-version loading."""
    import torch

    env = create_environment(cfg)
    try:
        model = _BUILDERS[algo](cfg, env, device="cpu")
        state = torch.load(weights_path, map_location="cpu")
        model.policy.load_state_dict(state)
    finally:
        env.close()
    return model


class _FrameHook:
    """Capture substep frames from highway-env's automatic rendering callback."""

    def __init__(self, env):
        self.env = env
        self.frames = None

    def _capture_frame(self):
        frame = self.env.render()
        if frame is not None and self.frames is not None:
            self.frames.append(np.asarray(frame))


def overlay_episode_stats(frames, marks, crashed, fps):
    """Overlay a live overtake counter (and a CRASHED banner at the end) on frames.

    marks: list of (end_frame_idx_exclusive, overtake_count) checkpoints.
    """
    from PIL import Image, ImageDraw, ImageFont

    size = max(14, frames[0].shape[0] // 9)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:
        font = ImageFont.load_default()
    out = []
    mi, ot = 0, 0
    crash_from = len(frames) - 2 * fps  # flash CRASHED over the last ~2 s
    for i, f in enumerate(frames):
        while mi < len(marks) and i >= marks[mi][0]:
            ot = marks[mi][1]
            mi += 1
        img = Image.fromarray(np.asarray(f, dtype=np.uint8))
        d = ImageDraw.Draw(img)
        pad = max(3, size // 4)
        text = f"overtakes: {ot}"
        box = d.textbbox((pad, pad), text, font=font)
        d.rectangle((0, 0, box[2] + pad, box[3] + pad), fill=(0, 0, 0))
        d.text((pad, pad), text, fill=(255, 255, 255), font=font)
        if crashed and i >= crash_from:
            cb = d.textbbox((0, 0), "CRASHED", font=font)
            cw = cb[2] - cb[0]
            x = (img.width - cw) // 2
            d.rectangle((x - pad, 0, x + cw + pad, cb[3] + pad), fill=(160, 0, 0))
            d.text((x, pad), "CRASHED", fill=(255, 255, 255), font=font)
        out.append(np.asarray(img))
    return out


def play_episode(model, env, hook, seed, cfg, apply_shield):
    """Play one seeded episode; return (frames, marks, overtakes, steps, crashed)."""
    obs, info = env.reset(seed=int(seed))
    if hasattr(model, "reset_states"):
        model.reset_states()
    fsm_state = cfg["fsm"]["initial_state"]
    done = False
    steps = 0
    ep_frames = []
    marks = []
    hook.frames = ep_frames
    hook._capture_frame()
    while not done:
        action, _ = model.predict(obs, deterministic=cfg["eval"]["deterministic"])
        if apply_shield:
            action, fsm_state = safety_shield(action, predicates(read_scene(env), cfg), fsm_state, cfg)
        obs, _, terminated, truncated, info = env.step(action)
        hook._capture_frame()
        steps += 1
        marks.append((len(ep_frames), int(info.get("overtakes", 0))))
        done = terminated or truncated
    return ep_frames, marks, int(info.get("overtakes", 0)), steps, bool(info.get("crashed"))


def record_best_episode(model, cfg, out_path, apply_shield=False, best_of=10):
    """Record ONE episode: the best of `best_of` candidates (most overtakes,
    then longest)."""
    fps = int(cfg["env"]["config"].get("simulation_frequency", 15))
    seeds = [int(cfg["eval_seeds"][0]) + i for i in range(best_of)]

    env = create_environment(cfg, render=True)
    if hasattr(model, "set_eval_env"):
        model.set_eval_env(env)
    hook = _FrameHook(env)
    env.unwrapped._record_video_wrapper = hook
    best = None
    try:
        for seed in seeds:
            ep = play_episode(model, env, hook, seed, cfg, apply_shield)
            frames, marks, ot, steps, crashed = ep
            print(f"[demo] candidate seed {seed}: {ot} overtakes in {steps} steps"
                  f"{'  (crashed)' if crashed else ''}", flush=True)
            if best is None or (ot, steps) > (best[2], best[3]):
                best = ep
    finally:
        env.unwrapped._record_video_wrapper = None
        env.close()

    frames, marks, ot, steps, crashed = best
    frames = overlay_episode_stats(frames, marks, crashed, fps)
    print(f"[demo] best episode: {ot} overtakes, {steps} steps, crashed={crashed} "
          f"({len(frames) / fps:.1f}s)", flush=True)
    return save_mp4(frames, out_path, fps=fps), ot, steps


def record_episodes(model, cfg, out_path, apply_shield, episode_seeds, min_seconds=0):
    """Replay the given eval-episode seeds back-to-back (best first), stopping
    once the clip reaches min_seconds. Each episode keeps its own overtake
    counter and CRASHED banner — shows who survives longer."""
    fps = int(cfg["env"]["config"].get("simulation_frequency", 15))
    env = create_environment(cfg, render=True)
    if hasattr(model, "set_eval_env"):
        model.set_eval_env(env)
    hook = _FrameHook(env)
    env.unwrapped._record_video_wrapper = hook
    frames = []
    total_ot = total_steps = 0
    try:
        for i, seed in enumerate(episode_seeds):
            ep_frames, marks, ot, steps, crashed = play_episode(
                model, env, hook, seed, cfg, apply_shield)
            frames.extend(overlay_episode_stats(ep_frames, marks, crashed, fps))
            total_ot += ot
            total_steps += steps
            print(f"[demo] episode {i + 1} (seed {seed}): {ot} overtakes in {steps} steps"
                  f"{'  (crashed)' if crashed else ''}", flush=True)
            if min_seconds and len(frames) >= min_seconds * fps:
                break
    finally:
        env.unwrapped._record_video_wrapper = None
        env.close()

    if min_seconds:  # same clip length for every config, so grids stay in sync
        frames = frames[:int(min_seconds * fps)]
    print(f"[demo] clip: {len(frames)} frames @ {fps} fps = {len(frames) / fps:.1f}s", flush=True)
    return save_mp4(frames, out_path, fps=fps), total_ot, total_steps


def record(model, cfg, out_path, apply_shield=False, min_seconds=30, no_crash=False):
    """Play episodes until the clip is >= min_seconds and save an MP4."""
    fps = int(cfg["env"]["config"].get("simulation_frequency", 15))
    target_frames = int(min_seconds * fps)
    max_episodes = int(cfg["eval"].get("video_max_episodes", 60))
    seed0 = int(cfg["eval_seeds"][0])

    env = create_environment(cfg, render=True)
    if hasattr(model, "set_eval_env"):
        model.set_eval_env(env)
    hook = _FrameHook(env)
    env.unwrapped._record_video_wrapper = hook
    frames = []
    total_overtakes = total_steps = 0
    try:
        ep = 0
        while len(frames) < target_frames and ep < max_episodes:
            ep_frames, marks, ot, steps, crashed = play_episode(
                model, env, hook, seed0 + ep, cfg, apply_shield)
            if no_crash and crashed:
                print(f"[demo] episode {ep + 1}: crashed — discarded (--no-crash)", flush=True)
                ep += 1
                continue
            frames.extend(overlay_episode_stats(ep_frames, marks, crashed, fps))
            total_overtakes += ot
            total_steps += steps
            print(f"[demo] episode {ep + 1}: {ot} overtakes in {steps} steps "
                  f"({len(ep_frames) / fps:.1f}s)"
                  f"{'  (crashed)' if crashed else ''}", flush=True)
            ep += 1
    finally:
        env.unwrapped._record_video_wrapper = None
        env.close()

    print(f"[demo] clip: {len(frames)} frames @ {fps} fps = {len(frames) / fps:.1f}s", flush=True)
    return save_mp4(frames, out_path, fps=fps), total_overtakes, total_steps


def main():
    ap = argparse.ArgumentParser(description="Record a driving video from a saved model.")
    ap.add_argument("--model", default=None, help="SB3 .zip or bare .pt state_dict (not needed for --algo mcts)")
    ap.add_argument("--algo", default="ppo", choices=["ppo", "dqn", "qrdqn", "mcts"])
    ap.add_argument("--shield", action="store_true", help="apply NeSy safety shield")
    ap.add_argument("--out", default=None, help="output .mp4 path")
    ap.add_argument("--config", default="configs/highway.yaml")
    ap.add_argument("--seconds", type=float, default=None, help="min video length")
    ap.add_argument("--duration", type=int, default=None, help="episode length in sim-seconds")
    ap.add_argument("--no-crash", dest="no_crash", action="store_true",
                    help="discard crashed episodes")
    ap.add_argument("--episode-seed", type=str, default=None,
                    help="eval-episode seed(s) to replay exactly, comma-separated; "
                         "with --seconds, plays them until the clip is that long")
    ap.add_argument("--best-of", type=int, default=None,
                    help="record the best of N candidate episodes (most overtakes)")
    args = ap.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(_REPO, args.config)
    cfg = load_config(cfg_path)
    if args.duration is not None:
        cfg["env"]["config"]["duration"] = args.duration
    min_seconds = args.seconds if args.seconds is not None else cfg["eval"].get("video_seconds", 30)
    if args.out is None and args.model is None:
        ap.error("--out is required when --model is not given")
    out_path = args.out or (os.path.splitext(args.model)[0] + ("_shield.mp4" if args.shield else ".mp4"))

    if args.algo == "mcts":
        from labs.lab4_velocity_obstacles import MCTSPolicy
        model = MCTSPolicy(cfg)
        print("using MCTS planner (no checkpoint)")
    elif args.model is None:
        ap.error("--model is required unless --algo mcts")
    elif args.model.endswith((".pt", ".pth")):
        model = load_policy_weights(args.model, args.algo, cfg)
        print(f"loaded {args.algo} weights: {args.model}")
    else:
        model = load_model(args.model, args.algo)
        print(f"loaded {args.algo} model: {args.model}")

    if args.episode_seed is not None:
        seeds = [int(s) for s in args.episode_seed.split(",")]
        floor = args.seconds or 0
        print(f"replaying {len(seeds)} eval episode(s), >= {floor:.0f}s "
              f"(shield={args.shield}) -> {out_path}")
        path, overtakes, steps = record_episodes(
            model, cfg, out_path, apply_shield=args.shield,
            episode_seeds=seeds, min_seconds=floor)
    elif args.best_of is not None:
        print(f"recording best of {args.best_of} episodes (shield={args.shield}) -> {out_path}")
        path, overtakes, steps = record_best_episode(
            model, cfg, out_path, apply_shield=args.shield, best_of=args.best_of)
    else:
        print(f"recording >= {min_seconds:.0f}s (shield={args.shield}) -> {out_path}")
        path, overtakes, steps = record(model, cfg, out_path, apply_shield=args.shield,
                                        min_seconds=min_seconds, no_crash=args.no_crash)

    size = os.path.getsize(path) if path and os.path.exists(path) else 0
    if size > 0:
        print(f"[demo] overtakes: {overtakes} in {steps} steps "
              f"({100.0 * overtakes / max(1, steps):.1f} per 100 steps)")
        print(f"[demo] OK    {path}  ({size // 1024} KB)")
    else:
        print(f"[demo] FAIL  {out_path}  (no frames written)")
        sys.exit(1)


if __name__ == "__main__":
    main()
