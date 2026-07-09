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


def record(model, cfg, out_path, apply_shield=False, min_seconds=30, no_crash=False):
    """Play episodes until the clip is >= min_seconds and save an MP4."""
    fps = int(cfg["env"]["config"].get("simulation_frequency", 15))
    target_frames = int(min_seconds * fps)
    max_episodes = int(cfg["eval"].get("video_max_episodes", 60))
    seed0 = int(cfg["eval_seeds"][0])

    env = create_environment(cfg, render=True)
    hook = _FrameHook(env)
    env.unwrapped._record_video_wrapper = hook
    frames = []
    total_overtakes = total_steps = 0
    try:
        ep = 0
        while len(frames) < target_frames and ep < max_episodes:
            obs, info = env.reset(seed=seed0 + ep)
            if hasattr(model, "reset_states"):
                model.reset_states()
            fsm_state = cfg["fsm"]["initial_state"]
            done = False
            steps = 0
            ep_frames = []
            hook.frames = ep_frames
            hook._capture_frame()
            while not done:
                action, _ = model.predict(obs, deterministic=cfg["eval"]["deterministic"])
                if apply_shield:
                    action, fsm_state = safety_shield(action, predicates(read_scene(env), cfg), fsm_state, cfg)
                obs, _, terminated, truncated, info = env.step(action)
                hook._capture_frame()
                steps += 1
                done = terminated or truncated
            crashed = bool(info.get("crashed"))
            ot = int(info.get("overtakes", 0))
            if no_crash and crashed:
                print(f"[demo] episode {ep + 1}: crashed — discarded (--no-crash)", flush=True)
                ep += 1
                continue
            frames.extend(ep_frames)
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
    ap.add_argument("--model", required=True, help="SB3 .zip or bare .pt state_dict")
    ap.add_argument("--algo", default="ppo", choices=["ppo", "dqn", "qrdqn"])
    ap.add_argument("--shield", action="store_true", help="apply NeSy safety shield")
    ap.add_argument("--out", default=None, help="output .mp4 path")
    ap.add_argument("--config", default="configs/highway.yaml")
    ap.add_argument("--seconds", type=float, default=None, help="min video length")
    ap.add_argument("--duration", type=int, default=None, help="episode length in sim-seconds")
    ap.add_argument("--no-crash", dest="no_crash", action="store_true",
                    help="discard crashed episodes")
    args = ap.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(_REPO, args.config)
    cfg = load_config(cfg_path)
    if args.duration is not None:
        cfg["env"]["config"]["duration"] = args.duration
    min_seconds = args.seconds if args.seconds is not None else cfg["eval"].get("video_seconds", 30)
    out_path = args.out or (os.path.splitext(args.model)[0] + ("_shield.mp4" if args.shield else ".mp4"))

    print(f"loading {args.algo} model: {args.model}")
    if args.model.endswith((".pt", ".pth")):
        model = load_policy_weights(args.model, args.algo, cfg)
    else:
        model = load_model(args.model, args.algo)
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
