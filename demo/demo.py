"""Standalone demo: record an MP4 of a saved policy driving highway-v0.

Runs OUTSIDE any notebook kernel and renders offscreen, so pygame can never crash
a kernel. Works for Part-1 (PPO/DQN) and Part-2 (NeSy, ``--shield``) checkpoints.

Usage:
    python demo/demo.py --model /content/drive/MyDrive/nesy-highway-driving/checkpoints/part1_best_ppo.zip
    python demo/demo.py --model .../part2_nesy.zip --shield --out drive/videos/part2_nesy.mp4

The clip plays episodes back to back until it is at least ``--seconds`` long.
"""

import os
import sys
import argparse

# Offscreen + no audio must be set BEFORE pygame/highway-env is imported. A headless
# machine has no sound device, so pygame.init() segfaults opening ALSA -> disable it.
# For video we start a real xvfb display on Linux (real frames); we only fall back to
# the dummy video driver if xvfb is unavailable, because highway-env draws BLANK frames
# when it detects SDL_VIDEODRIVER=dummy.
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
if sys.platform == "linux" and not os.environ.get("DISPLAY"):
    try:
        from pyvirtualdisplay import Display

        _XVFB = Display(visible=0, size=(1400, 900))
        _XVFB.start()
    except Exception as e:  # pragma: no cover - platform dependent
        print("xvfb unavailable, falling back to dummy video (frames may be blank):", e)
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# Make the repo importable regardless of where the script is launched from.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

from utils import load_config, save_mp4  # noqa: E402
from envs.highway_factory import create_environment, read_scene  # noqa: E402
from agents.baselines import load_model  # noqa: E402
from nesy.roadmap import predicates, safety_shield  # noqa: E402


def record(model, cfg, out_path, apply_shield=False, min_seconds=30):
    """Play episodes until the clip is >= ``min_seconds``, save an MP4, return path."""
    fps = cfg["eval"].get("video_fps", 10)
    target_frames = int(min_seconds * fps)
    max_episodes = int(cfg["eval"].get("video_max_episodes", 60))
    seed0 = int(cfg["eval_seeds"][0])

    env = create_environment(cfg, render=True)
    frames = []
    try:
        ep = 0
        while len(frames) < target_frames and ep < max_episodes:
            obs, _ = env.reset(seed=seed0 + ep)
            fsm_state = cfg["fsm"]["initial_state"]
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=cfg["eval"]["deterministic"])
                if apply_shield:
                    action, fsm_state = safety_shield(action, predicates(read_scene(env), cfg), fsm_state, cfg)
                obs, _, terminated, truncated, _ = env.step(action)
                frame = env.render()
                if frame is not None:
                    frames.append(np.asarray(frame))
                done = terminated or truncated
            ep += 1
    finally:
        env.close()

    return save_mp4(frames, out_path, fps=fps)


def main():
    ap = argparse.ArgumentParser(description="Record a driving video from a saved model.")
    ap.add_argument("--model", required=True, help="path to the .zip checkpoint (from Drive)")
    ap.add_argument("--algo", default="ppo", choices=["ppo", "dqn"], help="algorithm (default: ppo)")
    ap.add_argument("--shield", action="store_true", help="apply the NeSy safety shield (Part 2)")
    ap.add_argument("--out", default=None, help="output .mp4 path (default: alongside the model)")
    ap.add_argument("--config", default="configs/highway.yaml", help="config YAML")
    ap.add_argument("--seconds", type=float, default=None, help="min video length (default: config eval.video_seconds)")
    args = ap.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(_REPO, args.config)
    cfg = load_config(cfg_path)
    min_seconds = args.seconds if args.seconds is not None else cfg["eval"].get("video_seconds", 30)
    out_path = args.out or (os.path.splitext(args.model)[0] + ("_shield.mp4" if args.shield else ".mp4"))

    print(f"loading {args.algo} model: {args.model}")
    model = load_model(args.model, args.algo)
    print(f"recording >= {min_seconds:.0f}s (shield={args.shield}) -> {out_path}")
    path = record(model, cfg, out_path, apply_shield=args.shield, min_seconds=min_seconds)

    size = os.path.getsize(path) if path and os.path.exists(path) else 0
    if size > 0:
        print(f"[demo] OK    {path}  ({size // 1024} KB)")
    else:
        print(f"[demo] FAIL  {out_path}  (no file written or empty)")
        sys.exit(1)


if __name__ == "__main__":
    main()
