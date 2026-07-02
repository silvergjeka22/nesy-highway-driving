"""Standalone demo: record an MP4 of a MetaDrive policy from the top-down view.

Runs OUTSIDE the notebook kernel and renders **offscreen on CPU** with MetaDrive's
pygame top-down renderer (no GPU / EGL / xvfb needed), so it never overloads the
machine or crashes a kernel. Plays episodes back to back until the clip is at least
``--seconds`` long, then encodes an H.264 MP4 and prints ``[demo] OK/FAIL``.

Usage (from the Part-3 notebook this is launched as a background subprocess):
    python demo/demo_md.py --model <RESULTS>/checkpoints/part3_metadrive.zip
    python demo/demo_md.py --model .../part3_metadrive.zip --shield --out .../videos/part3.mp4
"""

import os
import sys
import argparse

# MetaDrive's top-down renderer uses pygame; disable audio so pygame.init() cannot
# segfault on a headless machine. Top-down draws to an offscreen surface (window=False),
# so no virtual display is required.
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

from stable_baselines3 import PPO  # noqa: E402
from utils import load_config, save_mp4  # noqa: E402
from envs.metadrive_factory import make_env_md, read_scene_md, filter_action_md  # noqa: E402


def record(model, cfg, out_path, apply_filter=False, min_seconds=30):
    """Play episodes until the clip is >= ``min_seconds``, save an MP4, return path."""
    md = cfg["metadrive"]
    fps = md.get("video_fps", 20)
    size = tuple(md.get("video_size", [800, 800]))
    target_frames = int(min_seconds * fps)
    seed0 = int(cfg["eval_seeds"][0])

    env = make_env_md(cfg, render=True, seed=seed0)
    base = env.unwrapped   # MetaDrive's top-down render(mode=...) lives on the base env
    frames = []
    try:
        ep = 0
        while len(frames) < target_frames:
            obs, _ = env.reset(seed=seed0 + ep)
            done = False
            while not done and len(frames) < target_frames:
                action, _ = model.predict(obs, deterministic=cfg["eval"]["deterministic"])
                if apply_filter:
                    action = filter_action_md(action, read_scene_md(env), cfg)
                obs, _, terminated, truncated, _ = env.step(action)
                frame = base.render(mode="top_down", window=False,
                                    screen_size=size, film_size=(size[0] * 2, size[1] * 2))
                if frame is not None:
                    frames.append(np.asarray(frame))
                done = terminated or truncated
            ep += 1
    finally:
        env.close()

    return save_mp4(frames, out_path, fps=fps)


def main():
    ap = argparse.ArgumentParser(description="Record a MetaDrive top-down driving video.")
    ap.add_argument("--model", required=True, help="path to the .zip checkpoint (from Drive)")
    ap.add_argument("--shield", action="store_true", help="apply the continuous CBF/VO filter")
    ap.add_argument("--out", default=None, help="output .mp4 path (default: alongside the model)")
    ap.add_argument("--config", default="configs/highway.yaml", help="config YAML")
    ap.add_argument("--seconds", type=float, default=None, help="min length (default: metadrive.video_seconds)")
    args = ap.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(_REPO, args.config)
    cfg = load_config(cfg_path)
    min_seconds = args.seconds if args.seconds is not None else cfg["metadrive"].get("video_seconds", 30)
    out_path = args.out or (os.path.splitext(args.model)[0] + ("_shield.mp4" if args.shield else ".mp4"))

    print(f"loading MetaDrive PPO: {args.model}")
    model = PPO.load(args.model)
    print(f"recording >= {min_seconds:.0f}s (filter={args.shield}) -> {out_path}")
    path = record(model, cfg, out_path, apply_filter=args.shield, min_seconds=min_seconds)

    size = os.path.getsize(path) if path and os.path.exists(path) else 0
    if size > 0:
        print(f"[demo] OK    {path}  ({size // 1024} KB)")
    else:
        print(f"[demo] FAIL  {out_path}  (no file written or empty)")
        sys.exit(1)


if __name__ == "__main__":
    main()
