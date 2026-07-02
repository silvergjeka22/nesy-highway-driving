"""Standalone demo: record an MP4 of a MetaDrive policy and save it to Drive.

Runs OUTSIDE the notebook kernel and renders offscreen, so it never touches the
kernel. Two views:

  * ``--view 3d`` (default) — MetaDrive's **3D chase camera** (``main_camera``),
    rendered offscreen. Needs a **GPU runtime** (the 3D renderer uses OpenGL). The
    PPO drives on its vector observation, which we reconstruct alongside the image
    obs the 3D engine produces.
  * ``--view topdown`` — the pygame **top-down** view (CPU, no GPU). Also the
    automatic fallback if the 3D renderer is unavailable.

Plays episodes until the clip is >= ``--seconds``, encodes an H.264 MP4, prints
``[demo] OK/FAIL``.

Usage (the Part-3 notebook launches this as a background subprocess):
    python demo/demo_md.py --model <RESULTS>/checkpoints/part3_metadrive.zip --shield
"""

import os
import sys
import argparse

# MetaDrive's renderers use pygame/panda3d; disable audio so pygame.init() cannot
# segfault on a headless machine.
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

from stable_baselines3 import PPO  # noqa: E402
from utils import load_config, save_mp4  # noqa: E402
from envs.metadrive_factory import (  # noqa: E402
    make_env_md, read_scene_md, filter_action_md, VelocityActionWrapper,
)


def _build_3d_env(cfg, seed):
    """A MetaDrive env with the 3D ``main_camera`` image buffer (for offscreen capture)."""
    from metadrive.envs import MetaDriveEnv

    md = cfg["metadrive"]
    w, h = md.get("video_size", [800, 800])
    md_config = {
        "use_render": False,          # no popup window
        "image_observation": True,    # spin up the offscreen 3D renderer
        "sensors": {"main_camera": ()},
        "vehicle_config": {"image_source": "main_camera",
                           "lidar": {"num_lasers": md.get("lidar_num_lasers", 72)}},
        "window_size": (w, h),
        "norm_pixel": False,          # uint8 [0,255] frames
        "stack_size": 3,
        "traffic_density": md["traffic_density"],
        "num_scenarios": md["num_scenarios"],
        "start_seed": int(seed),
        "horizon": md["horizon"],
        "map": md["map"],
    }
    return VelocityActionWrapper(MetaDriveEnv(md_config), cfg)


def record_3d(model, cfg, out_path, apply_filter=False, min_seconds=30):
    """Record the 3D chase-camera view (offscreen, GPU). Returns the saved path."""
    from metadrive.obs.state_obs import LidarStateObservation

    md = cfg["metadrive"]
    fps = md.get("video_fps", 20)
    target = int(min_seconds * fps)
    seed0 = int(cfg["eval_seeds"][0])

    env = _build_3d_env(cfg, seed0)
    # The 3D env's observation is the image; rebuild the vector obs the PPO expects.
    policy_obs = LidarStateObservation(env.unwrapped.config)

    frames = []
    try:
        ep = 0
        while len(frames) < target:
            obs, _ = env.reset(seed=seed0 + ep)
            done = False
            while not done and len(frames) < target:
                vec = policy_obs.observe(env.unwrapped.agent)
                action, _ = model.predict(vec, deterministic=cfg["eval"]["deterministic"])
                if apply_filter:
                    action = filter_action_md(action, read_scene_md(env), cfg)
                obs, _, terminated, truncated, _ = env.step(action)
                frames.append(np.asarray(obs["image"][..., -1], dtype=np.uint8))
                done = terminated or truncated
            ep += 1
    finally:
        env.close()

    return save_mp4(frames, out_path, fps=fps)


def record_topdown(model, cfg, out_path, apply_filter=False, min_seconds=30):
    """Record the CPU pygame top-down view (offscreen). Returns the saved path."""
    md = cfg["metadrive"]
    fps = md.get("video_fps", 20)
    size = tuple(md.get("video_size", [800, 800]))
    target = int(min_seconds * fps)
    seed0 = int(cfg["eval_seeds"][0])

    env = make_env_md(cfg, render=True, seed=seed0)
    base = env.unwrapped   # MetaDrive's top-down render(mode=...) lives on the base env
    frames = []
    try:
        ep = 0
        while len(frames) < target:
            obs, _ = env.reset(seed=seed0 + ep)
            done = False
            while not done and len(frames) < target:
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
    ap = argparse.ArgumentParser(description="Record a MetaDrive driving video.")
    ap.add_argument("--model", required=True, help="path to the .zip checkpoint (from Drive)")
    ap.add_argument("--view", default="3d", choices=["3d", "topdown"],
                    help="3d chase camera (needs GPU) or cpu top-down (default: 3d)")
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
    print(f"recording >= {min_seconds:.0f}s (view={args.view}, filter={args.shield}) -> {out_path}")

    path = None
    if args.view == "3d":
        try:
            path = record_3d(model, cfg, out_path, apply_filter=args.shield, min_seconds=min_seconds)
        except Exception as e:  # 3D renderer unavailable (e.g. CPU runtime) -> top-down
            print(f"[demo] 3D render failed ({type(e).__name__}: {e}); falling back to top-down.")
            path = None
    if path is None:
        path = record_topdown(model, cfg, out_path, apply_filter=args.shield, min_seconds=min_seconds)

    size = os.path.getsize(path) if path and os.path.exists(path) else 0
    if size > 0:
        print(f"[demo] OK    {path}  ({size // 1024} KB)")
    else:
        print(f"[demo] FAIL  {out_path}  (no file written or empty)")
        sys.exit(1)


if __name__ == "__main__":
    main()
