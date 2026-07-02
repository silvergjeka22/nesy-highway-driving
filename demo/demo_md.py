"""Standalone demo: record an MP4 of a MetaDrive policy and save it to Drive.

Runs OUTSIDE the notebook kernel and renders offscreen, so it never touches the
kernel. Two drivers and two views:

  * ``--nesy`` (Part 3) — drive with the **Part-2 discrete model via the Lab-1 bridge**
    (reconstructed highway obs -> manoeuvre -> (v,ω) -> CBF/VO); ``--model`` is the
    part2_nesy checkpoint. Without ``--nesy`` it drives a continuous MetaDrive PPO.
  * ``--view 3d`` (default) — MetaDrive's **3D chase camera** (needs a GPU runtime);
    ``--view topdown`` (or the auto-fallback) — the CPU pygame top-down view.

Plays episodes until the clip is >= ``--seconds``, encodes H.264, prints ``[demo] OK/FAIL``.

    python demo/demo_md.py --model <Drive>/checkpoints/part2_nesy.zip --nesy
"""

import os
import sys
import argparse

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")   # pygame: no audio device on headless

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

from stable_baselines3 import PPO  # noqa: E402
from utils import load_config, save_mp4  # noqa: E402
from envs.metadrive_factory import (  # noqa: E402
    make_env_md, read_scene_md, filter_action_md, nesy_md_action, VelocityActionWrapper,
)


def _act(model, obs, env, cfg, nesy, apply_filter, fsm):
    """Produce the next normalised MetaDrive action (and updated fsm state)."""
    if nesy:                                          # Part-2 model via the Lab-1 bridge
        return nesy_md_action(model, env, cfg, fsm, shield=True)
    action, _ = model.predict(obs, deterministic=cfg["eval"]["deterministic"])
    if apply_filter:                                 # continuous PPO + CBF/VO filter
        action = filter_action_md(action, read_scene_md(env), cfg)
    return action, fsm


def _build_3d_env(cfg, seed):
    """A MetaDrive env with the 3D ``main_camera`` image buffer (for offscreen capture)."""
    from metadrive.envs import MetaDriveEnv

    md = cfg["metadrive"]
    w, h = md.get("video_size", [800, 800])
    md_config = {
        "use_render": False, "image_observation": True,
        "sensors": {"main_camera": ()},
        "vehicle_config": {"image_source": "main_camera",
                           "lidar": {"num_lasers": md.get("lidar_num_lasers", 72)}},
        "window_size": (w, h), "norm_pixel": False, "stack_size": 3,
        "traffic_density": md["traffic_density"], "num_scenarios": md["num_scenarios"],
        "start_seed": int(seed), "horizon": md["horizon"], "map": md["map"],
    }
    return VelocityActionWrapper(MetaDriveEnv(md_config), cfg)


def record_3d(model, cfg, out_path, nesy=False, apply_filter=False, min_seconds=30):
    """Record the 3D chase-camera view (offscreen, GPU). Returns the saved path."""
    md = cfg["metadrive"]
    fps = md.get("video_fps", 20)
    target = int(min_seconds * fps)
    seed0 = int(cfg["eval_seeds"][0])
    env = _build_3d_env(cfg, seed0)
    frames = []
    try:
        ep = 0
        while len(frames) < target:
            obs, _ = env.reset(seed=seed0 + ep)
            fsm = cfg["fsm"]["initial_state"]
            done = False
            while not done and len(frames) < target:
                action, fsm = _act(model, obs, env, cfg, nesy, apply_filter, fsm)
                obs, _, terminated, truncated, _ = env.step(action)
                frames.append(np.asarray(obs["image"][..., -1], dtype=np.uint8))
                done = terminated or truncated
            ep += 1
    finally:
        env.close()
    return save_mp4(frames, out_path, fps=fps)


def record_topdown(model, cfg, out_path, nesy=False, apply_filter=False, min_seconds=30):
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
            fsm = cfg["fsm"]["initial_state"]
            done = False
            while not done and len(frames) < target:
                action, fsm = _act(model, obs, env, cfg, nesy, apply_filter, fsm)
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
    ap.add_argument("--model", required=True, help="checkpoint: part2_nesy.zip (with --nesy) or a MetaDrive PPO")
    ap.add_argument("--nesy", action="store_true", help="drive the Part-2 model via the Lab-1 bridge")
    ap.add_argument("--view", default="3d", choices=["3d", "topdown"],
                    help="3d chase camera (needs GPU) or cpu top-down (default: 3d)")
    ap.add_argument("--shield", action="store_true", help="continuous CBF/VO filter (for a plain PPO model)")
    ap.add_argument("--out", default=None, help="output .mp4 path (default: alongside the model)")
    ap.add_argument("--config", default="configs/highway.yaml", help="config YAML")
    ap.add_argument("--seconds", type=float, default=None, help="min length (default: metadrive.video_seconds)")
    args = ap.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(_REPO, args.config)
    cfg = load_config(cfg_path)
    min_seconds = args.seconds if args.seconds is not None else cfg["metadrive"].get("video_seconds", 30)
    suffix = "_nesy.mp4" if args.nesy else ("_shield.mp4" if args.shield else ".mp4")
    out_path = args.out or (os.path.splitext(args.model)[0] + suffix)

    print(f"loading {'Part-2 (bridge)' if args.nesy else 'MetaDrive'} PPO: {args.model}")
    model = PPO.load(args.model)
    print(f"recording >= {min_seconds:.0f}s (view={args.view}, nesy={args.nesy}) -> {out_path}")

    kw = dict(nesy=args.nesy, apply_filter=args.shield, min_seconds=min_seconds)
    path = None
    if args.view == "3d":
        try:
            path = record_3d(model, cfg, out_path, **kw)
        except Exception as e:   # 3D renderer unavailable (e.g. CPU runtime) -> top-down
            print(f"[demo] 3D render failed ({type(e).__name__}: {e}); falling back to top-down.")
            path = None
    if path is None:
        path = record_topdown(model, cfg, out_path, **kw)

    size = os.path.getsize(path) if path and os.path.exists(path) else 0
    if size > 0:
        print(f"[demo] OK    {path}  ({size // 1024} KB)")
    else:
        print(f"[demo] FAIL  {out_path}  (no file written or empty)")
        sys.exit(1)


if __name__ == "__main__":
    main()
