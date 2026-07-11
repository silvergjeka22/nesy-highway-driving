"""Record an MP4 of a discrete policy driving MetaDrive via the Lab-1 bridge (Part 3).

Two views:
  --view topdown  — 2D top-down (CPU, works everywhere)
  --view 3d       — 3D chase-camera via panda3d offscreen (GPU runtime)

Usage:
    python demo/demo_md.py --model .../dqn.zip --algo dqn --view topdown
    python demo/demo_md.py --model .../dqn.zip --algo dqn --view 3d
"""

import os
import sys
import argparse

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/xdg-nesy")
try:
    os.makedirs("/tmp/xdg-nesy", mode=0o700, exist_ok=True)
except OSError:
    pass

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

from utils import load_config, save_mp4  # noqa: E402
from agents.baselines import load_model  # noqa: E402
from envs.metadrive_factory import (  # noqa: E402
    make_env_md, nesy_md_action, count_passes_md, read_scene_md,
)

# Patch torch for numpy 1.x/2.x cross-version compatibility.
import torch as _th  # noqa: E402
_orig_as_tensor = _th.as_tensor
def _as_tensor_compat(data, dtype=None, device=None):
    try:
        return _orig_as_tensor(data, dtype=dtype, device=device)
    except RuntimeError:
        if hasattr(data, "tolist"):
            return _orig_as_tensor(data.tolist(), dtype=dtype, device=device)
        raise
_th.as_tensor = _as_tensor_compat

_orig_tensor_numpy = _th.Tensor.numpy
def _tensor_numpy_compat(self, *args, **kwargs):
    try:
        return _orig_tensor_numpy(self, *args, **kwargs)
    except RuntimeError:
        return np.asarray(self.detach().cpu().tolist())
_th.Tensor.numpy = _tensor_numpy_compat


def draw_telemetry(frame, v, omega, fsm_state=None):
    """Overlay speed, yaw-rate, and FSM state onto a video frame."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(img)
    size = max(14, img.height // 30)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:
        font = ImageFont.load_default()
    text = f"v = {v:5.2f} m/s\nomega = {omega:+.2f} rad/s"
    if fsm_state:
        text += f"\nFSM: {fsm_state}"
    pad = max(4, size // 3)
    box = draw.multiline_textbbox((pad, pad), text, font=font)
    draw.rectangle((0, 0, box[2] + pad, box[3] + pad), fill=(0, 0, 0))
    draw.multiline_text((pad, pad), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def _record(model, cfg, env, grab_frame, out_path, shield, min_seconds, fps):
    """Drive env, grab frames, print overtakes per episode, save MP4."""
    target = int(min_seconds * fps)
    seed0 = int(cfg["eval_seeds"][0])
    frames = []
    total_ot = total_steps = 0
    try:
        ep = 0
        while len(frames) < target:
            obs, _ = env.reset(seed=seed0 + ep)
            fsm = cfg["fsm"]["initial_state"]
            _, ahead = count_passes_md(env, set())
            done = False
            ot = steps = 0
            bridge = {}
            while not done and len(frames) < target:
                action, fsm = nesy_md_action(model, env, cfg, fsm, shield=shield, bridge=bridge)
                obs, _, terminated, truncated, _ = env.step(action)
                passed, ahead = count_passes_md(env, ahead)
                ot += passed
                steps += 1
                frame = grab_frame(env, obs)
                if frame is not None:
                    frame = draw_telemetry(
                        frame, read_scene_md(env)["ego"]["v"],
                        float(action[1]) * cfg["metadrive"]["omega_max"],
                        fsm_state=fsm)
                    frames.append(np.asarray(frame))
                done = terminated or truncated
            total_ot += ot
            total_steps += steps
            print(f"[demo] episode {ep + 1}: {ot} overtakes in {steps} steps", flush=True)
            ep += 1
    finally:
        env.close()
    print(f"[demo] overtakes: {total_ot} in {total_steps} steps "
          f"({100.0 * total_ot / max(1, total_steps):.1f} per 100 steps)")
    return save_mp4(frames, out_path, fps=fps)


def record_topdown(model, cfg, out_path, shield=True, min_seconds=30):
    """Record the top-down 2D view (offscreen, CPU)."""
    md = cfg["metadrive"]
    size = tuple(md.get("video_size", [800, 800]))
    env = make_env_md(cfg, render=True, seed=int(cfg["eval_seeds"][0]))

    def grab(env, obs):
        return env.unwrapped.render(mode="top_down", window=False,
                                    screen_size=size, film_size=(size[0] * 2, size[1] * 2))

    return _record(model, cfg, env, grab, out_path, shield, min_seconds, md.get("video_fps", 20))


def record_3d(model, cfg, out_path, shield=True, min_seconds=30):
    """Record a 3D chase-camera video via panda3d offscreen rendering."""
    md = cfg["metadrive"]
    env = make_env_md(cfg, render=False, seed=int(cfg["eval_seeds"][0]), video_3d=True)

    def grab(env, obs):
        try:
            agent = env.unwrapped.agent
            cam = env.unwrapped.engine.get_sensor("rgb_camera")
            img = cam.perceive(
                to_float=False,
                new_parent_node=agent.origin,
                position=(0, -7.5, 3.5),
                hpr=(0, -15, 0),
            )
            img = np.asarray(img, dtype=np.uint8)
            if img.ndim == 3 and img.shape[2] > 3:
                img = img[:, :, :3]
            if img.ndim == 3 and img.shape[2] == 3:
                img = img[..., ::-1]
            return img
        except Exception as e:
            print(f"[3d] frame grab failed: {e}", flush=True)
            return None

    return _record(model, cfg, env, grab, out_path, shield, min_seconds, md.get("video_fps", 20))


def main():
    ap = argparse.ArgumentParser(description="Record a MetaDrive video (discrete model via Lab-1 bridge).")
    ap.add_argument("--model", required=True, help="discrete checkpoint (e.g. dqn.zip)")
    ap.add_argument("--algo", default="dqn", choices=["ppo", "dqn", "qrdqn"])
    ap.add_argument("--no-shield", action="store_true", help="disable FSM shield + CBF/VO filter")
    ap.add_argument("--view", default="topdown", choices=["topdown", "3d"])
    ap.add_argument("--out", default=None, help="output .mp4 path")
    ap.add_argument("--config", default="configs/highway.yaml")
    ap.add_argument("--seconds", type=float, default=None, help="min clip length")
    args = ap.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(_REPO, args.config)
    cfg = load_config(cfg_path)
    min_seconds = args.seconds if args.seconds is not None else cfg["metadrive"].get("video_seconds", 30)
    shield = not args.no_shield
    out_path = args.out or (os.path.splitext(args.model)[0] + "_metadrive.mp4")

    print(f"loading {args.algo} model (Lab-1 bridge): {args.model}")
    model = load_model(args.model, args.algo)
    print(f"recording >= {min_seconds:.0f}s (view={args.view}, shield={shield}) -> {out_path}")

    if args.view == "3d":
        path = record_3d(model, cfg, out_path, shield=shield, min_seconds=min_seconds)
    else:
        path = record_topdown(model, cfg, out_path, shield=shield, min_seconds=min_seconds)

    size = os.path.getsize(path) if path and os.path.exists(path) else 0
    if size > 0:
        print(f"[demo] OK    {path}  ({size // 1024} KB)")
    else:
        print(f"[demo] FAIL  {out_path}  (no frames written)")
        sys.exit(1)


if __name__ == "__main__":
    main()
