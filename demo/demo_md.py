"""Standalone demo: record an MP4 of a discrete policy driving MetaDrive via the labs.

Runs OUTSIDE the notebook kernel and renders offscreen, so it never touches the
kernel. The driver is a **discrete Part-1/Part-2 model run through the Lab-1 bridge**
(reconstructed highway obs -> manoeuvre -> (v,ω)); ``--shield`` adds the FSM shield +
CBF/VO filter on top.

Two views:
  * ``--view 3d``      — 3D chase-camera via panda3d offscreen rendering (GPU runtime).
  * ``--view topdown``  — top-down pygame view (CPU, works everywhere).

Plays episodes until the clip is >= ``--seconds``, prints the overtakes per episode
(the proof it passes traffic), encodes H.264, prints ``[demo] OK/FAIL``.

    python demo/demo_md.py --model <Drive>/checkpoints/dqn.zip --algo dqn --shield
    python demo/demo_md.py --model <ckpt> --algo dqn --view 3d
"""

import os
import sys
import argparse

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")   # pygame: no audio device on headless
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # offscreen rendering, no display needed
_xdg = os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/xdg-nesy")
try:
    os.makedirs(_xdg, mode=0o700, exist_ok=True)
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


def draw_telemetry(frame, v, omega, fsm_state=None):
    """Print live telemetry onto one video frame; returns the annotated RGB array.

    Top-left corner, white on black: the ego's current speed ``v`` [m/s], the
    commanded yaw-rate ``omega`` [rad/s], and (when given) the FSM state. Pure
    function on the pixel array — works for both the 3D chase-cam and the
    top-down frames. PIL is already a dependency (imageio/matplotlib stack).
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(img)
    size = max(14, img.height // 30)
    try:
        font = ImageFont.load_default(size=size)     # Pillow >= 10
    except TypeError:                                 # older Pillow: fixed-size bitmap font
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
    """Drive ``env``, grab one frame per step via ``grab_frame(env, obs)``, print overtakes."""
    target = int(min_seconds * fps)
    seed0 = int(cfg["eval_seeds"][0])
    frames = []
    total_ot = total_steps = 0
    try:
        ep = 0
        while len(frames) < target:
            obs, _ = env.reset(seed=seed0 + ep)
            fsm = cfg["fsm"]["initial_state"]
            _, ahead = count_passes_md(env, set())     # prime the ahead-set
            done = False
            ot = steps = 0
            while not done and len(frames) < target:
                action, fsm = nesy_md_action(model, env, cfg, fsm, shield=shield)
                obs, _, terminated, truncated, _ = env.step(action)
                passed, ahead = count_passes_md(env, ahead)
                ot += passed
                steps += 1
                frame = grab_frame(env, obs)
                if frame is not None:
                    # Live telemetry on every frame: measured ego speed + the
                    # commanded yaw-rate (the normalised action × omega_max).
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
    """Record the top-down pygame view (offscreen, CPU). Runs on any runtime."""
    md = cfg["metadrive"]
    size = tuple(md.get("video_size", [800, 800]))
    env = make_env_md(cfg, render=True, seed=int(cfg["eval_seeds"][0]))

    def grab(env, obs):   # MetaDrive's top-down render(mode=...) lives on the base env
        return env.unwrapped.render(mode="top_down", window=False,
                                    screen_size=size, film_size=(size[0] * 2, size[1] * 2))

    return _record(model, cfg, env, grab, out_path, shield, min_seconds, md.get("video_fps", 20))


def record_3d(model, cfg, out_path, shield=True, min_seconds=30):
    """Record a 3D chase-camera video via MetaDrive's offscreen rendering pipeline."""
    md = cfg["metadrive"]
    env = make_env_md(cfg, render=False, seed=int(cfg["eval_seeds"][0]), video_3d=True)

    def grab(env, obs):
        if isinstance(obs, dict) and "image" in obs:
            img = obs["image"]
            if img.ndim == 4:
                img = img[..., -1]
            img = np.asarray(img)
            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            if img.ndim == 3 and img.shape[2] > 3:
                img = img[:, :, :3]
            return img
        try:
            cam = env.unwrapped.engine.get_sensor("main_camera")
            img = cam.perceive(to_float=False)
            return np.asarray(img, dtype=np.uint8)[:, :, :3]
        except Exception:
            return None

    return _record(model, cfg, env, grab, out_path, shield, min_seconds, md.get("video_fps", 20))


def main():
    ap = argparse.ArgumentParser(description="Record a MetaDrive driving video (discrete model via the Lab-1 bridge).")
    ap.add_argument("--model", required=True, help="a discrete Part-1/Part-2 checkpoint (e.g. dqn.zip, part2_nesy.zip)")
    ap.add_argument("--algo", default="dqn", choices=["ppo", "dqn", "qrdqn"], help="algorithm of the checkpoint")
    ap.add_argument("--no-shield", action="store_true",
                    help="drive WITHOUT the FSM shield + CBF/VO filter (default: shield on)")
    ap.add_argument("--view", default="topdown", choices=["topdown", "3d"],
                    help="3d = chase-camera via offscreen panda3d; topdown = 2D pygame (default)")
    ap.add_argument("--out", default=None, help="output .mp4 path (default: alongside the model)")
    ap.add_argument("--config", default="configs/highway.yaml", help="config YAML")
    ap.add_argument("--seconds", type=float, default=None, help="min length (default: metadrive.video_seconds)")
    args = ap.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(_REPO, args.config)
    cfg = load_config(cfg_path)
    min_seconds = args.seconds if args.seconds is not None else cfg["metadrive"].get("video_seconds", 30)
    shield = not args.no_shield
    out_path = args.out or (os.path.splitext(args.model)[0] + "_metadrive.mp4")

    print(f"loading {args.algo} model (Lab-1 bridge driver): {args.model}")
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
        print(f"[demo] FAIL  {out_path}  (no file written or empty)")
        sys.exit(1)


if __name__ == "__main__":
    main()
