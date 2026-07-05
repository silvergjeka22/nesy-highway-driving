"""Standalone demo: record an MP4 of a discrete policy driving MetaDrive via the labs.

Runs OUTSIDE the notebook kernel and renders offscreen, so it never touches the
kernel. The driver is a **discrete Part-1/Part-2 model run through the Lab-1 bridge**
(reconstructed highway obs -> manoeuvre -> (v,ω)); ``--shield`` adds the FSM shield +
CBF/VO filter on top. Two views:

  * ``--view 3d`` (default) — MetaDrive's **3D chase camera** (needs a GPU runtime);
  * ``--view topdown`` (or the auto-fallback) — the CPU pygame top-down view.

Plays episodes until the clip is >= ``--seconds``, prints the overtakes per episode
(the proof it passes traffic), encodes H.264, prints ``[demo] OK/FAIL``.

    python demo/demo_md.py --model <Drive>/checkpoints/dqn.zip --algo dqn --shield
"""

import os
import sys
import argparse
import traceback

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")   # pygame: no audio device on headless
# "XDG_RUNTIME_DIR not set" is a harmless SDL notice on Colab — give SDL a real dir.
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
    make_env_md, nesy_md_action, count_passes_md, read_scene_md, VelocityActionWrapper,
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


def _build_3d_env(cfg, seed):
    """A MetaDrive env with a dedicated ``RGBCamera`` sensor for the 3D video.

    An explicit ``RGBCamera(width, height)`` renders into its own exact-size
    framebuffer — unlike ``main_camera``, whose buffer follows the OS window and
    can come back a few pixels short (784×800 vs the declared 800×800), crashing
    MetaDrive's image observation. Frames are grabbed straight from the sensor
    with a chase-view pose (see ``record_3d``), so the video works wherever the
    3D engine does (GPU/EGL on headless Linux).
    """
    from metadrive.envs import MetaDriveEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera

    md = cfg["metadrive"]
    w, h = md.get("video_size", [800, 800])
    md_config = {
        "use_render": False, "image_observation": True,   # offscreen 3D engine
        "sensors": {"rgb_camera": (RGBCamera, w, h)},
        "vehicle_config": {"image_source": "rgb_camera",
                           "lidar": {"num_lasers": md.get("lidar_num_lasers", 72)}},
        "norm_pixel": False, "stack_size": 1,
        "traffic_density": md["traffic_density"], "num_scenarios": md["num_scenarios"],
        "start_seed": int(seed), "horizon": md["horizon"], "map": md["map"],
    }
    return VelocityActionWrapper(MetaDriveEnv(md_config), cfg)


def _record(model, cfg, env, grab_frame, out_path, shield, min_seconds, fps):
    """Shared loop: bridge-drive ``env``, grab one frame per step, print overtakes.

    Returns the saved MP4 path. ``grab_frame(env, obs)`` supplies the pixels, so
    the 3D and top-down recorders differ only in env construction + frame source.
    """
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


def record_3d(model, cfg, out_path, shield=True, min_seconds=30):
    """Record the 3D chase-camera view (offscreen; needs a GPU/EGL runtime)."""
    md = cfg["metadrive"]
    env = _build_3d_env(cfg, int(cfg["eval_seeds"][0]))

    def grab(env, obs):
        # Borrow the camera for a chase shot behind/above the ego (panda3d:
        # x right, y forward, z up; hpr in degrees). perceive() restores the
        # camera afterwards. The sensor returns BGR -> flip to RGB.
        u = env.unwrapped
        cam = u.engine.get_sensor("rgb_camera")
        bgr = cam.perceive(to_float=False, new_parent_node=u.agent.origin,
                           position=(0.0, -7.5, 3.0), hpr=(0.0, -12.0, 0.0))
        return np.asarray(bgr, dtype=np.uint8)[..., ::-1]

    return _record(model, cfg, env, grab, out_path, shield, min_seconds, md.get("video_fps", 20))


def record_topdown(model, cfg, out_path, shield=True, min_seconds=30):
    """Record the CPU pygame top-down view (offscreen; runs anywhere)."""
    md = cfg["metadrive"]
    size = tuple(md.get("video_size", [800, 800]))
    env = make_env_md(cfg, render=True, seed=int(cfg["eval_seeds"][0]))

    def grab(env, obs):   # MetaDrive's top-down render(mode=...) lives on the base env
        return env.unwrapped.render(mode="top_down", window=False,
                                    screen_size=size, film_size=(size[0] * 2, size[1] * 2))

    return _record(model, cfg, env, grab, out_path, shield, min_seconds, md.get("video_fps", 20))


def _has_gpu():
    """True only if a CUDA/EGL GPU is actually present.

    The 3D chase camera needs one; without it MetaDrive's 3D engine can SEGFAULT
    the whole process (an uncatchable crash that skips the top-down fallback), so
    on a CPU runtime we must NOT even attempt 3D.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    import shutil
    return shutil.which("nvidia-smi") is not None


def main():
    ap = argparse.ArgumentParser(description="Record a MetaDrive driving video (discrete model via the Lab-1 bridge).")
    ap.add_argument("--model", required=True, help="a discrete Part-1/Part-2 checkpoint (e.g. dqn.zip, part2_nesy.zip)")
    ap.add_argument("--algo", default="dqn", choices=["ppo", "dqn", "qrdqn"], help="algorithm of the checkpoint")
    ap.add_argument("--view", default="auto", choices=["auto", "3d", "topdown"],
                    help="auto = 3d chase camera only if a GPU is present, else cpu top-down (default: auto)")
    ap.add_argument("--no-shield", action="store_true",
                    help="drive WITHOUT the FSM shield + CBF/VO filter (default: shield on)")
    ap.add_argument("--out", default=None, help="output .mp4 path (default: alongside the model)")
    ap.add_argument("--config", default="configs/highway.yaml", help="config YAML")
    ap.add_argument("--seconds", type=float, default=None, help="min length (default: metadrive.video_seconds)")
    args = ap.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(_REPO, args.config)
    cfg = load_config(cfg_path)
    min_seconds = args.seconds if args.seconds is not None else cfg["metadrive"].get("video_seconds", 30)
    shield = not args.no_shield
    out_path = args.out or (os.path.splitext(args.model)[0] + "_metadrive.mp4")

    view = args.view
    if view == "auto":
        view = "3d" if _has_gpu() else "topdown"
        print(f"[demo] auto view -> {view} ({'GPU present' if view == '3d' else 'no GPU: CPU top-down'})")

    print(f"loading {args.algo} model (Lab-1 bridge driver): {args.model}")
    model = load_model(args.model, args.algo)
    print(f"recording >= {min_seconds:.0f}s (view={view}, shield={shield}) -> {out_path}")

    path = None
    if view == "3d":
        try:
            path = record_3d(model, cfg, out_path, shield=shield, min_seconds=min_seconds)
        except Exception:   # 3D renderer unavailable (e.g. CPU runtime) -> top-down
            print("[demo] 3D render failed — full traceback (fix or use a GPU runtime):")
            traceback.print_exc()
            print("[demo] falling back to the CPU top-down view.")
            path = None
    if path is None:
        path = record_topdown(model, cfg, out_path, shield=shield, min_seconds=min_seconds)

    size = os.path.getsize(path) if path and os.path.exists(path) else 0
    if size > 0:
        print(f"[demo] OK    {path}  ({size // 1024} KB)")
    else:
        print(f"[demo] FAIL  {out_path}  (no file written or empty). The top-down "
              f"recorder produced no frames — check the traceback above; on a CPU "
              f"runtime the 3D view is skipped automatically.")
        sys.exit(1)


if __name__ == "__main__":
    main()
