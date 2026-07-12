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

import numpy as np 

from utils import load_config, save_mp4 
from agents.baselines import load_model 
from envs.metadrive_factory import ( 
    make_env_md, nesy_md_action, count_passes_md, read_scene_md,
)

# Patch torch for numpy 1.x/2.x cross-version compatibility.
import torch as _th 
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


def draw_telemetry(frame, v, omega, fsm_state=None, overtakes=None,
                   lane_changes=None, crashed=False):
    """Overlay speed, yaw-rate, FSM state, overtake/lane-change counts (and CRASHED)."""
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
    if overtakes is not None:
        text += f"\novertakes: {overtakes}"
    if lane_changes is not None:
        text += f"\nlane changes: {lane_changes}"
    pad = max(4, size // 3)
    box = draw.multiline_textbbox((pad, pad), text, font=font)
    draw.rectangle((0, 0, box[2] + pad, box[3] + pad), fill=(0, 0, 0))
    draw.multiline_text((pad, pad), text, fill=(255, 255, 255), font=font)
    if crashed:
        cb = draw.textbbox((0, 0), "CRASHED", font=font)
        cw = cb[2] - cb[0]
        x = (img.width - cw) // 2
        draw.rectangle((x - pad, 0, x + cw + pad, cb[3] + pad), fill=(160, 0, 0))
        draw.text((x, pad), "CRASHED", fill=(255, 255, 255), font=font)
    return np.asarray(img)


def _play_episode(model, cfg, env, grab_frame, seed, shield, max_frames):
    """Play one seeded episode; return (frames, overtakes, lane_changes, steps, crashed)."""
    env.reset(seed=int(seed))
    fsm = cfg["fsm"]["initial_state"]
    _, ahead = count_passes_md(env, set())
    prev_lane = read_scene_md(env)["ego"].get("lane")
    done = False
    ot = lc = steps = 0
    bridge = {}
    frames = []
    info = {}
    while not done and len(frames) < max_frames:
        action, fsm = nesy_md_action(model, env, cfg, fsm, shield=shield, bridge=bridge)
        obs, _, terminated, truncated, info = env.step(action)
        passed, ahead = count_passes_md(env, ahead)
        ot += passed
        steps += 1
        done = terminated or truncated
        sc = read_scene_md(env)
        lane = sc["ego"].get("lane")
        lc += int(lane != prev_lane)
        prev_lane = lane
        frame = grab_frame(env, obs)
        if frame is not None:
            frame = draw_telemetry(
                frame, sc["ego"]["v"],
                float(action[1]) * cfg["metadrive"]["omega_max"],
                fsm_state=fsm, overtakes=ot, lane_changes=lc,
                crashed=done and bool(info.get("crash", False)))
            frames.append(np.asarray(frame))
    return frames, ot, lc, steps, bool(info.get("crash", False))


def _record(model, cfg, env, grab_frame, out_path, shield, min_seconds, fps,
            best_of=None, episode_seed=None):
    """Record episodes and save an MP4. With `episode_seed`, replay exactly that
    eval episode; with `best_of`, keep the most active of N episodes; otherwise
    concatenate episodes until >= min_seconds."""
    target = int(min_seconds * fps)
    seed0 = int(cfg["eval_seeds"][0])
    try:
        if episode_seed is not None:
            frames, ot, lc, steps, crashed = _play_episode(
                model, cfg, env, grab_frame, episode_seed, shield, target)
            print(f"[demo] eval episode (seed {episode_seed}): {ot} overtakes, "
                  f"{lc} lane changes, {steps} steps, crashed={crashed} "
                  f"({len(frames) / fps:.1f}s)", flush=True)
        elif best_of:
            best = None
            for i in range(best_of):
                ep = _play_episode(model, cfg, env, grab_frame, seed0 + i, shield, target)
                print(f"[demo] candidate seed {seed0 + i}: {ep[1]} overtakes, "
                      f"{ep[2]} lane changes in {ep[3]} steps"
                      f"{'  (crashed)' if ep[4] else ''}", flush=True)
                # most action wins: overtakes + lane changes, then longest
                if best is None or (ep[1] + ep[2], ep[3]) > (best[1] + best[2], best[3]):
                    best = ep
            frames, ot, lc, steps, crashed = best
            print(f"[demo] best episode: {ot} overtakes, {lc} lane changes, {steps} steps, "
                  f"crashed={crashed} ({len(frames) / fps:.1f}s)", flush=True)
        else:
            frames = []
            total_ot = total_steps = 0
            ep = 0
            while len(frames) < target:
                ep_frames, ot, lc, steps, _ = _play_episode(
                    model, cfg, env, grab_frame, seed0 + ep, shield, target - len(frames))
                frames.extend(ep_frames)
                total_ot += ot
                total_steps += steps
                print(f"[demo] episode {ep + 1}: {ot} overtakes, {lc} lane changes "
                      f"in {steps} steps", flush=True)
                ep += 1
            print(f"[demo] overtakes: {total_ot} in {total_steps} steps "
                  f"({100.0 * total_ot / max(1, total_steps):.1f} per 100 steps)")
    finally:
        env.close()
    return save_mp4(frames, out_path, fps=fps)


def record_topdown(model, cfg, out_path, shield=True, min_seconds=30,
                   best_of=None, episode_seed=None):
    """Record the top-down 2D view (offscreen, CPU), zoomed on the ego."""
    md = cfg["metadrive"]
    size = tuple(md.get("video_size", [900, 500]))
    scaling = float(md.get("video_scaling", 10.0))
    # same scenario window as evaluate_nesy_md, so an eval episode seed replays
    # the SAME scenario here (MetaDrive wraps seeds into [start_seed, +num_scenarios))
    env = make_env_md(cfg, render=True, seed=int(cfg["seed"]))

    # The renderer clamps scaling to film_height/map_length, so the film must
    # cover the whole map at the requested zoom or the view ends up far away.
    bb = env.unwrapped.current_map.road_network.get_bounding_box()
    film = int(scaling * (max(bb[1] - bb[0], bb[3] - bb[2]) + 20.0))

    def grab(env, obs):
        return env.unwrapped.render(mode="top_down", window=False, screen_size=size,
                                    film_size=(film, film), scaling=scaling)

    return _record(model, cfg, env, grab, out_path, shield, min_seconds,
                   md.get("video_fps", 20), best_of=best_of, episode_seed=episode_seed)


def record_3d(model, cfg, out_path, shield=True, min_seconds=30,
              best_of=None, episode_seed=None):
    """Record a 3D chase-camera video via panda3d offscreen rendering."""
    md = cfg["metadrive"]
    env = make_env_md(cfg, render=False, seed=int(cfg["seed"]), video_3d=True)

    def grab(env, obs):
        try:
            agent = env.unwrapped.agent
            cam = env.unwrapped.engine.get_sensor("rgb_camera")
            img = cam.perceive(
                to_float=False,
                new_parent_node=agent.origin,
                position=(0, -6.0, 2.5),   # close chase cam: 6 m back, 2.5 m up
                hpr=(0, -12, 0),
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

    return _record(model, cfg, env, grab, out_path, shield, min_seconds,
                   md.get("video_fps", 20), best_of=best_of, episode_seed=episode_seed)


def main():
    ap = argparse.ArgumentParser(description="Record a MetaDrive video (discrete model via Lab-1 bridge).")
    ap.add_argument("--model", required=True, help="discrete checkpoint (e.g. dqn.zip)")
    ap.add_argument("--algo", default="dqn", choices=["ppo", "dqn", "qrdqn"])
    ap.add_argument("--no-shield", action="store_true", help="disable FSM shield + CBF/VO filter")
    ap.add_argument("--view", default="topdown", choices=["topdown", "3d"])
    ap.add_argument("--out", default=None, help="output .mp4 path")
    ap.add_argument("--config", default="configs/highway.yaml")
    ap.add_argument("--seconds", type=float, default=None, help="min clip length")
    ap.add_argument("--best-of", type=int, default=None,
                    help="record only the best of N episodes (most overtakes + lane changes)")
    ap.add_argument("--episode-seed", type=int, default=None,
                    help="replay exactly this eval-episode seed (single episode)")
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
        path = record_3d(model, cfg, out_path, shield=shield, min_seconds=min_seconds,
                         best_of=args.best_of, episode_seed=args.episode_seed)
    else:
        path = record_topdown(model, cfg, out_path, shield=shield, min_seconds=min_seconds,
                              best_of=args.best_of, episode_seed=args.episode_seed)

    size = os.path.getsize(path) if path and os.path.exists(path) else 0
    if size > 0:
        print(f"[demo] OK    {path}  ({size // 1024} KB)")
    else:
        print(f"[demo] FAIL  {out_path}  (no frames written)")
        sys.exit(1)


if __name__ == "__main__":
    main()
