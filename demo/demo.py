"""Standalone demo: record an MP4 of a saved policy driving highway-v0.

Runs OUTSIDE any notebook kernel and renders offscreen, so pygame can never crash
a kernel. Works for Part-1 (PPO/DQN/QR-DQN) and Part-2 (NeSy,
``--shield``) checkpoints.

Usage:
    python demo/demo.py --model .../checkpoints/part1_best.zip --algo ppo
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
# SDL/pygame print "XDG_RUNTIME_DIR not set in the environment" on Colab (no login
# session sets it). It is a harmless notice, not the cause of a failure — give SDL a
# real runtime dir so the message stops at the source rather than suppressing it.
_xdg = os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/xdg-nesy")
try:
    os.makedirs(_xdg, mode=0o700, exist_ok=True)
except OSError:
    pass
if sys.platform == "linux" and not os.environ.get("DISPLAY"):
    try:
        from pyvirtualdisplay import Display

        # color_depth=24 is required: pygame/SDL2 on Colab SEGFAULTS (exit -11) when
        # it opens a surface on xvfb's default 16-bit visual, so the video subprocess
        # died before writing a frame. Pinning a 24-bit display fixes the crash.
        _XVFB = Display(visible=0, size=(1400, 900), color_depth=24)
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
from agents.baselines import load_model, as_predictor, build_ppo, build_dqn, build_qrdqn  # noqa: E402
from nesy.roadmap import predicates, safety_shield  # noqa: E402

_BUILDERS = {"ppo": build_ppo, "dqn": build_dqn, "qrdqn": build_qrdqn}


def load_policy_weights(weights_path, algo, cfg):
    """Rebuild a model from a torch policy ``state_dict`` (``.pt``).

    Cross-version escape hatch: a full SB3 ``.zip`` saved on Colab (newer SB3 +
    NumPy 2) can't be unpickled on an Intel-Mac stack (older SB3 + NumPy 1) — it
    trips on ``FloatSchedule`` and the NumPy-2 RNG. A bare ``state_dict`` is just
    torch tensors, so it loads anywhere. We build a fresh model from the config
    (identical architecture) and copy the weights into its policy.
    """
    import torch

    env = create_environment(cfg)  # only needed for the observation/action spaces
    try:
        model = _BUILDERS[algo](cfg, env, device="cpu")
        state = torch.load(weights_path, map_location="cpu")
        model.policy.load_state_dict(state)
    finally:
        env.close()
    return model


class _FrameHook:
    """Collect the intermediate simulation frames highway-env renders during a
    step. The env only *decides* at ``policy_frequency`` (2 Hz), but simulates at
    ``simulation_frequency`` (10 Hz): capturing one frame per decision plays back
    5× too fast and jerky. highway-env's ``_automatic_rendering`` calls
    ``_record_video_wrapper._capture_frame()`` at every physics substep — this
    shim implements that interface and appends each substep frame to the current
    episode's list."""

    def __init__(self, env):
        self.env = env
        self.frames = None            # rebound to each episode's frame list

    def _capture_frame(self):
        frame = self.env.render()
        if frame is not None and self.frames is not None:
            self.frames.append(np.asarray(frame))


def record(model, cfg, out_path, apply_shield=False, min_seconds=30, no_crash=False):
    """Play episodes until the clip is >= ``min_seconds`` and save an MP4.

    Frames are captured at the SIMULATION frequency (every physics substep, via
    :class:`_FrameHook`), not just at the 2 Hz decision points, and encoded at
    that same rate — so the clip is smooth and plays in real time. Prints one
    line per episode (overtakes, steps, crash) so the demo visibly proves the
    policy passes traffic. ``no_crash=True`` keeps only crash-free episodes
    (resamples fresh seeds and discards any run that crashes), so the saved clip
    shows clean driving only. Returns ``(path, total_overtakes, total_steps)``.
    """
    fps = int(cfg["env"]["config"].get("simulation_frequency", 15))   # real-time playback
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
            if hasattr(model, "reset_states"):    # stateful policy hook (feed-forward: no-op)
                model.reset_states()
            fsm_state = cfg["fsm"]["initial_state"]
            done = False
            steps = 0
            ep_frames = []
            hook.frames = ep_frames
            hook._capture_frame()     # first render creates the viewer and turns
                                      # on auto-rendering of the substeps
            while not done:
                action, _ = model.predict(obs, deterministic=cfg["eval"]["deterministic"])
                if apply_shield:
                    action, fsm_state = safety_shield(action, predicates(read_scene(env), cfg), fsm_state, cfg)
                obs, _, terminated, truncated, info = env.step(action)
                hook._capture_frame()             # the step's final substep frame
                steps += 1
                done = terminated or truncated
            crashed = bool(info.get("crashed"))
            ot = int(info.get("overtakes", 0))
            if no_crash and crashed:
                print(f"[demo] episode {ep + 1}: crashed after {steps} steps — discarded (--no-crash)",
                      flush=True)
                ep += 1
                continue
            frames.extend(ep_frames)
            total_overtakes += ot
            total_steps += steps
            print(f"[demo] episode {ep + 1}: {ot} overtakes in {steps} steps "
                  f"({len(ep_frames) / fps:.1f}s of video)"
                  f"{'  (crashed)' if crashed else ''}", flush=True)
            ep += 1
    finally:
        env.unwrapped._record_video_wrapper = None
        env.close()

    print(f"[demo] clip: {len(frames)} frames @ {fps} fps = {len(frames) / fps:.1f}s", flush=True)
    return save_mp4(frames, out_path, fps=fps), total_overtakes, total_steps


def main():
    ap = argparse.ArgumentParser(description="Record a driving video from a saved model.")
    ap.add_argument("--model", required=True,
                    help="checkpoint: SB3 .zip or a bare policy state_dict .pt")
    ap.add_argument("--algo", default="ppo", choices=["ppo", "dqn", "qrdqn"],
                    help="algorithm (default: ppo)")
    ap.add_argument("--shield", action="store_true", help="apply the NeSy safety shield (Part 2)")
    ap.add_argument("--out", default=None, help="output .mp4 path (default: alongside the model)")
    ap.add_argument("--config", default="configs/highway.yaml", help="config YAML")
    ap.add_argument("--seconds", type=float, default=None, help="min video length (default: config eval.video_seconds)")
    ap.add_argument("--duration", type=int, default=None,
                    help="episode length in sim-seconds; pass a huge value (e.g. 100000) for effectively infinite episodes that end only on a crash")
    ap.add_argument("--no-crash", dest="no_crash", action="store_true",
                    help="record only crash-free episodes (discard and resample any run that crashes)")
    args = ap.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(_REPO, args.config)
    cfg = load_config(cfg_path)
    if args.duration is not None:
        cfg["env"]["config"]["duration"] = args.duration
    min_seconds = args.seconds if args.seconds is not None else cfg["eval"].get("video_seconds", 30)
    out_path = args.out or (os.path.splitext(args.model)[0] + ("_shield.mp4" if args.shield else ".mp4"))

    print(f"loading {args.algo} model: {args.model}")
    if args.model.endswith((".pt", ".pth")):
        # weights-only path (torch state_dict) — cross-version safe
        model = load_policy_weights(args.model, args.algo, cfg)
    else:
        model = load_model(args.model, args.algo)
    model = as_predictor(model, args.algo)   # feed-forward models pass through unchanged
    print(f"recording >= {min_seconds:.0f}s (shield={args.shield}) -> {out_path}")
    path, overtakes, steps = record(model, cfg, out_path, apply_shield=args.shield,
                                    min_seconds=min_seconds, no_crash=args.no_crash)

    size = os.path.getsize(path) if path and os.path.exists(path) else 0
    if size > 0:
        print(f"[demo] overtakes: {overtakes} in {steps} steps "
              f"({100.0 * overtakes / max(1, steps):.1f} per 100 steps)")
        print(f"[demo] OK    {path}  ({size // 1024} KB)")
    else:
        print(f"[demo] FAIL  {out_path}  (no file written or empty — {steps} steps rendered "
              f"0 usable frames). Most common cause on Colab: plain pygame crashing the "
              f"renderer. Fix: `pip uninstall -y pygame && pip install pygame-ce`, then rerun.")
        sys.exit(1)


if __name__ == "__main__":
    main()
