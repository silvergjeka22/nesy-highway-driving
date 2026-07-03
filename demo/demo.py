"""Standalone demo: record an MP4 of a saved policy driving highway-v0.

Runs OUTSIDE any notebook kernel and renders offscreen, so pygame can never crash
a kernel. Works for Part-1 (RecurrentPPO/DQN/QR-DQN) and Part-2 (NeSy,
``--shield``) checkpoints.

Usage:
    python demo/demo.py --model .../checkpoints/part1_best.zip --algo rppo
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
from agents.baselines import load_model, as_predictor, build_rppo, build_dqn, build_qrdqn  # noqa: E402
from nesy.roadmap import predicates, safety_shield  # noqa: E402

_BUILDERS = {"rppo": build_rppo, "dqn": build_dqn, "qrdqn": build_qrdqn}


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


def record(model, cfg, out_path, apply_shield=False, min_seconds=30, no_crash=False):
    """Play episodes until the clip is >= ``min_seconds`` and save an MP4.

    Prints one line per episode (overtakes, steps, crash) so the demo visibly
    proves the policy passes traffic. ``no_crash=True`` keeps only crash-free
    episodes (resamples fresh seeds and discards any run that crashes), so the
    saved clip shows clean driving only. Returns
    ``(path, total_overtakes, total_steps)``.
    """
    fps = cfg["eval"].get("video_fps", 10)
    target_frames = int(min_seconds * fps)
    max_episodes = int(cfg["eval"].get("video_max_episodes", 60))
    seed0 = int(cfg["eval_seeds"][0])

    env = create_environment(cfg, render=True)
    frames = []
    total_overtakes = total_steps = 0
    try:
        ep = 0
        while len(frames) < target_frames and ep < max_episodes:
            obs, info = env.reset(seed=seed0 + ep)
            if hasattr(model, "reset_states"):    # RecurrentPPO: fresh LSTM state
                model.reset_states()
            fsm_state = cfg["fsm"]["initial_state"]
            done = False
            steps = 0
            ep_frames = []
            while not done:
                action, _ = model.predict(obs, deterministic=cfg["eval"]["deterministic"])
                if apply_shield:
                    action, fsm_state = safety_shield(action, predicates(read_scene(env), cfg), fsm_state, cfg)
                obs, _, terminated, truncated, info = env.step(action)
                frame = env.render()
                if frame is not None:
                    ep_frames.append(np.asarray(frame))
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
            print(f"[demo] episode {ep + 1}: {ot} overtakes in {steps} steps"
                  f"{'  (crashed)' if crashed else ''}", flush=True)
            ep += 1
    finally:
        env.close()

    return save_mp4(frames, out_path, fps=fps), total_overtakes, total_steps


def main():
    ap = argparse.ArgumentParser(description="Record a driving video from a saved model.")
    ap.add_argument("--model", required=True,
                    help="checkpoint: SB3 .zip or a bare policy state_dict .pt")
    ap.add_argument("--algo", default="rppo", choices=["rppo", "dqn", "qrdqn"],
                    help="algorithm (default: rppo)")
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
    model = as_predictor(model, args.algo)   # RecurrentPPO gets the stateful LSTM facade
    print(f"recording >= {min_seconds:.0f}s (shield={args.shield}) -> {out_path}")
    path, overtakes, steps = record(model, cfg, out_path, apply_shield=args.shield,
                                    min_seconds=min_seconds, no_crash=args.no_crash)

    size = os.path.getsize(path) if path and os.path.exists(path) else 0
    if size > 0:
        print(f"[demo] overtakes: {overtakes} in {steps} steps "
              f"({100.0 * overtakes / max(1, steps):.1f} per 100 steps)")
        print(f"[demo] OK    {path}  ({size // 1024} KB)")
    else:
        print(f"[demo] FAIL  {out_path}  (no file written or empty)")
        sys.exit(1)


if __name__ == "__main__":
    main()
