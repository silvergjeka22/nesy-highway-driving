"""Notebook-facing trainers for the continuous off-policy agents (SAC / MACURA).

``train_offpolicy(cfg, algo)`` runs the paper training loop (warmup → act with the
paper exploration noise → store → update) on the hybrid-action highway env, with
the same live progress numbers as the Part-1 trainers: running overtake total,
overtakes/ep (and per 100 steps), crash rate, episode reward/length — plus the
MACURA-specific κ and imagined-data share. ``as_predictor`` wraps a trained agent
in the SB3 ``.predict`` interface so ``evaluate`` / ``sanity_rollout`` / the demo
work unchanged.
"""
import os
import time

import numpy as np

from utils import set_global_seeds, drive_path, curve_dir
from envs.highway_factory import create_hybrid_environment
from agents.baselines import resolve_device
from agents.offpolicy.sac import SAC
from agents.offpolicy.macura import MACURA

_ALGOS = {"sac": SAC, "macura": MACURA}


def build_offpolicy(cfg, algo, device=None):
    """Construct a SAC or MACURA agent for the hybrid highway env (no training)."""
    op = cfg["offpolicy"]
    # dual stopping needs the traffic-rule parameters + observation layout
    op["rules"] = cfg["rules"]
    op["env_layout"] = {
        "lanes_count": cfg["env"]["config"]["lanes_count"],
        "features_per_vehicle": op["model_based"].get("features_per_vehicle", 5),
    }
    env = create_hybrid_environment(cfg)
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    env.close()
    agent = _ALGOS[algo.lower()](state_dim, action_dim, op, device or resolve_device(cfg))
    return agent


def train_offpolicy(cfg, algo, path=None, tag=None):
    """Train SAC or MACURA on the hybrid-action highway env; returns the agent.

    The loop is the paper's: ``warmup_steps`` random actions seed the buffer, then
    the agent acts with its paper exploration noise (MACURA: pink β=1, SAC: white),
    stores every real transition, and calls ``update_step`` each step (for MACURA
    that includes ensemble retraining + uncertainty-truncated imagined rollouts).

    ``tag`` names the checkpoint and the training-curve folder (defaults to
    ``algo``) — pass e.g. ``tag="macura_tl"`` for the dual-stopping variant so its
    curves/checkpoint don't overwrite plain MACURA's.
    """
    op = cfg["offpolicy"]
    sh = op["shared"]
    set_global_seeds(cfg["seed"])
    device = resolve_device(cfg)
    tag = (tag or algo).lower()
    label = tag.upper()

    env = create_hybrid_environment(cfg, seed=cfg["seed"])
    agent = build_offpolicy(cfg, algo, device)
    path = path or drive_path(cfg, "checkpoints", tag)
    total = int(sh["total_steps"])
    warmup = int(sh["warmup_steps"])
    pf = int(cfg.get("print_freq", 200))

    # Training curve, same columns the PPO/DQN trainers log — so
    # eval.plots.plot_training_curves works on SAC/MACURA tags unchanged.
    csv_path = os.path.join(curve_dir(cfg, tag), "progress.csv")
    with open(csv_path, "w") as f:
        f.write("time/total_timesteps,rollout/ep_rew_mean,rollout/ep_len_mean,"
                "rollout/ep_overtakes_mean,rollout/ep_crash_rate\n")

    print(f"[{label}] training for {total} steps (warmup {warmup} random) on device='{device}' "
          f"| exploration={getattr(agent.policy if hasattr(agent, 'policy') else agent, 'exploration', '?')}",
          flush=True)

    obs, info = env.reset(seed=cfg["seed"])
    agent.reset_noise()
    ep_rew = ep_len = 0.0
    ep_hist, best_rew, t0 = [], -float("inf"), time.time()
    total_overtakes = 0

    for step in range(1, total + 1):
        if step <= warmup:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, explore=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        # time-limit truncation is not a real "done" for bootstrapping
        agent.store(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        ep_rew += float(reward)
        ep_len += 1

        metrics = agent.update_step(step) if step > warmup else {}

        if terminated or truncated:
            total_overtakes += int(info.get("overtakes", 0))
            ep_hist.append({"r": ep_rew, "l": ep_len,
                            "overtakes": int(info.get("overtakes", 0)),
                            "crashed": bool(info.get("crashed", False))})
            ep_hist = ep_hist[-100:]
            obs, info = env.reset()
            agent.reset_noise()
            ep_rew = ep_len = 0.0

        if step % pf == 0 and ep_hist:
            r = float(np.mean([e["r"] for e in ep_hist]))
            ln = float(np.mean([e["l"] for e in ep_hist]))
            ot = float(np.mean([e["overtakes"] for e in ep_hist]))
            ot100 = 100.0 * sum(e["overtakes"] for e in ep_hist) / max(1, sum(e["l"] for e in ep_hist))
            crash = float(np.mean([e["crashed"] for e in ep_hist]))
            with open(csv_path, "a") as f:
                f.write(f"{step},{r:.4f},{ln:.2f},{ot:.4f},{crash:.4f}\n")
            sps = step / max(time.time() - t0, 1e-6)
            extra = ""
            if "kappa" in metrics:
                extra = (f" | κ {metrics['kappa']:.3f} | imag {metrics.get('imagined_pct', 0):.0f}%"
                         f" | depth {metrics.get('rollout_depth_mean', 0):.1f}"
                         f" | term {metrics.get('model_term_frac', 0):.1%}")
                if "tl_cut_frac" in metrics:
                    extra += f" | TLcut {metrics['tl_cut_frac']:.1%}"
            flag = ""
            if r > best_rew and step > warmup:
                best_rew = r
                agent.save(path)
                flag = "  <- new best, saved"
            print(f"[{label}] step {step:>6} | ep_rew_mean {r:7.2f} | ep_len_mean {ln:5.1f} | "
                  f"overtakes {total_overtakes} total, {ot:4.2f}/ep ({ot100:4.1f}/100 steps) | "
                  f"crash {crash:4.0%}{extra} | {sps:4.0f} steps/s"
                  f" | ETA {(total - step) / max(sps, 1e-6) / 60:4.1f} min{flag}", flush=True)

    if best_rew == -float("inf"):
        agent.save(path)
    print(f"[{label}] done. best ep_rew_mean={best_rew:.2f} -> {path}_policy.pt", flush=True)
    env.close()
    return agent


class as_predictor:
    """SB3-compatible ``.predict`` facade over a SAC/MACURA agent, so the shared
    ``evaluate`` / ``sanity_rollout`` harness works on them unchanged."""

    def __init__(self, agent):
        self.agent = agent

    def predict(self, obs, deterministic=True):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        return self.agent.act(obs, explore=not deterministic), None
