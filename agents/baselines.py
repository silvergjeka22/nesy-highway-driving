"""RL trainers: PPO/DQN baselines (Part 1) and the logic-reward fine-tune (Part 2).

Function-only. Each trainer builds one plain env from the config, seeds
everything, trains a stable-baselines3 model on the resolved device (CPU or
GPU), saves the best-by-reward checkpoint, and returns the model. No top-level
execution — the notebooks orchestrate.
"""

import math
import time

import numpy as np

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from utils import set_global_seeds, drive_path, curve_dir
from envs.highway_factory import create_environment

_ALGOS = {"ppo": PPO, "dqn": DQN}


def _monitored_env(cfg, logic_reward=False):
    """One training env, wrapped so each finished episode also records its
    overtake count and crash flag — that is what lets the progress prints and
    the training curves show cars-passed-over-time, not just reward."""
    env = create_environment(cfg, seed=cfg["seed"], logic_reward=logic_reward)
    return Monitor(env, info_keywords=("overtakes", "crashed"))


def resolve_device(cfg):
    """Return 'cuda' or 'cpu' so training works on both GPU and CPU machines.

    ``cfg['device']`` may be ``auto``/``cuda``/``gpu`` (use the GPU when present,
    else fall back to CPU) or ``cpu`` (force CPU).
    """
    want = str(cfg.get("device", "auto")).lower()
    if want == "cpu":
        return "cpu"
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    return "cuda" if has_cuda else "cpu"


class _ProgressPrinter(BaseCallback):
    """Print a training line every ``print_freq`` steps and save the best model.

    Shows the running mean episode reward + length, plus — when the env's Monitor
    records them — the overtaking progress (cars passed per episode and per 100
    steps) and the crash rate, so it is obvious whether the agent is learning to
    pass traffic or just to survive. Whenever the mean reward improves it
    checkpoints the model to ``best_path`` — so the saved checkpoint is the best
    (highest reward) policy seen, not just the final one. The same numbers are
    logged to ``progress.csv`` for the training-curve plots.
    """

    def __init__(self, tag, print_freq=200, best_path=None, total_steps=None):
        super().__init__()
        self.tag = tag
        self.print_freq = max(1, int(print_freq))
        self.best_path = best_path
        self.total_steps = total_steps
        self.best_rew = -float("inf")
        self.saved_best = False
        self._first_done = False
        self._t0 = None
        self._start_step = 0
        self._next = self.print_freq

    def _on_training_start(self):
        self._t0 = time.time()
        self._start_step = self.num_timesteps
        self._next = self.num_timesteps + self.print_freq

    def _speed(self):
        elapsed = max(time.time() - (self._t0 or time.time()), 1e-6)
        done = max(self.num_timesteps - self._start_step, 0)
        sps = done / elapsed
        if self.total_steps:
            eta = max(0.0, (self.total_steps - self.num_timesteps) / max(sps, 1e-6))
            return f" | {sps:4.0f} steps/s | ETA {eta / 60:4.1f} min"
        return f" | {sps:4.0f} steps/s"

    def _on_step(self):
        if not self._first_done and (self.model.ep_info_buffer or []):
            self._first_done = True
            e = list(self.model.ep_info_buffer)[-1]
            print(f"[{self.tag}] step {self.num_timesteps:>7} | first episode done "
                  f"| reward {float(e['r']):.2f} | length {int(e['l'])}{self._speed()}", flush=True)

        if self.num_timesteps >= self._next:
            self._next += self.print_freq
            buf = list(self.model.ep_info_buffer or [])
            if buf:
                r = float(np.mean([e["r"] for e in buf]))
                ln = float(np.mean([e["l"] for e in buf]))
                ot = ""
                if "overtakes" in buf[-1]:   # Monitor(info_keywords) provides these
                    ot_mean = float(np.mean([e["overtakes"] for e in buf]))
                    ot100 = 100.0 * sum(e["overtakes"] for e in buf) / max(1, sum(e["l"] for e in buf))
                    crash = float(np.mean([e["crashed"] for e in buf]))
                    ot = f" | overtakes/ep {ot_mean:4.2f} ({ot100:4.1f}/100 steps) | crash {crash:4.0%}"
                    self.model.logger.record("rollout/ep_overtakes_mean", ot_mean)
                    self.model.logger.record("rollout/ep_crash_rate", crash)
                extra = f" | eps {self.model.exploration_rate:.3f}" if hasattr(self.model, "exploration_rate") else ""
                flag = ""
                if self.best_path is not None and r > self.best_rew:
                    self.best_rew = r
                    self.model.save(self.best_path)
                    self.saved_best = True
                    flag = "  <- new best, saved"
                print(f"[{self.tag}] step {self.num_timesteps:>7} | "
                      f"ep_rew_mean {r:7.2f} | ep_len_mean {ln:6.1f}{ot}"
                      f"{extra}{self._speed()}{flag}", flush=True)
            else:
                print(f"[{self.tag}] step {self.num_timesteps:>7} | "
                      f"collecting first episodes…{self._speed()}", flush=True)
        return True


def _attach_logger(model, cfg, tag):
    """Log training curves to ``metrics/curves/<tag>/progress.csv`` (CSV only).

    SB3 writes ``rollout/ep_rew_mean`` and ``rollout/ep_len_mean`` vs
    ``time/total_timesteps`` there, so ``eval.plots.plot_training_curves`` can plot
    PPO vs DQN. CSV-only keeps the console output to the clean progress lines.
    """
    folder = curve_dir(cfg, tag)
    model.set_logger(configure(folder, ["csv"]))
    return folder


def _effective_total(total_timesteps, rollout_steps):
    """Round a step budget up to a whole rollout so the printed ETA ends at 0.

    SB3 only checks ``total_timesteps`` at rollout boundaries, so a run actually
    stops at the next multiple of the rollout size (``n_steps`` for PPO,
    ``train_freq`` for DQN). Training to that number keeps the ETA honest.
    """
    rollout_steps = max(1, int(rollout_steps))
    return math.ceil(total_timesteps / rollout_steps) * rollout_steps


# =============================================================================
# Part 1 — baselines on highway-env
# =============================================================================
def build_ppo(cfg, env, device=None):
    """Construct the PPO model (on-policy policy-gradient) from the config."""
    p = cfg["ppo"]
    return PPO(
        p["policy"], env,
        learning_rate=p["learning_rate"], n_steps=p["n_steps"],
        batch_size=p["batch_size"], n_epochs=p["n_epochs"],
        gamma=p["gamma"], gae_lambda=p["gae_lambda"], clip_range=p["clip_range"],
        ent_coef=p["ent_coef"], vf_coef=p["vf_coef"], max_grad_norm=p["max_grad_norm"],
        policy_kwargs=p.get("policy_kwargs"), device=device or resolve_device(cfg),
        seed=cfg["seed"], verbose=0,
    )


def build_dqn(cfg, env, device=None):
    """Construct the DQN model (off-policy value-based) from the config."""
    d = cfg["dqn"]
    return DQN(
        d["policy"], env,
        learning_rate=d["learning_rate"], buffer_size=d["buffer_size"],
        learning_starts=d["learning_starts"], batch_size=d["batch_size"],
        gamma=d["gamma"], train_freq=d["train_freq"], gradient_steps=d["gradient_steps"],
        target_update_interval=d["target_update_interval"],
        exploration_fraction=d["exploration_fraction"],
        exploration_final_eps=d["exploration_final_eps"],
        policy_kwargs=d.get("policy_kwargs"), device=device or resolve_device(cfg),
        seed=cfg["seed"], verbose=0,
    )


def train_ppo(cfg, path=None):
    """Train the PPO baseline (recommended) and checkpoint the best to Drive."""
    set_global_seeds(cfg["seed"])
    device = resolve_device(cfg)
    env = _monitored_env(cfg)
    model = build_ppo(cfg, env, device)
    _attach_logger(model, cfg, "ppo")

    path = path or drive_path(cfg, "checkpoints", "ppo.zip")
    pf = cfg.get("print_freq", 200)
    total = _effective_total(cfg["ppo"]["total_timesteps"], cfg["ppo"]["n_steps"])
    print(f"[PPO] training for {total} steps on device='{device}' (printing every {pf} steps)…", flush=True)
    printer = _ProgressPrinter("PPO", pf, best_path=path, total_steps=total)
    model.learn(total_timesteps=total, callback=printer)

    if printer.saved_best:
        print(f"[PPO] done. best ep_rew_mean={printer.best_rew:.2f} -> {path}", flush=True)
        model = PPO.load(path)
    else:
        model.save(path)
        print(f"[PPO] done. saved final model -> {path}", flush=True)
    env.close()
    return model


def train_dqn(cfg, path=None):
    """Train the DQN baseline (second baseline) and checkpoint the best to Drive."""
    set_global_seeds(cfg["seed"])
    device = resolve_device(cfg)
    env = _monitored_env(cfg)
    model = build_dqn(cfg, env, device)
    _attach_logger(model, cfg, "dqn")

    path = path or drive_path(cfg, "checkpoints", "dqn.zip")
    pf = cfg.get("print_freq", 200)
    tf = cfg["dqn"]["train_freq"] if isinstance(cfg["dqn"]["train_freq"], int) else 1
    total = _effective_total(cfg["dqn"]["total_timesteps"], tf)
    print(f"[DQN] training for {total} steps on device='{device}' (printing every {pf} steps)…", flush=True)
    printer = _ProgressPrinter("DQN", pf, best_path=path, total_steps=total)
    model.learn(total_timesteps=total, callback=printer)

    if printer.saved_best:
        print(f"[DQN] done. best ep_rew_mean={printer.best_rew:.2f} -> {path}", flush=True)
        model = DQN.load(path)
    else:
        model.save(path)
        print(f"[DQN] done. saved final model -> {path}", flush=True)
    env.close()
    return model


def load_model(path, algo):
    """Reload a saved checkpoint. ``algo`` is ``'ppo'`` or ``'dqn'``."""
    key = algo.lower()
    if key not in _ALGOS:
        raise ValueError(f"Unknown algo '{algo}'; expected one of {list(_ALGOS)}")
    return _ALGOS[key].load(path)


# =============================================================================
# Part 2 — logic-shaped reward fine-tune (Step C)
# =============================================================================
def _to_device(model, cfg):
    """Move an already-loaded SB3 model to the resolved device (for fine-tuning)."""
    dev = resolve_device(cfg)
    try:
        import torch
        model.device = torch.device(dev)
        model.policy.to(dev)
    except Exception:
        pass
    return model


def finetune_logic_reward(model, cfg, drive_dir=None):
    """Warm-start ``model`` and continue training on the logic-augmented reward.

    The genuine "fine-tune": same policy, a lower learning rate, fewer steps, and
    an env whose reward includes ``- Σ λ_i · violation_i`` (LogicRewardWrapper).
    Returns the fine-tuned model and checkpoints it as ``part2_nesy.zip``.
    """
    set_global_seeds(cfg["seed"])
    ft = cfg["finetune"]
    env = _monitored_env(cfg, logic_reward=True)

    model.set_env(env)
    _to_device(model, cfg)
    ft_lr = ft["learning_rate"]
    model.learning_rate = ft_lr
    model.lr_schedule = lambda _progress_remaining: ft_lr

    _attach_logger(model, cfg, "part2_nesy")
    path = drive_dir or drive_path(cfg, "checkpoints", "part2_nesy.zip")
    pf = cfg.get("print_freq", 200)
    start = int(model.num_timesteps)
    extra = _effective_total(ft["total_timesteps"], getattr(model, "n_steps", 1))
    print(f"[NESY-FT] fine-tuning for {extra} more steps on device='{model.device}' "
          f"(printing every {pf} steps)…", flush=True)
    printer = _ProgressPrinter("NESY-FT", pf, best_path=path, total_steps=start + extra)
    model.learn(total_timesteps=extra, callback=printer, reset_num_timesteps=False)

    if printer.saved_best:
        print(f"[NESY-FT] done. best ep_rew_mean={printer.best_rew:.2f} -> {path}", flush=True)
        model = type(model).load(path)
    else:
        model.save(path)
        print(f"[NESY-FT] done. saved final model -> {path}", flush=True)
    env.close()
    return model
