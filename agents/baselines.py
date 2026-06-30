"""RL trainers: PPO/DQN baselines (Part 1), logic-reward fine-tune (Part 2),
and continuous PPO on MetaDrive (Part 3).

Function-only. Each trainer builds a fresh env from the same config, seeds
everything, trains a stable-baselines3 model, saves a checkpoint to Drive, and
returns the model. No top-level execution — the notebooks orchestrate.
"""

# Mute legacy-gym / pkg_resources warnings before SB3 imports its compat shim.
from utils import silence_warnings, set_global_seeds, drive_path, curve_dir

silence_warnings()

import os  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from stable_baselines3 import PPO, DQN  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv  # noqa: E402

from envs.highway_factory import make_env  # noqa: E402

_ALGOS = {"ppo": PPO, "dqn": DQN}


def _train_env(cfg, fast=False, logic_reward=False, seed=None):
    """Build the training env: ``cfg['n_envs']`` parallel highway-envs in a
    ``SubprocVecEnv`` (the model-free speedup — many envs step at once across CPU
    cores) when ``n_envs > 1``, else a single env. Eval and video stay single-env.

    Falls back to ``DummyVecEnv`` (sequential, same API) if subprocesses can't
    start on the platform. Each sub-env is Monitor-wrapped by ``make_vec_env`` so
    ``ep_rew_mean`` / curves still work.
    """
    n_envs = int(cfg.get("n_envs", 1))
    if n_envs <= 1:
        return make_env(cfg, render=False, fast=fast, seed=seed, logic_reward=logic_reward)

    def _factory():
        return make_env(cfg, render=False, fast=fast, logic_reward=logic_reward)

    try:
        return make_vec_env(_factory, n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv)
    except Exception as e:  # pragma: no cover - platform dependent
        print(f"  (SubprocVecEnv unavailable: {e}; using DummyVecEnv)", flush=True)
        return make_vec_env(_factory, n_envs=n_envs, seed=seed, vec_env_cls=DummyVecEnv)


class _ProgressPrinter(BaseCallback):
    """Print a training line every ``print_freq`` steps AND save the best model.

    Shows the running mean episode reward + length (from SB3's Monitor buffer) so
    you can watch learning progress, and whenever the mean reward improves it
    checkpoints the model to ``best_path`` — so the saved checkpoint is the
    **best (highest-reward)** policy seen, not just the final one.
    """

    def __init__(self, tag, print_freq=500, best_path=None, total_steps=None, curve_csv=None):
        super().__init__()
        self.tag = tag
        self.print_freq = max(1, int(print_freq))
        self.best_path = best_path
        self.total_steps = total_steps
        self.curve_csv = curve_csv
        self.best_rew = -float("inf")
        self.saved_best = False
        self._first_done = False
        self._curve_init = False
        self._t0 = None
        self._start_step = 0
        self._next = self.print_freq

    def _on_training_start(self):
        # Anchor progress to THIS call's starting step so the rate, ETA and print
        # cadence are correct even for a warm-started fine-tune (where
        # num_timesteps already carries the Part-1 step count).
        self._t0 = time.time()
        self._start_step = self.num_timesteps
        self._next = self.num_timesteps + self.print_freq

    def _log_curve(self, step, r, ln):
        """Append one fine-grained curve point (every print_freq) for plotting."""
        if not self.curve_csv:
            return
        mode = "w" if not self._curve_init else "a"
        with open(self.curve_csv, mode) as f:
            if not self._curve_init:
                f.write("step,ep_rew_mean,ep_len_mean\n")
                self._curve_init = True
            f.write(f"{step},{r:.4f},{ln:.4f}\n")

    def _speed(self):
        elapsed = max(time.time() - (self._t0 or time.time()), 1e-6)
        done = max(self.num_timesteps - self._start_step, 0)   # steps in THIS session
        sps = done / elapsed
        if self.total_steps:
            eta = max(0.0, (self.total_steps - self.num_timesteps) / max(sps, 1e-6))
            return f" | {sps:4.0f} steps/s | ETA {eta/60:4.1f} min"
        return f" | {sps:4.0f} steps/s"

    def _on_step(self):
        # Early signal: print as soon as the very first episode completes, so you
        # see training is live well before the first `print_freq` checkpoint.
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
                extra = ""
                if hasattr(self.model, "exploration_rate"):  # DQN
                    extra = f" | eps {self.model.exploration_rate:.3f}"
                self._log_curve(self.num_timesteps, r, ln)
                flag = ""
                if self.best_path is not None and r > self.best_rew:
                    self.best_rew = r
                    self.model.save(self.best_path)
                    self.saved_best = True
                    flag = "  <- new best, saved"
                print(f"[{self.tag}] step {self.num_timesteps:>7} | "
                      f"ep_rew_mean {r:7.2f} | ep_len_mean {ln:6.1f} | "
                      f"episodes {len(buf)}{extra}{self._speed()}{flag}", flush=True)
            else:
                print(f"[{self.tag}] step {self.num_timesteps:>7} | "
                      f"collecting first episodes…{self._speed()}", flush=True)
        return True


def _resolve_device(cfg):
    """Resolve the requested device, falling back to CPU if CUDA is absent.

    ``cfg['device']`` may be ``cuda`` (use the GPU — e.g. Colab T4), ``cpu``, or
    ``auto`` (let SB3 decide). On free Colab with a T4, ``cuda`` puts the policy
    on the GPU.
    """
    want = str(cfg.get("device", "auto")).lower()
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    if want in ("cuda", "gpu"):
        return "cuda" if has_cuda else "cpu"
    return want


def _to_device(model, cfg):
    """Move an already-loaded SB3 model to the resolved device (for fine-tuning)."""
    dev = _resolve_device(cfg)
    if dev == "auto":
        return model
    try:
        import torch
        model.device = torch.device(dev)
        model.policy.to(dev)
    except Exception:
        pass
    return model


def _attach_logger(model, cfg, tag):
    """Log training curves to ``metrics/curves/<tag>/`` (CSV + TensorBoard).

    SB3 writes ``progress.csv`` with ``rollout/ep_rew_mean`` and
    ``rollout/ep_len_mean`` vs ``time/total_timesteps`` there, so the notebooks
    can plot PPO-vs-DQN training curves with ``eval.plots.plot_training_curves``.
    """
    folder = curve_dir(cfg, tag)
    model.set_logger(configure(folder, ["stdout", "csv", "tensorboard"]))
    return folder


def _effective_total(total_timesteps, rollout_steps):
    """Round a step budget up to a whole number of rollouts.

    SB3 only checks ``total_timesteps`` at rollout boundaries, so a run actually
    stops at the next multiple of the rollout size — ``n_steps * n_envs`` for PPO,
    ``train_freq * n_envs`` for DQN. Training to *that* number (and reporting it)
    is what makes the printed ETA reach 0 exactly when training ends, instead of
    hitting 0 early and overshooting (e.g. 1000 steps with 8 envs × 512 actually
    runs 4096). Same actual run length either way; only the reported target moves.
    """
    rollout_steps = max(1, int(rollout_steps))
    return math.ceil(total_timesteps / rollout_steps) * rollout_steps


# =============================================================================
# Part 1 — baselines on highway-env
# =============================================================================
def train_ppo(cfg, drive_dir=None, fast=False):
    """Train the PPO baseline (recommended) and checkpoint it to Drive."""
    set_global_seeds(cfg["seed"])
    p = cfg["ppo"]
    env = _train_env(cfg, fast=fast, seed=cfg["seed"])

    device = _resolve_device(cfg)

    model = PPO(
        p["policy"], env,
        learning_rate=p["learning_rate"], n_steps=p["n_steps"],
        batch_size=p["batch_size"], n_epochs=p["n_epochs"],
        gamma=p["gamma"], gae_lambda=p["gae_lambda"], clip_range=p["clip_range"],
        ent_coef=p["ent_coef"], vf_coef=p["vf_coef"], max_grad_norm=p["max_grad_norm"],
        policy_kwargs=p.get("policy_kwargs"), device=device,
        seed=cfg["seed"], verbose=p.get("verbose", 1),
    )
    print("model ready")
    _attach_logger(model, cfg, "ppo")
    path = drive_dir or drive_path(cfg, "checkpoints", "ppo.zip")
    pf = p.get("print_freq", cfg.get("print_freq", 500))
    n_envs = max(1, int(cfg.get("n_envs", 1)))
    # PPO collects n_steps per env before each update, so the run rounds up to the
    # next multiple of n_steps*n_envs — train to that so the ETA ends at 0.
    total = _effective_total(p["total_timesteps"], p["n_steps"] * n_envs)
    rounded = f" (rounded up from {p['total_timesteps']})" if total != p["total_timesteps"] else ""
    print(f"[PPO] training for {total} steps{rounded} on device='{device}' "
          f"with n_envs={n_envs} (printing every {pf} steps)…", flush=True)
    printer = _ProgressPrinter("PPO", pf, best_path=path, total_steps=total,
                               curve_csv=os.path.join(curve_dir(cfg, "ppo"), "curve.csv"))
    model.learn(total_timesteps=total, callback=printer)

    if printer.saved_best:
        print(f"[PPO] done. best ep_rew_mean={printer.best_rew:.2f} -> {path}", flush=True)
        model = PPO.load(path)
    else:
        model.save(path)
        print(f"[PPO] done. saved final model -> {path}", flush=True)
    env.close()
    return model


def train_dqn(cfg, drive_dir=None, fast=False):
    """Train the DQN baseline (second required baseline) and checkpoint it."""
    set_global_seeds(cfg["seed"])
    d = cfg["dqn"]
    env = _train_env(cfg, fast=fast, seed=cfg["seed"])

    device = _resolve_device(cfg)
    model = DQN(
        d["policy"], env,
        learning_rate=d["learning_rate"], buffer_size=d["buffer_size"],
        learning_starts=d["learning_starts"], batch_size=d["batch_size"],
        gamma=d["gamma"], train_freq=d["train_freq"], gradient_steps=d["gradient_steps"],
        target_update_interval=d["target_update_interval"],
        exploration_fraction=d["exploration_fraction"],
        exploration_final_eps=d["exploration_final_eps"],
        policy_kwargs=d.get("policy_kwargs"), device=device,
        seed=cfg["seed"], verbose=d.get("verbose", 1),
    )
    _attach_logger(model, cfg, "dqn")
    path = drive_dir or drive_path(cfg, "checkpoints", "dqn.zip")
    pf = d.get("print_freq", cfg.get("print_freq", 500))
    n_envs = max(1, int(cfg.get("n_envs", 1)))
    # DQN collects train_freq steps per env before each gradient update.
    tf = d["train_freq"] if isinstance(d["train_freq"], int) else 1
    total = _effective_total(d["total_timesteps"], tf * n_envs)
    rounded = f" (rounded up from {d['total_timesteps']})" if total != d["total_timesteps"] else ""
    print(f"[DQN] training for {total} steps{rounded} on device='{device}' "
          f"with n_envs={n_envs} (printing every {pf} steps)…", flush=True)
    printer = _ProgressPrinter("DQN", pf, best_path=path, total_steps=total,
                               curve_csv=os.path.join(curve_dir(cfg, "dqn"), "curve.csv"))
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
def finetune_logic_reward(model, cfg, drive_dir=None, fast=False):
    """Warm-start ``model`` and continue training on the logic-augmented reward.

    The genuine "fine-tune": same policy, a lower learning rate, fewer steps, and
    an env whose reward includes ``- Σ λ_i · violation_i`` (LogicRewardWrapper).
    Returns the fine-tuned model and checkpoints it as ``ppo_nesy.zip``.
    """
    set_global_seeds(cfg["seed"])
    ft = cfg["finetune"]
    env = make_env(cfg, render=False, seed=cfg["seed"], fast=fast, logic_reward=True)

    model.set_env(env)
    _to_device(model, cfg)
    # Lower, constant learning rate for the fine-tune.
    ft_lr = ft["learning_rate"]
    model.learning_rate = ft_lr
    model.lr_schedule = lambda _progress_remaining: ft_lr

    _attach_logger(model, cfg, "part2_nesy")
    path = drive_dir or drive_path(cfg, "checkpoints", "part2_nesy.zip")
    pf = ft.get("print_freq", cfg.get("print_freq", 500))
    # Warm-start: training continues from the Part-1 step count (reset_num_timesteps
    # =False). Round the extra budget to a whole rollout and target start+extra so
    # the ETA counts down over THIS fine-tune, not the absolute timeline.
    start = int(model.num_timesteps)
    extra = _effective_total(ft["total_timesteps"], getattr(model, "n_steps", 1))
    print(f"[NESY-FT] fine-tuning for {extra} more steps on "
          f"device='{model.device}' (printing every {pf} steps)…", flush=True)
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


# =============================================================================
# Part 3 — continuous PPO on MetaDrive (velocity action)
# =============================================================================
def train_ppo_md(cfg, drive_dir=None):
    """Train PPO with a continuous ``(v, ω)`` head on MetaDrive."""
    from envs.metadrive_factory import make_env_md

    set_global_seeds(cfg["seed"])
    p = cfg["metadrive"]["ppo"]
    env = make_env_md(cfg, render=False, seed=cfg["seed"])

    device = _resolve_device(cfg)
    model = PPO(
        p["policy"], env,
        learning_rate=p["learning_rate"], n_steps=p["n_steps"],
        batch_size=p["batch_size"], n_epochs=p["n_epochs"],
        gamma=p["gamma"], gae_lambda=p["gae_lambda"], clip_range=p["clip_range"],
        ent_coef=p.get("ent_coef", 0.0), policy_kwargs=p.get("policy_kwargs"),
        device=device, seed=cfg["seed"], verbose=p.get("verbose", 1),
    )
    _attach_logger(model, cfg, "part3_metadrive")
    path = drive_dir or drive_path(cfg, "checkpoints", "part3_metadrive.zip")
    pf = p.get("print_freq", cfg.get("print_freq", 500))
    # Single MetaDrive env, so the rollout is n_steps; round the budget up to it.
    total = _effective_total(p["total_timesteps"], p["n_steps"])
    print(f"[MD-PPO] training for {total} steps on device='{device}' "
          f"(printing every {pf} steps)…", flush=True)
    printer = _ProgressPrinter("MD-PPO", pf, best_path=path, total_steps=total)
    model.learn(total_timesteps=total, callback=printer)

    if printer.saved_best:
        print(f"[MD-PPO] done. best ep_rew_mean={printer.best_rew:.2f} -> {path}", flush=True)
        model = PPO.load(path)
    else:
        model.save(path)
        print(f"[MD-PPO] done. saved final model -> {path}", flush=True)
    env.close()
    return model
