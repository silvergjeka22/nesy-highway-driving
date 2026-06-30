"""RL trainers: PPO/DQN baselines (Part 1), logic-reward fine-tune (Part 2),
and continuous PPO on MetaDrive (Part 3).

Function-only. Each trainer builds a fresh env from the same config, seeds
everything, trains a stable-baselines3 model, saves a checkpoint to Drive, and
returns the model. No top-level execution — the notebooks orchestrate.
"""

# Mute legacy-gym / pkg_resources warnings before SB3 imports its compat shim.
from utils import silence_warnings, set_global_seeds, drive_path, curve_dir

silence_warnings()

import numpy as np  # noqa: E402

from stable_baselines3 import PPO, DQN  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from envs.highway_factory import make_env  # noqa: E402

_ALGOS = {"ppo": PPO, "dqn": DQN}


class _ProgressPrinter(BaseCallback):
    """Print a training line every ``print_freq`` steps AND save the best model.

    Shows the running mean episode reward + length (from SB3's Monitor buffer) so
    you can watch learning progress, and whenever the mean reward improves it
    checkpoints the model to ``best_path`` — so the saved checkpoint is the
    **best (highest-reward)** policy seen, not just the final one.
    """

    def __init__(self, tag, print_freq=500, best_path=None):
        super().__init__()
        self.tag = tag
        self.print_freq = max(1, int(print_freq))
        self.best_path = best_path
        self.best_rew = -float("inf")
        self.saved_best = False
        self._next = self.print_freq

    def _on_step(self):
        if self.num_timesteps >= self._next:
            self._next += self.print_freq
            buf = list(self.model.ep_info_buffer or [])
            if buf:
                r = float(np.mean([e["r"] for e in buf]))
                ln = float(np.mean([e["l"] for e in buf]))
                extra = ""
                if hasattr(self.model, "exploration_rate"):  # DQN
                    extra = f" | eps {self.model.exploration_rate:.3f}"
                flag = ""
                if self.best_path is not None and r > self.best_rew:
                    self.best_rew = r
                    self.model.save(self.best_path)
                    self.saved_best = True
                    flag = "  <- new best, saved"
                print(f"[{self.tag}] step {self.num_timesteps:>7} | "
                      f"ep_rew_mean {r:7.2f} | ep_len_mean {ln:6.1f} | "
                      f"episodes {len(buf)}{extra}{flag}", flush=True)
            else:
                print(f"[{self.tag}] step {self.num_timesteps:>7} | "
                      f"collecting first episodes…", flush=True)
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


# =============================================================================
# Part 1 — baselines on highway-env
# =============================================================================
def train_ppo(cfg, drive_dir=None, fast=False):
    """Train the PPO baseline (recommended) and checkpoint it to Drive."""
    set_global_seeds(cfg["seed"])
    p = cfg["ppo"]
    env = make_env(cfg, render=False, seed=cfg["seed"], fast=fast)

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
    _attach_logger(model, cfg, "ppo")
    path = drive_dir or drive_path(cfg, "checkpoints", "ppo.zip")
    pf = p.get("print_freq", cfg.get("print_freq", 500))
    print(f"[PPO] training for {p['total_timesteps']} steps on device='{device}' "
          f"(printing every {pf} steps)…", flush=True)
    printer = _ProgressPrinter("PPO", pf, best_path=path)
    model.learn(total_timesteps=p["total_timesteps"], callback=printer)

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
    env = make_env(cfg, render=False, seed=cfg["seed"], fast=fast)

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
    print(f"[DQN] training for {d['total_timesteps']} steps on device='{device}' "
          f"(printing every {pf} steps)…", flush=True)
    printer = _ProgressPrinter("DQN", pf, best_path=path)
    model.learn(total_timesteps=d["total_timesteps"], callback=printer)

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


def select_best(candidates, cfg):
    """Pick the best Part-1 model by the explicit YAML rule.

    Args:
        candidates: ``{tag: {"model": sb3_model, "metrics": evaluate(...)}}`` —
            e.g. ``{"ppo": {...}, "dqn": {...}}``.
        cfg: full config; uses the ``select:`` block.

    Returns:
        ``(best_model, tag)``. Default rule (``min_crash_rate``): among models
        whose mean return is within ``within_return_pct`` of the top, choose the
        lowest crash rate (tie-broken by higher return) — safety-first.
    """
    sc = cfg.get("select", {})
    within = sc.get("within_return_pct", 0.10)
    items = [(tag, c["model"], c["metrics"]["summary"]) for tag, c in candidates.items()]
    if not items:
        raise ValueError("select_best: no candidates given")

    top_return = max(s["return"]["mean"] for _, _, s in items)
    thresh = top_return - abs(top_return) * within
    eligible = [(t, m, s) for (t, m, s) in items if s["return"]["mean"] >= thresh] or items

    best = min(eligible, key=lambda x: (x[2]["crash_rate"], -x[2]["return"]["mean"]))
    return best[1], best[0]


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
    print(f"[NESY-FT] fine-tuning for {ft['total_timesteps']} steps on "
          f"device='{model.device}' (printing every {pf} steps)…", flush=True)
    printer = _ProgressPrinter("NESY-FT", pf, best_path=path)
    model.learn(total_timesteps=ft["total_timesteps"], callback=printer,
                reset_num_timesteps=False)

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
    print(f"[MD-PPO] training for {p['total_timesteps']} steps on device='{device}' "
          f"(printing every {pf} steps)…", flush=True)
    printer = _ProgressPrinter("MD-PPO", pf, best_path=path)
    model.learn(total_timesteps=p["total_timesteps"], callback=printer)

    if printer.saved_best:
        print(f"[MD-PPO] done. best ep_rew_mean={printer.best_rew:.2f} -> {path}", flush=True)
        model = PPO.load(path)
    else:
        model.save(path)
        print(f"[MD-PPO] done. saved final model -> {path}", flush=True)
    env.close()
    return model
