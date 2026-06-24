"""RL trainers: PPO/DQN baselines (Part 1), logic-reward fine-tune (Part 2),
and continuous PPO on MetaDrive (Part 3).

Function-only. Each trainer builds a fresh env from the same config, seeds
everything, trains a stable-baselines3 model, saves a checkpoint to Drive, and
returns the model. No top-level execution — the notebooks orchestrate.
"""

# Mute legacy-gym / pkg_resources warnings before SB3 imports its compat shim.
from utils import silence_warnings, set_global_seeds, drive_path

silence_warnings()

from stable_baselines3 import PPO, DQN  # noqa: E402

from envs.highway_factory import make_env  # noqa: E402

_ALGOS = {"ppo": PPO, "dqn": DQN}


# =============================================================================
# Part 1 — baselines on highway-env
# =============================================================================
def train_ppo(cfg, drive_dir=None, fast=False):
    """Train the PPO baseline (recommended) and checkpoint it to Drive."""
    set_global_seeds(cfg["seed"])
    p = cfg["ppo"]
    env = make_env(cfg, render=False, seed=cfg["seed"], fast=fast)

    model = PPO(
        p["policy"], env,
        learning_rate=p["learning_rate"], n_steps=p["n_steps"],
        batch_size=p["batch_size"], n_epochs=p["n_epochs"],
        gamma=p["gamma"], gae_lambda=p["gae_lambda"], clip_range=p["clip_range"],
        ent_coef=p["ent_coef"], vf_coef=p["vf_coef"], max_grad_norm=p["max_grad_norm"],
        policy_kwargs=p.get("policy_kwargs"), tensorboard_log=_tb_dir(cfg),
        seed=cfg["seed"], verbose=p.get("verbose", 1),
    )
    model.learn(total_timesteps=p["total_timesteps"], progress_bar=True)

    path = drive_dir or drive_path(cfg, "checkpoints", "ppo.zip")
    model.save(path)
    env.close()
    return model


def train_dqn(cfg, drive_dir=None, fast=False):
    """Train the DQN baseline (second required baseline) and checkpoint it."""
    set_global_seeds(cfg["seed"])
    d = cfg["dqn"]
    env = make_env(cfg, render=False, seed=cfg["seed"], fast=fast)

    model = DQN(
        d["policy"], env,
        learning_rate=d["learning_rate"], buffer_size=d["buffer_size"],
        learning_starts=d["learning_starts"], batch_size=d["batch_size"],
        gamma=d["gamma"], train_freq=d["train_freq"], gradient_steps=d["gradient_steps"],
        target_update_interval=d["target_update_interval"],
        exploration_fraction=d["exploration_fraction"],
        exploration_final_eps=d["exploration_final_eps"],
        policy_kwargs=d.get("policy_kwargs"), tensorboard_log=_tb_dir(cfg),
        seed=cfg["seed"], verbose=d.get("verbose", 1),
    )
    model.learn(total_timesteps=d["total_timesteps"], progress_bar=True)

    path = drive_dir or drive_path(cfg, "checkpoints", "dqn.zip")
    model.save(path)
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
    # Lower, constant learning rate for the fine-tune.
    ft_lr = ft["learning_rate"]
    model.learning_rate = ft_lr
    model.lr_schedule = lambda _progress_remaining: ft_lr

    model.learn(total_timesteps=ft["total_timesteps"], progress_bar=True,
                reset_num_timesteps=False)

    path = drive_dir or drive_path(cfg, "checkpoints", "part2_nesy.zip")
    model.save(path)
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

    model = PPO(
        p["policy"], env,
        learning_rate=p["learning_rate"], n_steps=p["n_steps"],
        batch_size=p["batch_size"], n_epochs=p["n_epochs"],
        gamma=p["gamma"], gae_lambda=p["gae_lambda"], clip_range=p["clip_range"],
        ent_coef=p.get("ent_coef", 0.0), policy_kwargs=p.get("policy_kwargs"),
        tensorboard_log=_tb_dir(cfg), seed=cfg["seed"], verbose=p.get("verbose", 1),
    )
    model.learn(total_timesteps=p["total_timesteps"], progress_bar=True)

    path = drive_dir or drive_path(cfg, "checkpoints", "part3_metadrive.zip")
    model.save(path)
    env.close()
    return model


def _tb_dir(cfg):
    try:
        return drive_path(cfg, "tensorboard")
    except KeyError:
        return None
