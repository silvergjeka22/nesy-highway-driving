"""Model-free RL baselines for Part 1: PPO (recommended) and DQN.

Function-only module. Each trainer builds a fresh env from the same config,
seeds everything, trains a stable-baselines3 model, saves a checkpoint to the
Drive results folder, and returns the in-memory model. ``load_model`` reloads a
saved checkpoint for evaluation.

No top-level execution — the Colab notebook orchestrates these calls.
"""

from stable_baselines3 import PPO, DQN

from utils import set_global_seeds, drive_path
from envs.highway_factory import make_env

# Algorithm registry so callers can stay generic (load_model, eval harness).
_ALGOS = {"ppo": PPO, "dqn": DQN}


def train_ppo(cfg, drive_dir=None, fast=False):
    """Train the PPO baseline and checkpoint it.

    Args:
        cfg: full project config dict.
        drive_dir: optional override for the checkpoint path; defaults to
            ``<drive_root>/checkpoints/ppo.zip``.
        fast: use ``highway-fast-v0`` for a quick smoke run.

    Returns:
        The trained PPO model.
    """
    set_global_seeds(cfg["seed"])
    p = cfg["ppo"]
    env = make_env(cfg, render=False, seed=cfg["seed"], fast=fast)

    model = PPO(
        p["policy"],
        env,
        learning_rate=p["learning_rate"],
        n_steps=p["n_steps"],
        batch_size=p["batch_size"],
        n_epochs=p["n_epochs"],
        gamma=p["gamma"],
        gae_lambda=p["gae_lambda"],
        clip_range=p["clip_range"],
        ent_coef=p["ent_coef"],
        vf_coef=p["vf_coef"],
        max_grad_norm=p["max_grad_norm"],
        policy_kwargs=p.get("policy_kwargs"),
        tensorboard_log=_tb_dir(cfg),
        seed=cfg["seed"],
        verbose=p.get("verbose", 1),
    )
    model.learn(total_timesteps=p["total_timesteps"], progress_bar=True)

    path = drive_dir or drive_path(cfg, "checkpoints", "ppo.zip")
    model.save(path)
    env.close()
    return model


def train_dqn(cfg, drive_dir=None, fast=False):
    """Train the DQN baseline and checkpoint it.

    Same env/observation/eval-seed discipline as ``train_ppo`` for a fair
    head-to-head comparison.
    """
    set_global_seeds(cfg["seed"])
    d = cfg["dqn"]
    env = make_env(cfg, render=False, seed=cfg["seed"], fast=fast)

    model = DQN(
        d["policy"],
        env,
        learning_rate=d["learning_rate"],
        buffer_size=d["buffer_size"],
        learning_starts=d["learning_starts"],
        batch_size=d["batch_size"],
        gamma=d["gamma"],
        train_freq=d["train_freq"],
        gradient_steps=d["gradient_steps"],
        target_update_interval=d["target_update_interval"],
        exploration_fraction=d["exploration_fraction"],
        exploration_final_eps=d["exploration_final_eps"],
        policy_kwargs=d.get("policy_kwargs"),
        tensorboard_log=_tb_dir(cfg),
        seed=cfg["seed"],
        verbose=d.get("verbose", 1),
    )
    model.learn(total_timesteps=d["total_timesteps"], progress_bar=True)

    path = drive_dir or drive_path(cfg, "checkpoints", "dqn.zip")
    model.save(path)
    env.close()
    return model


def load_model(path, algo):
    """Reload a saved checkpoint.

    Args:
        path: path to the ``.zip`` checkpoint.
        algo: ``"ppo"`` or ``"dqn"``.

    Returns:
        The loaded SB3 model (no env attached; the eval harness supplies one).
    """
    key = algo.lower()
    if key not in _ALGOS:
        raise ValueError(f"Unknown algo '{algo}'; expected one of {list(_ALGOS)}")
    return _ALGOS[key].load(path)


def _tb_dir(cfg):
    """TensorBoard log dir under Drive, or None if not configured."""
    try:
        return drive_path(cfg, "tensorboard")
    except KeyError:
        return None
