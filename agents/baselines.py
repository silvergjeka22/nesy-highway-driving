import math
import time

import numpy as np

from stable_baselines3 import DQN, PPO
from sb3_contrib import QRDQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from utils import set_global_seeds, drive_path, curve_dir
from envs.highway_factory import create_environment

ALGOS = {"ppo": PPO, "dqn": DQN, "qrdqn": QRDQN}


def monitored_env(cfg, logic_reward=False):
    """Create a training env wrapped in SB3's Monitor for episode stats."""
    env = create_environment(cfg, seed=cfg["seed"], logic_reward=logic_reward)
    return Monitor(env, info_keywords=("overtakes", "crashed", "lane_changes"))


def resolve_device(cfg):
    """Return 'cuda' if available and requested, else 'cpu'."""
    want = str(cfg.get("device", "auto")).lower()
    if want == "cpu":
        return "cpu"
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    return "cuda" if has_cuda else "cpu"


class EntropyDecay(BaseCallback):
    """Linearly decay PPO's entropy coefficient from start to end over training."""

    def __init__(self, start, end, total_steps):
        super().__init__()
        self.start = float(start)
        self.end = float(end)
        self.total = max(1, int(total_steps))

    def _on_step(self):
        frac = min(1.0, self.num_timesteps / self.total)
        self.model.ent_coef = self.start + frac * (self.end - self.start)
        return True


class ProgressPrinter(BaseCallback):
    """Print training progress: reward, episode length, overtakes, crash rate.
    Also saves the best checkpoint by reward and logs metrics to CSV."""

    def __init__(self, tag, print_freq=200, best_path=None, total_steps=None):
        super().__init__()
        self.tag = tag
        self.print_freq = max(1, int(print_freq))
        self.best_path = best_path
        self.total_steps = total_steps
        self.best_rew = -float("inf")
        self.saved_best = False
        self.total_overtakes = 0
        self.action_counts = None
        self.first_done = False
        self.t0 = None
        self.start_step = 0
        self.next_print = self.print_freq

    def _on_training_start(self):
        self.t0 = time.time()
        self.start_step = self.num_timesteps
        self.next_print = self.num_timesteps + self.print_freq
        n = getattr(self.training_env.action_space, "n", 0)
        self.action_counts = np.zeros(int(n), dtype=np.int64) if n else None

    def _on_step(self):
        if self.action_counts is not None:
            acts = self.locals.get("actions")
            if acts is not None:
                for a in np.asarray(acts).ravel().astype(int):
                    if 0 <= a < len(self.action_counts):
                        self.action_counts[a] += 1

        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.total_overtakes += int(ep.get("overtakes", 0))

        if self.num_timesteps >= self.next_print:
            self.next_print += self.print_freq
            buf = list(self.model.ep_info_buffer or [])
            if buf:
                r = float(np.mean([e["r"] for e in buf]))
                ln = float(np.mean([e["l"] for e in buf]))

                ot_str = ""
                if "overtakes" in buf[-1]:
                    ot_mean = float(np.mean([e["overtakes"] for e in buf]))
                    crash = float(np.mean([e["crashed"] for e in buf]))
                    lc_mean = float(np.mean([e.get("lane_changes", 0) for e in buf]))
                    ot_str = f" | overtakes {ot_mean:.2f}/ep | crash {crash:.0%}"

                    self.model.logger.record("rollout/ep_rew_mean", r)
                    self.model.logger.record("rollout/ep_len_mean", ln)
                    self.model.logger.record("rollout/ep_overtakes_mean", ot_mean)
                    self.model.logger.record("rollout/ep_lane_changes_mean", lc_mean)
                    self.model.logger.record("rollout/ep_crash_rate", crash)
                    self.model.logger.record("time/total_timesteps", self.num_timesteps)
                    self.model.logger.dump(self.num_timesteps)

                flag = ""
                if self.best_path is not None and r > self.best_rew:
                    self.best_rew = r
                    self.model.save(self.best_path)
                    self.saved_best = True
                    flag = "  <- best"

                print(f"[{self.tag}] step {self.num_timesteps:>7} | "
                      f"reward {r:7.2f} | len {ln:5.1f}{ot_str}{flag}", flush=True)
        return True


def attach_logger(model, cfg, tag):
    """Set up CSV logging for training curves."""
    folder = curve_dir(cfg, tag)
    model.set_logger(configure(folder, ["csv"]))
    return folder


def effective_total(total_timesteps, rollout_steps):
    """Round total_timesteps up to a multiple of rollout_steps."""
    rollout_steps = max(1, int(rollout_steps))
    return math.ceil(total_timesteps / rollout_steps) * rollout_steps


# ---- Build models -----------------------------------------------------------

def build_ppo(cfg, env, device=None):
    """Build a PPO model from config (no training)."""
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


def build_q_learner(cls, q, cfg, env, device):
    """Build a DQN or QR-DQN model from config (shared construction)."""
    return cls(
        q["policy"], env,
        learning_rate=q["learning_rate"], buffer_size=q["buffer_size"],
        learning_starts=q["learning_starts"], batch_size=q["batch_size"],
        gamma=q["gamma"], train_freq=q["train_freq"], gradient_steps=q["gradient_steps"],
        target_update_interval=q["target_update_interval"],
        exploration_fraction=q["exploration_fraction"],
        exploration_final_eps=q["exploration_final_eps"],
        policy_kwargs=q.get("policy_kwargs"), device=device or resolve_device(cfg),
        seed=cfg["seed"], verbose=0,
    )


def build_dqn(cfg, env, device=None):
    return build_q_learner(DQN, cfg["dqn"], cfg, env, device)


def build_qrdqn(cfg, env, device=None):
    return build_q_learner(QRDQN, cfg["qrdqn"], cfg, env, device)


# ---- Training ---------------------------------------------------------------

def train_model(cfg, tag, model, env, rollout_steps, path=None, extra_callbacks=()):
    """Train a model, save the best checkpoint by reward, and log to CSV."""
    attach_logger(model, cfg, tag)
    path = path or drive_path(cfg, "checkpoints", f"{tag}.zip")
    pf = cfg.get("print_freq", 200)
    total = effective_total(cfg["train"]["total_timesteps"], rollout_steps)
    label = tag.upper()
    print(f"[{label}] training {total} steps on {model.device}", flush=True)
    printer = ProgressPrinter(label, pf, best_path=path, total_steps=total)
    model.learn(total_timesteps=total, callback=[printer, *extra_callbacks])

    if printer.saved_best:
        print(f"[{label}] done, best reward={printer.best_rew:.2f} -> {path}", flush=True)
        model = type(model).load(path)
    else:
        model.save(path)
        print(f"[{label}] done -> {path}", flush=True)
    env.close()
    return model


def train_ppo(cfg, path=None):
    """Train PPO with entropy decay. Returns the best model."""
    set_global_seeds(cfg["seed"])
    env = monitored_env(cfg)
    p = cfg["ppo"]
    decay = EntropyDecay(p["ent_coef"], p.get("ent_coef_final", p["ent_coef"]),
                         cfg["train"]["total_timesteps"])
    return train_model(cfg, "ppo", build_ppo(cfg, env), env, p["n_steps"], path,
                       extra_callbacks=(decay,))


def train_dqn(cfg, path=None):
    """Train DQN. Returns the best model."""
    set_global_seeds(cfg["seed"])
    env = monitored_env(cfg)
    tf = cfg["dqn"]["train_freq"] if isinstance(cfg["dqn"]["train_freq"], int) else 1
    return train_model(cfg, "dqn", build_dqn(cfg, env), env, tf, path)


def train_qrdqn(cfg, path=None):
    """Train QR-DQN. Returns the best model."""
    set_global_seeds(cfg["seed"])
    env = monitored_env(cfg)
    tf = cfg["qrdqn"]["train_freq"] if isinstance(cfg["qrdqn"]["train_freq"], int) else 1
    return train_model(cfg, "qrdqn", build_qrdqn(cfg, env), env, tf, path)


def load_model(path, algo):
    """Load a saved SB3 checkpoint by algorithm name."""
    key = algo.lower()
    if key not in ALGOS:
        raise ValueError(f"Unknown algo '{algo}'; expected one of {list(ALGOS)}")
    return ALGOS[key].load(path)


# ---- Part 2: logic-shaped reward fine-tune ----------------------------------

def to_device(model, cfg):
    """Move an SB3 model to the configured device (cpu or cuda)."""
    dev = resolve_device(cfg)
    try:
        import torch
        model.device = torch.device(dev)
        model.policy.to(dev)
    except Exception:
        pass
    return model


def finetune_logic_reward(model, cfg, drive_dir=None):
    """Fine-tune a trained model on reward - Σ λ·violation (logic-shaped reward).
    Warm-starts from the given model with a lower learning rate."""
    set_global_seeds(cfg["seed"])
    ft = cfg["finetune"]
    env = monitored_env(cfg, logic_reward=True)

    model.set_env(env)
    to_device(model, cfg)
    ft_lr = ft["learning_rate"]
    model.learning_rate = ft_lr
    model.lr_schedule = lambda _progress_remaining: ft_lr

    attach_logger(model, cfg, "part2_nesy")
    path = drive_dir or drive_path(cfg, "checkpoints", "part2_nesy.zip")
    pf = cfg.get("print_freq", 200)
    start = int(model.num_timesteps)
    extra = effective_total(ft["total_timesteps"], getattr(model, "n_steps", 1))
    print(f"[NESY-FT] fine-tuning {extra} steps on {model.device}", flush=True)
    printer = ProgressPrinter("NESY-FT", pf, best_path=path, total_steps=start + extra)
    model.learn(total_timesteps=extra, callback=printer, reset_num_timesteps=False)

    if printer.saved_best:
        print(f"[NESY-FT] done, best reward={printer.best_rew:.2f} -> {path}", flush=True)
        model = type(model).load(path)
    else:
        model.save(path)
        print(f"[NESY-FT] done -> {path}", flush=True)
    env.close()
    return model
