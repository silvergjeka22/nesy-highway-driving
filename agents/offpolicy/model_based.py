"""Shared Dyna machinery for MACURA / MBPO / M2AC — faithful to the official
MACURA repo (mbrl-lib fork) training loop:

  every `freq_train_model` env steps:
      1. retrain the ensemble on the FULL real buffer (to convergence)
      2. start `effective_rollouts_per_step × freq_train_model` imagined
         rollouts from states sampled uniformly from the real buffer
         (each algorithm decides how the rollouts are truncated / masked)
  every env step:
      G policy gradient steps; each update batch is drawn ENTIRELY from the
      real buffer with probability `real_ratio` (5%), otherwise entirely
      from the model buffer (batch-level mixing, as in the official code).

The model buffer expires data older than
`retain_epochs × (epoch_length / freq_train_model)` rollout rounds
(the paper's dynamic-lifetime SAC buffer).
"""
import numpy as np
import torch

from agents.offpolicy.sac import SAC
from agents.offpolicy.ensemble import GaussianEnsemble
from agents.offpolicy.replay_buffer import LifetimeReplayBuffer


class ModelBasedAgent:
    """Base class — subclasses implement `_generate_rollouts`."""

    name = "ModelBased"
    algo_key = None          # config section name, set by subclass

    def __init__(self, state_dim: int, action_dim: int, config: dict, device: str = "cpu"):
        cfg_s  = config["shared"]
        cfg_mb = config["model_based"]
        cfg_a  = config.get(self.algo_key, {})

        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.device     = torch.device(device)
        self.cfg        = config

        self.freq_train_model   = int(cfg_mb["freq_train_model"])
        self.max_rollout_length = int(cfg_mb["max_rollout_length"])
        self.g_max              = int(cfg_mb["g_max"])
        self.real_ratio         = float(cfg_s["real_ratio"])
        self.batch_size         = int(cfg_s["batch_size"])
        epoch_length            = int(cfg_s["epoch_length"])
        retain_epochs           = int(cfg_mb["retain_epochs"])

        # Paper semantics: rollout starts per round = effective/step × freq
        self.rollout_batch_size = (
            int(cfg_mb["effective_rollouts_per_step"]) * self.freq_train_model
        )
        self._trains_per_epoch = max(1, epoch_length // self.freq_train_model)
        self._retain_epochs    = retain_epochs

        self.policy   = SAC(state_dim, action_dim, config, device)
        self.ensemble = GaussianEnsemble(state_dim, action_dim, config, device)
        self.model_buffer = self._make_model_buffer()

        # Model-based agents explore with the (untrained) policy, not random actions
        self.policy.exploration   = cfg_a.get("exploration", "white")
        self.random_warmup        = bool(cfg_a.get("random_warmup", False))
        self._rng                 = np.random.default_rng(np.random.randint(2 ** 31))
        self._model_trained       = False
        self._last_metrics        = {}

        # ── Shared real-data-first curriculum (review/AGENT_PROMPT_curriculum.md §2) ──
        # real_ratio is read from this schedule per env-step (was a fixed self.real_ratio). The
        # SAME schedule is built for MACURA/MBPO/M2AC — no per-algorithm branching — so it cannot
        # bias the comparison; only their rollout strategy differs. Disabled / absent → constant
        # paper real_ratio (behaviour identical to before). Imported locally to avoid a
        # training↔algorithms package import cycle at module-load time.
        from agents.offpolicy.curriculum import RealRatioSchedule
        cur = config.get("curriculum") or {}
        if cur.get("enabled", False) and cur.get("real_ratio_knots"):
            self._real_ratio_schedule = RealRatioSchedule.from_config(cur)
        else:
            self._real_ratio_schedule = RealRatioSchedule.constant(self.real_ratio)

        # Optional warmup-only exploration override (review §3) — env interaction only, NOT an
        # update-rule change. OFF by default (null) → the per-algorithm paper noise throughout.
        self._main_exploration     = self.policy.exploration
        self._warmup_exploration   = cur.get("warmup_exploration")
        self._warmup_explore_until = int(cur.get("warmup_exploration_until_step", 0))
        if self._warmup_exploration is not None and self._warmup_explore_until > 0:
            self.policy.exploration = self._warmup_exploration
            self._explore_switched  = False
        else:
            self._explore_switched  = True

    def _make_model_buffer(self) -> LifetimeReplayBuffer:
        """Capacity = T_max × rollout_batch × trains_per_epoch × retain_epochs;
        data expires after retain_epochs × trains_per_epoch rounds (paper)."""
        lifetime = self._retain_epochs * self._trains_per_epoch
        capacity = (self.max_rollout_length * self.rollout_batch_size
                    * self._trains_per_epoch * self._retain_epochs)
        return LifetimeReplayBuffer(capacity, self.state_dim, self.action_dim, lifetime)

    # ── Delegation to the inner SAC ───────────────────────────────────────────

    @property
    def real_buffer(self):
        return self.policy.real_buffer

    def act(self, state, explore: bool = True):
        return self.policy.act(state, explore)

    def reset_noise(self):
        """Forward the per-episode exploration-noise reset to the inner SAC policy."""
        self.policy.reset_noise()

    def select_action(self, state, explore: bool = True):
        return self.policy.act(state, explore)

    def store(self, state, action, reward, next_state, done):
        self.policy.store(state, action, reward, next_state, done)

    # ── Per-step update (called once per env step after warmup) ───────────────

    def update_step(self, env_step: int) -> dict:
        metrics = {}

        # Restore the per-algorithm default exploration once past the real-data warmup (review §3).
        # No-op unless a warmup_exploration override is configured.
        if not self._explore_switched and env_step > self._warmup_explore_until:
            self.policy.exploration = self._main_exploration
            self.policy.reset_noise()
            self._explore_switched = True

        if env_step % self.freq_train_model == 0 and len(self.real_buffer) > 0:
            metrics.update(self.train_world_model())
            self.model_buffer.begin_round()
            metrics.update(self._generate_rollouts(env_step))
            self._model_trained = True

        if self._model_trained:
            # SHARED curriculum: real_ratio ramps with env-step (was a fixed self.real_ratio), so
            # early updates draw from the REAL buffer while the model matures, then lean on imagined
            # rollouts. Rollout length is untouched — each algorithm keeps its own strategy.
            rr = self._real_ratio_schedule(env_step)
            n_real, n_model = 0, 0
            for _ in range(self._num_updates()):
                use_real = self._rng.random() < rr
                buf = self.real_buffer if use_real else self.model_buffer
                if len(buf) < self.batch_size:
                    break
                n_real += int(use_real)
                n_model += int(not use_real)
                metrics.update(self.policy.update(buf, self.batch_size))
            # Curriculum visibility: scheduled target + REALISED real/imagined split this step.
            n_tot = n_real + n_model
            metrics["real_ratio_target"] = rr
            metrics["real_pct"]     = 100.0 * n_real / n_tot if n_tot else 100.0 * rr
            metrics["imagined_pct"] = 100.0 * n_model / n_tot if n_tot else 100.0 * (1.0 - rr)

        if metrics:
            self._last_metrics = metrics
        return metrics

    def _num_updates(self) -> int:
        """Fixed G for MBPO / M2AC (official); MACURA overrides with Eq. 22."""
        return self.g_max

    def train_world_model(self) -> dict:
        s, a, r, ns, d = self.real_buffer.sample_all()
        return self.ensemble.train(s, a, ns, r, d)   # d → fits the termination head

    # ── Rollouts — algorithm-specific ──────────────────────────────────────────

    def _generate_rollouts(self, env_step: int) -> dict:
        raise NotImplementedError

    def _sample_start_states(self) -> torch.Tensor:
        n = min(self.rollout_batch_size, len(self.real_buffer))
        states_np = self.real_buffer.sample_states(n)
        return torch.from_numpy(states_np).to(self.device)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path_prefix: str):
        self.policy.save(path_prefix)
        torch.save(self.ensemble.state_dict(), f"{path_prefix}_ensemble.pt")

    def load(self, path_prefix: str):
        self.policy.load(path_prefix)
        ckpt = torch.load(f"{path_prefix}_ensemble.pt", map_location=self.device)
        self.ensemble.load_state_dict(ckpt)
