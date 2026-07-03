"""SAC — continuous Soft Actor-Critic (Haarnoja et al. 2018), pranz24-style.

The exact backbone shared by all four Part 1 algorithms, matching the SAC
used in the official MACURA repo (mbrl/third_party/pytorch_sac_pranz24):
  - squashed Gaussian policy, log-std clamped to [-20, 2]
  - twin Q networks + soft target updates (τ = 0.005), γ = 0.99
  - automatic entropy tuning, target entropy = -dim(A), α init 0.2
  - Adam lr 3e-4 for actor, critics and α; batch 256; no gradient clipping

Exploration modes (paper §6 / official `exploration_type_env`) — env interaction only;
evaluation is ALWAYS deterministic (explore=False → tanh(μ)). a = tanh(μ(s) + σ(s)·ε_t):
  "det"          — no noise, mean action a = tanh(μ).
  "white"        — standard SAC sampling, ε ~ N(0,1) i.i.d. per step (β=0 spectrum).
  "pink"         — episode-scoped 1/f^β colored noise, β=1 (Eberhard et al. 2023; MACURA
                   default). A length-`max_episode_steps` sequence per action dim is drawn
                   each episode via colorednoise.powerlaw_psd_gaussian(β, …).
  "red"/"brown"  — episode-scoped colored noise, β=2 (Brownian / heavily correlated).
  <float>        — episode-scoped colored noise with that β (e.g. 0.0=white, 1.5).
Only the CORRELATION across steps differs; the per-step marginal is the same Gaussian. The
β/kind is resolved lazily from `self.exploration` so a per-algorithm override is honoured.

Used standalone (model-free SAC baseline) and as inner policy for
MACURA / MBPO / M2AC.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from agents.offpolicy.replay_buffer import ReplayBuffer

try:
    import colorednoise as cn
    _HAS_COLOREDNOISE = True
except ImportError:                                   # pragma: no cover
    _HAS_COLOREDNOISE = False


# ── Networks ──────────────────────────────────────────────────────────────────

class GaussianActor(nn.Module):
    """Squashed Gaussian policy."""

    def __init__(self, state_dim, action_dim, hidden_dim, hidden_layers,
                 log_std_min=-20.0, log_std_max=2.0):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        sizes, layers = [state_dim] + [hidden_dim] * hidden_layers, []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        self.trunk        = nn.Sequential(*layers)
        self.mean_head    = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        h       = self.trunk(state)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std.exp()

    def sample(self, state):
        """Reparameterised sample → (action, log_prob, mean_action)."""
        mean, std = self(state)
        dist = Normal(mean, std)
        x    = dist.rsample()
        return self._squash(x, dist, mean)

    def sample_eps(self, state, eps):
        """Sample with externally supplied noise ε (pink-noise exploration):
        x = μ + σ·ε  (official `sample_using_eps`)."""
        mean, std = self(state)
        dist = Normal(mean, std)
        x    = mean + std * eps
        return self._squash(x, dist, mean)

    @staticmethod
    def _squash(x, dist, mean):
        y = torch.tanh(x)
        log_prob = dist.log_prob(x) - torch.log(1.0 - y.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return y, log_prob, torch.tanh(mean)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, hidden_layers):
        super().__init__()
        sizes, layers = [state_dim + action_dim] + [hidden_dim] * hidden_layers, []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.q = nn.Sequential(*layers)

    def forward(self, state, action):
        return self.q(torch.cat([state, action], dim=-1))


# ── Agent ─────────────────────────────────────────────────────────────────────

class SAC:
    """Continuous SAC backbone + model-free baseline agent."""

    name = "SAC"

    def __init__(self, state_dim: int, action_dim: int, config: dict, device: str = "cpu"):
        cfg_s = config["shared"]
        cfg_a = config.get("sac", {})
        self.device     = torch.device(device)
        self.gamma      = cfg_s["gamma"]
        self.tau        = cfg_s["tau"]
        self.action_dim = action_dim
        self.target_ent = float(cfg_a.get("target_entropy", -action_dim))
        self.target_update_interval = int(cfg_a.get("target_update_interval", 1))

        h      = cfg_a.get("hidden_dim", 256)
        layers = cfg_a.get("hidden_layers", 2)
        lr     = cfg_s["learning_rate"]

        self.actor = GaussianActor(
            state_dim, action_dim, h, layers,
            cfg_a.get("log_std_min", -20.0), cfg_a.get("log_std_max", 2.0),
        ).to(self.device)
        self.critic1   = QNetwork(state_dim, action_dim, h, layers).to(self.device)
        self.critic2   = QNetwork(state_dim, action_dim, h, layers).to(self.device)
        self.target_c1 = copy.deepcopy(self.critic1)
        self.target_c2 = copy.deepcopy(self.critic2)

        alpha_init     = float(cfg_a.get("alpha_init", 0.2))
        self.log_alpha = torch.tensor([np.log(alpha_init)], requires_grad=True,
                                      device=self.device)
        self.alpha     = self.log_alpha.exp().item()

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr)
        self.alpha_opt  = torch.optim.Adam([self.log_alpha], lr=lr)

        self.real_buffer = ReplayBuffer(cfg_s["buffer_size"], state_dim, action_dim)
        self._update_count = 0

        # ── Exploration (env interaction) ─────────────────────────────────────
        base_cfg = config.get("sac_baseline", {})
        self.exploration    = base_cfg.get("exploration", "white")
        # Noise mode is resolved LAZILY from self.exploration (see _ensure_noise_resolved)
        # so a post-construction override (model_based.py sets policy.exploration per algo)
        # is picked up.
        self._resolved_for  = None
        self._noise_kind    = "white"
        self._noise_beta    = 0.0
        self.random_warmup  = bool(base_cfg.get("random_warmup", True))
        self._updates_per_step = int(base_cfg.get("updates_per_step", 1))
        self._epoch_length  = int(cfg_s.get("epoch_length", 1000))
        self._batch_size    = int(cfg_s.get("batch_size", 256))
        # Pink-noise sequence is sized to ONE EPISODE and regenerated on each env
        # reset (reset_noise() below) — Eberhard et al.'s ColoredNoiseProcess uses a
        # chunk = episode length, so the 1/f spectrum is scoped to the episode and a
        # low-frequency excursion can't leak across resets into the next episode.
        self._noise_seq_len = int(cfg_s.get("max_episode_steps", self._epoch_length))
        self._grad_clip     = float(cfg_s.get("grad_clip_norm", 1.0))
        self._noise_rng     = np.random.default_rng(np.random.randint(2 ** 31))
        self._noise_seq     = None
        self._noise_ptr     = 0

    # ── Exploration noise ─────────────────────────────────────────────────────
    #
    # PINK-NOISE EXPLORATION (paper detail — MACURA only; Eberhard et al. 2023):
    # instead of redrawing an independent ε ~ N(0,1) at every step (white noise),
    # a temporally-correlated 1/f^β sequence with β=1 is pre-generated for one
    # epoch (`epoch_length` steps, one channel per action dim) and consumed one
    # step at a time. Each action is a = tanh(μ(s) + σ(s)·ε_t), so the per-step
    # marginal is the same Gaussian the policy would sample — only the CORRELATION
    # across steps changes. Why it matters: white noise dithers around the mean
    # and self-cancels (in a car: throttle/steer jitter that goes nowhere), while
    # pink noise holds a direction for a while WITHOUT saturating into a constant
    # offset the way OU noise can — giving temporally-extended, state-space-
    # covering exploration in continuous control. Applied ONLY to environment
    # interaction (`act(explore=True)` below); model rollouts sample the policy
    # normally, and evaluation uses the deterministic mean action (explore=False).

    @staticmethod
    def _resolve_noise(mode):
        """Map an `exploration` setting → (kind, β). Kinds: 'det' (no noise),
        'white' (legacy per-step ε~N(0,1)), 'colored' (episode-scoped 1/f^β:
        β0=white, β1=pink, β2=red/Brownian). Accepts the strings
        white/pink/red/brown/det OR a numeric β (e.g. 1.5)."""
        if isinstance(mode, (int, float)):
            return "colored", float(mode)
        m = str(mode).strip().lower()
        if m in ("det", "none", "deterministic"):
            return "det", 0.0
        if m == "white":
            return "white", 0.0                       # legacy per-step standard sampling
        named = {"pink": 1.0, "red": 2.0, "brown": 2.0, "brownian": 2.0}
        if m in named:
            return "colored", named[m]
        try:
            return "colored", float(m)                # numeric string, e.g. "1.5"
        except ValueError:
            return "white", 0.0

    def _ensure_noise_resolved(self):
        """Re-resolve the noise kind/β if `self.exploration` changed (it is set after
        construction by model_based.py for MACURA/MBPO/M2AC)."""
        if self._resolved_for != self.exploration:
            self._noise_kind, self._noise_beta = self._resolve_noise(self.exploration)
            self._resolved_for = self.exploration
            self._noise_seq = None
            self._noise_ptr = 0

    def reset_noise(self):
        """Force fresh exploration-noise state on the next step. Called by the training
        loop at every episode reset so each episode gets its own episode-scale 1/f^β
        sequence (no-op for white/det, which carry no cross-step state)."""
        self._noise_seq = None
        self._noise_ptr = 0

    def _next_colored_eps(self) -> torch.Tensor:
        """Next step of the episode-scoped 1/f^β colored-noise sequence (β = self._noise_beta;
        β0 white, β1 pink, β2 red). A fresh length-`max_episode_steps` sequence (one per action
        dim) is drawn per episode (reset_noise) — and lazily if an episode runs longer —
        matching colorednoise.powerlaw_psd_gaussian(β, (dim(A), L))."""
        if self._noise_seq is None or self._noise_ptr >= self._noise_seq.shape[1]:
            if _HAS_COLOREDNOISE:
                self._noise_seq = cn.powerlaw_psd_gaussian(
                    self._noise_beta, (self.action_dim, self._noise_seq_len),
                    random_state=self._noise_rng,
                ).astype(np.float32)
            else:                                     # graceful fallback
                self._noise_seq = self._noise_rng.standard_normal(
                    (self.action_dim, self._noise_seq_len)).astype(np.float32)
            self._noise_ptr = 0
        eps = self._noise_seq[:, self._noise_ptr]
        self._noise_ptr += 1
        return torch.from_numpy(eps).to(self.device)

    # ── Acting ────────────────────────────────────────────────────────────────

    def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Env-interaction action using the configured exploration mode.

        explore=False (evaluation) ALWAYS returns the deterministic tanh(μ)
        regardless of mode — exploration noise never touches evaluation."""
        self._ensure_noise_resolved()
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            if not explore or self._noise_kind == "det":
                _, _, a = self.actor.sample(s)            # deterministic tanh(μ)
            elif self._noise_kind == "colored":
                # temporally-correlated ε_t with spectrum 1/f^β (β1=pink → MACURA default)
                a, _, _ = self.actor.sample_eps(s, self._next_colored_eps().unsqueeze(0))
            else:   # "white" — standard SAC sampling, ε ~ N(0,1) i.i.d. per step
                a, _, _ = self.actor.sample(s)
        return a.squeeze(0).cpu().numpy().astype(np.float32)

    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Kept for demo / evaluation compatibility (explore=False → tanh(μ))."""
        return self.act(state, explore)

    def select_action_batch(self, states: torch.Tensor, explore: bool = True):
        """Batched on-device action selection (used inside model rollouts)."""
        if explore:
            actions, log_probs, _ = self.actor.sample(states)
        else:
            with torch.no_grad():
                _, _, actions = self.actor.sample(states)
                log_probs = torch.zeros(states.shape[0], 1, device=self.device)
        return actions, log_probs

    # ── Data + training ───────────────────────────────────────────────────────

    def store(self, state, action, reward, next_state, done):
        self.real_buffer.add(state, np.asarray(action, dtype=np.float32),
                             reward, next_state, float(done))

    def update_step(self, env_step: int) -> dict:
        """Model-free baseline: `updates_per_step` gradient steps per env step."""
        metrics = {}
        for _ in range(self._updates_per_step):
            if len(self.real_buffer) < self._batch_size:
                break
            metrics = self.update(self.real_buffer, self._batch_size)
        if metrics:
            # Model-free: every update batch is REAL — no imagined rollouts ever.
            metrics["real_pct"], metrics["imagined_pct"] = 100.0, 0.0
        return metrics

    def update(self, buffer, batch_size: int) -> dict:
        """One SAC gradient step from `buffer` (pranz24 update_parameters)."""
        s, a, r, ns, d = buffer.sample(batch_size, self.device)

        with torch.no_grad():
            a_ns, log_p_ns, _ = self.actor.sample(ns)
            q_ns  = torch.min(self.target_c1(ns, a_ns), self.target_c2(ns, a_ns))
            q_tgt = r + self.gamma * (1.0 - d) * (q_ns - self.alpha * log_p_ns)

        c_loss = F.mse_loss(self.critic1(s, a), q_tgt) + \
                 F.mse_loss(self.critic2(s, a), q_tgt)
        self.critic_opt.zero_grad(set_to_none=True)
        c_loss.backward()
        # Gradient-norm clip — a numerical safeguard, not an algorithm change. MACURA
        # takes up to ~G≈10–20 updates/env-step (Eq. 22) almost entirely on imagined
        # data whose reward can hit the ±20 clamp; without clipping those large TD
        # errors produce huge, jittery steps that destabilise the policy. (no_limit
        # clipped to 1.0; pranz24 does not clip — this is the one safety net we keep.)
        if self._grad_clip:
            torch.nn.utils.clip_grad_norm_(
                list(self.critic1.parameters()) + list(self.critic2.parameters()),
                self._grad_clip)
        self.critic_opt.step()

        a_new, log_p_new, _ = self.actor.sample(s)
        q_min  = torch.min(self.critic1(s, a_new), self.critic2(s, a_new))
        a_loss = (self.alpha * log_p_new - q_min).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        a_loss.backward()
        if self._grad_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self._grad_clip)
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_p_new + self.target_ent).detach()).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        self._update_count += 1
        if self._update_count % self.target_update_interval == 0:
            self._soft_update(self.critic1, self.target_c1)
            self._soft_update(self.critic2, self.target_c2)

        return {
            "critic_loss": c_loss.item(),
            "actor_loss":  a_loss.item(),
            "alpha":       self.alpha,
            "entropy":     float(-log_p_new.mean().item()),
        }

    def _soft_update(self, src, tgt):
        for p, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path_prefix: str):
        torch.save({
            "actor":     self.actor.state_dict(),
            "critic1":   self.critic1.state_dict(),
            "critic2":   self.critic2.state_dict(),
            "log_alpha": self.log_alpha.data,
        }, f"{path_prefix}_policy.pt")

    def load(self, path_prefix: str):
        ckpt = torch.load(f"{path_prefix}_policy.pt", map_location=self.device)

        def _smart(model, sd):
            """Direct load, falling back to stripping the torch.compile prefix."""
            try:
                model.load_state_dict(sd)
            except RuntimeError:
                model.load_state_dict(
                    {k.replace("_orig_mod.", ""): v for k, v in sd.items()})
        _smart(self.actor,   ckpt["actor"])
        _smart(self.critic1, ckpt["critic1"])
        _smart(self.critic2, ckpt["critic2"])
        self.target_c1 = copy.deepcopy(self.critic1)
        self.target_c2 = copy.deepcopy(self.critic2)
        self.log_alpha.data.copy_(ckpt["log_alpha"])
        self.alpha = self.log_alpha.exp().item()
