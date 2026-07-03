"""Probabilistic ensemble dynamics model — faithful to MACURA (ICML 2024).

Replicates the mbrl-lib GaussianMLP ensemble used by the official MACURA repo:
  - E probabilistic networks, 4 hidden layers × 200 units, SiLU activations
  - input  : z-score-normalised [state ‖ action]  (inputs only, like mbrl-lib)
  - output : Gaussian over [Δstate ‖ reward]  (target_is_delta, learned_rewards)
  - fixed soft log-variance bounds  [min_log_var, max_log_var] = [-10, 0.5]
  - trained with the Gaussian NLL until validation stops improving
    (validation_ratio, patience, improvement_threshold from the paper)
  - "random_model" propagation: each rollout particle is advanced by a
    randomly drawn ELITE member at every step
  - uncertainty:
      u_GJS  — mean pairwise Geometric Jensen–Shannon divergence between all
               E(E−1)/2 member Gaussians (MACURA, paper §4 / Eq. 12-13)
      u_OvR  — KL(chosen member ‖ moment-matched mixture of the rest)
               (M2AC baseline, Pan et al. 2020)
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbabilisticNet(nn.Module):
    """One ensemble member: normalised [s‖a] → Gaussian over [Δs‖r]."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int,
                 hidden_layers: int, min_log_var: float, max_log_var: float):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        self.trunk        = nn.Sequential(*layers)
        self.mean_head    = nn.Linear(hidden_dim, out_dim)
        self.log_var_head = nn.Linear(hidden_dim, out_dim)
        # Fixed (non-learned) soft bounds — learn_logvar_bounds=false in the paper cfg
        self.register_buffer("max_log_var", torch.full((out_dim,), float(max_log_var)))
        self.register_buffer("min_log_var", torch.full((out_dim,), float(min_log_var)))

    def forward(self, x: torch.Tensor):
        h       = self.trunk(x)
        mean    = self.mean_head(h)
        log_var = self.max_log_var - F.softplus(self.max_log_var - self.log_var_head(h))
        log_var = self.min_log_var + F.softplus(log_var - self.min_log_var)
        return mean, log_var


class GaussianEnsemble:
    """E probabilistic networks + normaliser + uncertainty measures."""

    def __init__(self, state_dim: int, action_dim: int, config: dict, device: str = "cpu"):
        cfg             = config["ensemble"]
        self.E          = cfg["num_models"]
        self.num_elites = min(self.E, cfg.get("num_elites", self.E))
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.device     = torch.device(device)

        in_dim  = state_dim + action_dim
        out_dim = state_dim + 1
        self.nets = nn.ModuleList([
            ProbabilisticNet(in_dim, out_dim, cfg["hidden_dim"], cfg["hidden_layers"],
                             cfg["min_log_var"], cfg["max_log_var"])
            for _ in range(self.E)
        ]).to(self.device)

        # Single optimiser over the whole ensemble (mbrl-lib ModelTrainer semantics)
        self.optimizer = torch.optim.Adam(
            self.nets.parameters(),
            lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
        )

        self.batch_size            = cfg["batch_size"]
        self.validation_ratio      = cfg["validation_ratio"]
        self.patience              = cfg["patience"]
        self.improvement_threshold = cfg["improvement_threshold"]
        self.max_epochs            = cfg.get("max_epochs", 50)

        # Input normalisation — hard-updated from the FULL dataset at each train()
        # call (mbrl-lib Normalizer.update_stats), never an EMA.
        self.in_mean = torch.zeros(in_dim, device=self.device)
        self.in_std  = torch.ones(in_dim,  device=self.device)

        self.elite_indices = list(range(self.num_elites))

        # HighwayEnv adaptation: normalised Kinematics observations are bounded,
        # so imagined states/rewards are clamped to a generous envelope. The
        # real env can never leave it — the clamp only binds where the model is
        # already wrong, and it keeps early-training Q-targets finite.
        cfg_mb = config.get("model_based", {})
        self.obs_clip    = cfg_mb.get("obs_clip", 2.0)
        self.reward_clip = cfg_mb.get("reward_clip", 20.0)

        # ── LEARNED TERMINATION HEAD (restores the paper's per-env termination_fn) ──
        # A small MLP, PARALLEL to the Gaussian dynamics members: it predicts
        # P(episode terminates) from the normalised [state‖action] and is trained with
        # BCE on the env's real `done` flags. It does NOT touch the GaussianMLP members,
        # the GJS uncertainty, or κ — those see exactly the same inputs/outputs as
        # before. It replaces the reward-threshold crash proxy, which the Gaussian
        # reward head smears so badly that it never fired (term=0%): a sharp ±done
        # boundary is learnable with BCE where a −6 reward spike is not with NLL.
        self.term_net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.SiLU(),
            nn.Linear(128, 128),    nn.SiLU(),
            nn.Linear(128, 1),
        ).to(self.device)
        self.term_opt = torch.optim.Adam(self.term_net.parameters(),
                                         lr=cfg["learning_rate"])
        self.term_threshold  = float(cfg_mb.get("term_prob_threshold", 0.5))
        self.term_max_epochs = int(cfg_mb.get("term_max_epochs", 20))
        # Hand-coded crash term_fn radius (normalised relative distance). A non-ego
        # vehicle this close to the ego in the PREDICTED next-state ⇒ terminal — the
        # direct analogue of the env's real crash condition (vehicle overlap), which is
        # a KNOWN function of state (exactly what the paper's term_fn is). Conservative
        # / tight by default: the wrapper treats <0.25 as merely "dangerous", a real
        # crash is far closer, so this only cuts genuine imminent collisions and cannot
        # over-truncate. Tune UP if term% stays ~0, DOWN if rollout depth collapses.
        self.crash_distance  = float(cfg_mb.get("crash_distance", 0.04))

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _model_input(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions], dim=-1)
        return (x - self.in_mean) / self.in_std

    @staticmethod
    def _nll(mean, log_var, target) -> torch.Tensor:
        inv_var = torch.exp(-log_var)
        return ((mean - target) ** 2 * inv_var + log_var).mean()

    # ── Training ───────────────────────────────────────────────────────────────

    def train(self, states: np.ndarray, actions: np.ndarray,
              next_states: np.ndarray, rewards: np.ndarray,
              dones: np.ndarray = None) -> dict:
        """Train all members on the full real buffer until validation NLL stops
        improving (paper protocol). If `dones` is given, also fits the termination
        head (BCE). Returns training metrics."""
        N = len(states)
        inputs  = np.concatenate([states, actions], axis=-1).astype(np.float32)
        targets = np.concatenate(
            [next_states - states, rewards.reshape(-1, 1)], axis=-1
        ).astype(np.float32)

        # Hard normaliser update from the full dataset. Near-constant input
        # features (e.g. ego presence ≡ 1, ego x ≡ 0 with absolute=False) get
        # std := 1.0 — exactly mbrl-lib's Normalizer rule; clamping to a tiny
        # epsilon instead would scale those columns by ~1e8 and NaN the nets.
        std = inputs.std(0)
        std[std < 1e-5] = 1.0
        self.in_mean = torch.from_numpy(inputs.mean(0)).to(self.device)
        self.in_std  = torch.from_numpy(std).to(self.device)

        perm      = np.random.permutation(N)
        n_val     = max(1, int(N * self.validation_ratio))
        val_idx   = perm[:n_val]
        train_idx = perm[n_val:]
        n_train   = len(train_idx)

        x_all = torch.from_numpy(inputs).to(self.device)
        y_all = torch.from_numpy(targets).to(self.device)
        x_val = x_all[val_idx]
        y_val = y_all[val_idx]
        x_val_n = (x_val - self.in_mean) / self.in_std

        best_val      = np.full(self.E, np.inf)
        epochs_no_imp = 0
        train_loss    = 0.0
        epochs_run    = 0

        for epoch in range(self.max_epochs):
            epochs_run = epoch + 1
            # Bootstrap: each member sees its own shuffling of the training set
            member_perms = [train_idx[np.random.permutation(n_train)]
                            for _ in range(self.E)]
            epoch_loss, n_batches = 0.0, 0
            for start in range(0, n_train, self.batch_size):
                loss = 0.0
                for m, net in enumerate(self.nets):
                    idx = member_perms[m][start: start + self.batch_size]
                    x_n = (x_all[idx] - self.in_mean) / self.in_std
                    mean, log_var = net(x_n)
                    loss = loss + self._nll(mean, log_var, y_all[idx])
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                epoch_loss += float(loss.detach()) / self.E
                n_batches  += 1
            train_loss = epoch_loss / max(1, n_batches)

            # Per-member validation NLL — early stop when no member improves
            val_losses = self._validation_nll(x_val_n, y_val)
            improved = False
            for m in range(self.E):
                if not np.isfinite(best_val[m]):
                    member_improved = True
                else:
                    rel = (best_val[m] - val_losses[m]) / max(abs(best_val[m]), 1e-8)
                    member_improved = rel > self.improvement_threshold
                if member_improved:
                    best_val[m] = val_losses[m]
                    improved = True
            epochs_no_imp = 0 if improved else epochs_no_imp + 1
            if epochs_no_imp >= self.patience:
                break

        # Elite members = lowest validation NLL
        val_losses = self._validation_nll(x_val_n, y_val)
        order = np.argsort(val_losses)
        self.elite_indices = [int(i) for i in order[: self.num_elites]]

        # ── Termination head: BCE on the real done flags ──────────────────────────
        term_metrics = {}
        if dones is not None:
            term_metrics = self._train_term_head(x_all, train_idx, val_idx, dones)

        return {
            "model_loss":     float(train_loss),
            "model_val_nll":  float(np.mean(val_losses)),
            "model_epochs":   epochs_run,
            **term_metrics,
        }

    def _train_term_head(self, x_all, train_idx, val_idx, dones) -> dict:
        """Fit the termination head with BCE on the env's real `done` flags. Dones
        are sparse (crashes/off-road are a minority), so a pos_weight rebalances the
        loss. Trains to a small fixed budget with early stopping on validation BCE."""
        d = torch.from_numpy(np.asarray(dones, np.float32).reshape(-1, 1)).to(self.device)
        x_tr = (x_all[train_idx] - self.in_mean) / self.in_std
        x_va = (x_all[val_idx]   - self.in_mean) / self.in_std
        y_tr, y_va = d[train_idx], d[val_idx]

        n_pos = float(y_tr.sum().item())
        n_neg = float(len(y_tr) - n_pos)
        if n_pos < 1.0:                          # no terminations seen yet → predict ~0
            with torch.no_grad():
                p_hat = torch.sigmoid(self.term_net(x_va)).mean().item()
            return {"term_pos_rate": float(y_tr.mean().item()), "term_pred_rate": float(p_hat),
                    "term_val_bce": 0.0}
        pos_weight = torch.tensor([max(1.0, n_neg / max(1.0, n_pos))], device=self.device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best, no_imp = np.inf, 0
        ntr = len(train_idx)
        for _ in range(self.term_max_epochs):
            perm = np.random.permutation(ntr)
            for s in range(0, ntr, self.batch_size):
                bi = perm[s: s + self.batch_size]
                self.term_opt.zero_grad(set_to_none=True)
                loss = bce(self.term_net(x_tr[bi]), y_tr[bi])
                loss.backward()
                self.term_opt.step()
            with torch.no_grad():
                vbce = float(bce(self.term_net(x_va), y_va))
            if vbce < best - 1e-4:
                best, no_imp = vbce, 0
            else:
                no_imp += 1
            if no_imp >= self.patience:
                break
        with torch.no_grad():                    # recall on the held-out positives
            p_va = torch.sigmoid(self.term_net(x_va))
            recall = float(((p_va > self.term_threshold).float() * y_va).sum()
                           / max(1.0, float(y_va.sum())))
        return {"term_pos_rate": float(y_tr.mean().item()),
                "term_pred_rate": float((p_va > self.term_threshold).float().mean().item()),
                "term_recall": recall, "term_val_bce": best}

    @torch.no_grad()
    def _validation_nll(self, x_val_n: torch.Tensor, y_val: torch.Tensor) -> np.ndarray:
        losses = []
        for net in self.nets:
            mean, log_var = net(x_val_n)
            losses.append(float(self._nll(mean, log_var, y_val)))
        return np.asarray(losses)

    # ── Rollout step (random-elite propagation, paper) ─────────────────────────

    @torch.no_grad()
    def step(self, states: torch.Tensor, actions: torch.Tensor
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                        torch.Tensor, torch.Tensor]:
        """One imagined step for a batch of particles.

        Returns:
            next_states (B, S)     — s + Δs sampled from the chosen member
            rewards     (B,)       — sampled reward
            means_all   (E, B, S+1) — per-member Gaussian means over [Δs‖r]
            vars_all    (E, B, S+1) — per-member Gaussian variances
            chosen_idx  (B,)       — which member advanced each particle
            done        (B,) bool  — learned termination: P(done) > term_threshold
        """
        B   = states.shape[0]
        x_n = self._model_input(states, actions)

        means, log_vars = [], []
        for net in self.nets:
            m, lv = net(x_n)
            means.append(m)
            log_vars.append(lv)
        means_all = torch.stack(means, dim=0)              # (E, B, S+1)
        vars_all  = torch.stack(log_vars, dim=0).exp()     # (E, B, S+1)

        # random_model propagation: a random ELITE member per particle, per step
        elite = torch.as_tensor(self.elite_indices, device=self.device)
        chosen_idx = elite[torch.randint(len(self.elite_indices), (B,), device=self.device)]

        ar = torch.arange(B, device=self.device)
        mean_c = means_all[chosen_idx, ar]                 # (B, S+1)
        std_c  = vars_all[chosen_idx, ar].sqrt()
        sample = mean_c + std_c * torch.randn_like(std_c)

        next_states = states + sample[:, :-1]
        rewards     = sample[:, -1]
        if self.obs_clip:
            next_states = next_states.clamp(-self.obs_clip, self.obs_clip)
        if self.reward_clip:
            rewards = rewards.clamp(-self.reward_clip, self.reward_clip)

        # ── TERMINATION for imagined rollouts (paper's per-env termination_fn) ──
        # Primary: a HAND-CODED geometric crash test on the predicted next-state — a
        # present non-ego vehicle within `crash_distance` of the ego. This is the direct
        # analogue of the env's real crash condition (vehicle overlap), a KNOWN function
        # of state, which is precisely what the paper's term_fn is. It is OR'd with the
        # LEARNED head (which also catches off-road and anything else it has learned);
        # the learned head ALONE fired ~0% because crashes are rare and the BCE head
        # under-predicts. Both are SEPARATE from the GJS Gaussian, so the uncertainty
        # mechanism and κ are completely untouched.
        # Layout: n_veh×7 features [presence, rel_x, rel_y, ...] per vehicle; row 0 is
        # the ego, rows 1: are the other vehicles. (S//7)*7 keeps whole 7-feature rows.
        S       = next_states.shape[1]
        kin_dim = (S // 7) * 7
        kin     = next_states[:, :kin_dim].reshape(next_states.shape[0], -1, 7)
        others  = kin[:, 1:, :]                                   # (B, n_veh-1, 7)
        present = others[:, :, 0] > 0.5
        dist    = torch.sqrt(others[:, :, 1] ** 2 + others[:, :, 2] ** 2)
        crash   = ((dist < self.crash_distance) & present).any(dim=1)   # (B,)
        learned = torch.sigmoid(self.term_net(x_n)).squeeze(-1) > self.term_threshold
        done    = crash | learned
        return next_states, rewards, means_all, vars_all, chosen_idx, done

    # ── Uncertainty measures ───────────────────────────────────────────────────

    @staticmethod
    def gjs_pair(mu1, var1, mu2, var2) -> torch.Tensor:
        """Geometric Jensen–Shannon divergence between two diagonal Gaussians
        (closed form, α = ½ — exactly the official repo's calc_uncertainty_score_genShen).
        Inputs are (B, D); returns (B,)."""
        al = 0.5
        t1 = ((1 - al) * mu1 * mu1 / var1).sum(-1)
        t2 = (al * mu2 * mu2 / var2).sum(-1)
        var_al = 1.0 / ((1 - al) / var1 + al / var2)        # harmonic-mean covariance
        mu_al  = var_al * ((1 - al) * mu1 / var1 + al * mu2 / var2)
        t3 = (mu_al * mu_al / var_al).sum(-1)
        log_term = ((1 - al) * var1.log() + al * var2.log() - var_al.log()).sum(-1)
        return 0.5 * (t1 + t2 - t3 + log_term)

    @torch.no_grad()
    def gjs_uncertainty(self, means_all: torch.Tensor, vars_all: torch.Tensor) -> torch.Tensor:
        """u_GJS(s,a) = mean over all E(E−1)/2 member pairs of D_GJS (paper Eq. 13).
        means_all/vars_all: (E, B, D). Returns (B,)."""
        total, count = None, 0
        for j in range(1, self.E):
            for k in range(j):
                d = self.gjs_pair(means_all[j], vars_all[j], means_all[k], vars_all[k])
                total = d if total is None else total + d
                count += 1
        return total / count

    @torch.no_grad()
    def ovr_kl_uncertainty(self, means_all: torch.Tensor, vars_all: torch.Tensor,
                           chosen_idx: torch.Tensor) -> torch.Tensor:
        """M2AC one-vs-rest uncertainty: KL(chosen ‖ moment-matched mixture of rest).
        means_all/vars_all: (E, B, D); chosen_idx: (B,). Returns (B,)."""
        E, B, D = means_all.shape
        ar = torch.arange(B, device=means_all.device)
        mu_c  = means_all[chosen_idx, ar]                  # (B, D)
        var_c = vars_all[chosen_idx, ar]

        # Moment-matched mixture of the E−1 remaining members
        mask = torch.ones(E, B, 1, device=means_all.device)
        mask[chosen_idx, ar] = 0.0
        denom   = float(E - 1)
        mu_r    = (means_all * mask).sum(0) / denom
        ex2     = ((vars_all + means_all ** 2) * mask).sum(0) / denom
        var_r   = (ex2 - mu_r ** 2).clamp(min=1e-8)

        # KL(N_c ‖ N_r) for diagonal Gaussians
        log_term = (var_r.log() - var_c.log()).sum(-1)
        tr_term  = (var_c / var_r).sum(-1)
        quad     = ((mu_r - mu_c) ** 2 / var_r).sum(-1)
        return 0.5 * (log_term - D + tr_term + quad)

    @torch.no_grad()
    def gjs_uncertainty_np(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Convenience numpy interface (notebook checks / Part 2)."""
        s = torch.as_tensor(states,  dtype=torch.float32, device=self.device)
        a = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if s.ndim == 1:
            s, a = s.unsqueeze(0), a.unsqueeze(0)
        x_n = self._model_input(s, a)
        means, log_vars = [], []
        for net in self.nets:
            m, lv = net(x_n)
            means.append(m)
            log_vars.append(lv)
        means_all = torch.stack(means, 0)
        vars_all  = torch.stack(log_vars, 0).exp()
        return self.gjs_uncertainty(means_all, vars_all).cpu().numpy()

    # ── Persistence ───────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "nets":     [net.state_dict() for net in self.nets],
            "in_mean":  self.in_mean.cpu(),
            "in_std":   self.in_std.cpu(),
            "elites":   self.elite_indices,
            "term_net": self.term_net.state_dict(),
        }

    def load_state_dict(self, ckpt: dict):
        def _smart(net, sd):
            try:
                net.load_state_dict(sd)
            except RuntimeError:
                net.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
        if "nets" in ckpt:
            for net, sd in zip(self.nets, ckpt["nets"]):
                _smart(net, sd)
            self.in_mean = ckpt["in_mean"].to(self.device)
            self.in_std  = ckpt["in_std"].to(self.device)
            self.elite_indices = list(ckpt.get("elites", self.elite_indices))
            if "term_net" in ckpt:
                _smart(self.term_net, ckpt["term_net"])
        else:   # legacy format: {i: state_dict}
            for i, net in enumerate(self.nets):
                if i in ckpt:
                    _smart(net, ckpt[i])
