"""MACURA — Model-Based Actor-Critic with Uncertainty-Aware Rollout Adaption.
Frauenknecht et al., ICML 2024 (arXiv:2405.19014) — faithful re-implementation
of the official repo (github.com/Data-Science-in-Mechanical-Engineering/macura).

Mechanism (paper §4):
  - uncertainty u_GJS(s,a): mean pairwise Geometric Jensen–Shannon divergence
    between the E ensemble members' Gaussian predictions.
  - adaptive threshold κ: at every rollout round, the ζ-percentile (ζ=95) of
    the FIRST-step uncertainties across the rollout batch is scaled by ξ
    ("border" of the well-calibrated region); κ is the running average of
    these border values over the last `unc_thresh_run_avg_history` rounds.
  - a rollout particle continues only while u_GJS < κ, up to T_max steps —
    so rollout length adapts per state to local model reliability.
  - exploration: pink noise (β=1) on the env policy (Eberhard et al. 2023).
  - update count per env step scales with model-buffer filling (paper Eq. 22):
        G = int(2·|D_model| / capacity · G_max)
"""
import numpy as np
import torch

from agents.offpolicy.model_based import ModelBasedAgent


class MACURA(ModelBasedAgent):
    name = "MACURA"
    algo_key = "macura"

    def __init__(self, state_dim, action_dim, config, device="cpu"):
        super().__init__(state_dim, action_dim, config, device)
        cfg = config["macura"]
        self.xi          = float(cfg["xi"])
        self.zeta        = float(cfg["zeta"])
        history          = int(cfg.get("unc_thresh_run_avg_history", 2000))
        self._border_history = []          # last `history` per-round border values
        self._border_max     = history
        self.kappa           = 0.0         # current adaptive threshold (logged)
        # Dual stopping (NeSy novelty): rollouts ALSO stop where a temporal-logic
        # rule fires — never learn from imagined futures that break the law.
        # Rule params + obs layout are injected by the trainer (config["rules"] /
        # config["env_layout"]).
        self.tl_stopping = bool(cfg.get("tl_stopping", False))
        self._rules      = config.get("rules")
        layout           = config.get("env_layout", {})
        self._lanes      = int(layout.get("lanes_count", 4))
        self._feat_per_v = int(layout.get("features_per_vehicle", 5))
        if self.tl_stopping and self._rules is None:
            raise ValueError("tl_stopping needs config['rules'] (injected by the trainer)")

    # Paper Eq. 22 — gradient steps scale with model-buffer filling
    def _num_updates(self) -> int:
        cap = max(1, self.model_buffer.capacity)
        return int((len(self.model_buffer) * 2 / cap) * self.g_max)

    @torch.no_grad()
    def _generate_rollouts(self, env_step: int) -> dict:
        states = self._sample_start_states()
        n_start = states.shape[0]
        if n_start < 2:
            return {}

        stored    = 0
        terminals = 0
        tl_cut    = 0
        threshold = self.kappa

        for depth in range(self.max_rollout_length):
            actions, _ = self.policy.select_action_batch(states, explore=True)
            next_states, rewards, means_all, vars_all, _, term = \
                self.ensemble.step(states, actions)
            u = self.ensemble.gjs_uncertainty(means_all, vars_all)   # (B,)

            if depth == 0:
                # border = ξ × ζ-percentile of first-step uncertainties;
                # κ = running average incl. the current round (official update rule)
                border = float(torch.quantile(u, self.zeta / 100.0)) * self.xi
                c = len(self._border_history)
                threshold = (border + c * (np.mean(self._border_history) if c else 0.0)) / (c + 1)
                self._border_history.append(border)
                if len(self._border_history) > self._border_max:
                    self._border_history.pop(0)
                self.kappa = threshold

            certain = u < threshold                                   # (B,) bool
            if self.tl_stopping:
                # dual stopping: drop imagined transitions that land in a rule
                # violation (RG1 unsafe gap / RG3 over limit) — same gating as
                # the uncertainty cut, entirely separate from GJS/κ.
                from agents.offpolicy.dual_stopping import tl_violation_mask
                illegal = tl_violation_mask(next_states, self._rules,
                                            self._lanes, self._feat_per_v)
                tl_cut += int((certain & illegal).sum().item())
                certain = certain & ~illegal
            if not bool(certain.any()):
                break

            # `term` (from the ensemble's learned termination head) → done=1 and the
            # particle stops: the env termination_fn for imagined rollouts.
            keep_states  = states[certain]
            keep_actions = actions[certain]
            keep_next    = next_states[certain]
            keep_rewards = rewards[certain]
            keep_done    = term[certain].to(torch.float32)
            m = keep_states.shape[0]
            self.model_buffer.add_batch(
                keep_states.cpu().numpy(),
                keep_actions.cpu().numpy(),
                keep_rewards.cpu().numpy(),
                keep_next.cpu().numpy(),
                keep_done.cpu().numpy(),       # real done → SAC masks the bootstrap
            )
            stored    += m
            terminals += int(keep_done.sum().item())

            # Continue only particles that are BOTH certain AND not terminated.
            cont   = certain & (~term)
            states = next_states[cont]
            if states.shape[0] == 0:
                break

        out = {
            "rollout_depth_mean": stored / n_start,
            "kappa":              self.kappa,
            "model_transitions":  stored,
            "model_term_frac":    terminals / max(1, stored),
        }
        if self.tl_stopping:
            out["tl_cut_frac"] = tl_cut / max(1, stored + tl_cut)
        return out
