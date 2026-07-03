"""Shared real-data-first curriculum for the MODEL-BASED agents (MACURA / MBPO / M2AC).

A DOCUMENTED DEVIATION from the paper's fixed real_ratio (see AGENT_PROMPT_curriculum §2):
while the world model is untrained, imagined rollouts poison the policy, so we train on REAL data
first and phase in imagined rollouts only as the model matures.

The schedule controls ONLY `real_ratio` — the probability that a SAC update batch is drawn from the
REAL buffer (vs the imagined/model buffer). It does NOT touch rollout length, so each algorithm
keeps its own rollout strategy intact (MACURA's adaptive-κ GJS truncation, MBPO's truncated-linear
schedule, M2AC's masking). It is applied IDENTICALLY to all three model-based agents — one shared
function, no per-algorithm branching — so it cannot bias MACURA vs MBPO vs M2AC. SAC is model-free
(already 100% real) and never uses this.

Default schedule (piecewise-linear in env-step, clamped to the first/last ratio outside the range):

    step ≤ 10k : 1.00              model matures on real data; the agent bootstraps like SAC
    10k → 20k  : 1.00 → 0.20       phase in imagined rollouts as the model becomes reliable
    20k → 30k  : 0.20 → 0.10       model-based regime: ~10–20% real / 80–90% imagined
    step ≥ 30k : 0.10              paper-ish steady state
"""
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class RealRatioSchedule:
    """Piecewise-linear `real_ratio(step)` defined by sorted (step, ratio) knots and clamped to
    the first/last ratio outside the knot range."""

    knots: List[Tuple[int, float]]

    def __post_init__(self):
        self.knots = sorted(((int(s), float(r)) for s, r in self.knots), key=lambda k: k[0])
        if not self.knots:
            raise ValueError("RealRatioSchedule needs at least one (step, real_ratio) knot")

    @classmethod
    def constant(cls, ratio: float) -> "RealRatioSchedule":
        """Degenerate schedule = the paper's fixed real_ratio (curriculum disabled)."""
        return cls([(0, float(ratio))])

    @classmethod
    def from_config(cls, cfg: dict) -> "RealRatioSchedule":
        knots = cfg.get("real_ratio_knots")
        if not knots:
            raise ValueError("curriculum config missing 'real_ratio_knots'")
        return cls([(k["step"], k["real_ratio"]) for k in knots])

    def __call__(self, step: int) -> float:
        ks = self.knots
        if step <= ks[0][0]:
            return ks[0][1]
        if step >= ks[-1][0]:
            return ks[-1][1]
        for (s0, r0), (s1, r1) in zip(ks, ks[1:]):
            if s0 <= step <= s1:
                if s1 == s0:
                    return r1
                frac = (step - s0) / (s1 - s0)
                return r0 + frac * (r1 - r0)
        return ks[-1][1]      # unreachable (clamped above), kept for safety


def real_ratio(step: int, schedule: RealRatioSchedule) -> float:
    """Shared real-data ratio at env-step `step` (see module docstring)."""
    return schedule(step)
