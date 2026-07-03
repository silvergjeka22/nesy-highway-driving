"""Dual stopping: temporal-logic rule checks on IMAGINED states (the NeSy novelty).

MACURA already stops an imagined rollout where the model is *uncertain* (GJS > κ).
Dual stopping additionally stops it where a hard traffic rule *fires* — the agent
never learns from imagined futures that break the law. Checked directly on the
batched, normalised observation (imagined states never exist in the simulator, so
the scene-dict predicates cannot be used):

  * RG1 — unsafe gap to the same-lane leader (the RSS safe-distance formula,
    identical to ``nesy.roadmap.safe_distance``).
  * RG3 — ego above the legal speed limit.

Observation layout: ``n_veh × F`` rows of [presence, x, y, vx, vy]; row 0 is the
ego (absolute), rows 1+ the neighbours (ego-relative). highway-env normalises with
MAX_SPEED = 40: x by ±5·MAX_SPEED, vx by ±2·MAX_SPEED, y by ±4·lanes.
"""
import torch

_MAX_SPEED = 40.0     # highway_env.vehicle.kinematics.Vehicle.MAX_SPEED (norm base)


@torch.no_grad()
def tl_violation_mask(states, rules, lanes_count, features_per_vehicle=5):
    """Boolean mask (B,): True where an imagined state violates RG1 or RG3."""
    F = int(features_per_vehicle)
    B, S = states.shape
    kin = states[:, :(S // F) * F].reshape(B, -1, F)
    ego, others = kin[:, 0], kin[:, 1:]

    v_ego = ego[:, 3] * (2.0 * _MAX_SPEED)                     # forward speed [m/s]
    over_limit = v_ego > float(rules["v_max"])                 # RG3

    # nearest present, same-lane leader (rel_y within half a lane width ≈ 2 m)
    present = others[:, :, 0] > 0.5
    same_lane = (others[:, :, 2].abs() * (4.0 * lanes_count)) < 2.0
    ahead = others[:, :, 1] > 0.0
    cand = present & same_lane & ahead
    rel_x = torch.where(cand, others[:, :, 1], torch.full_like(others[:, :, 1], 1e9))
    gap_norm, idx = rel_x.min(dim=1)
    has_leader = gap_norm < 1e8

    gap = gap_norm * (5.0 * _MAX_SPEED) - float(rules["car_length"])
    v_lead = (v_ego + others[torch.arange(B, device=states.device), idx, 3]
              * (2.0 * _MAX_SPEED)).clamp(min=0.0)
    d_safe = (v_ego * float(rules["t_d"])
              + v_ego ** 2 / (2.0 * abs(float(rules["a_min_ego"])))
              - v_lead ** 2 / (2.0 * abs(float(rules["a_min_other"]))))
    too_close = has_leader & (gap < d_safe)                    # RG1

    return over_limit | too_close
