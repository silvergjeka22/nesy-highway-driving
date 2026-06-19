# Project 2 — Baseline Autonomous Driving (no MACURA)

**A simplified, model-free autonomous-driving project that deliberately excludes MACURA, with a
roadmap to add Neuro-Symbolic (NeSy) components later.**

This document describes the **project idea only** — the task, the goal, the chosen algorithms,
and the NeSy next steps. No code.

---

## 1. The idea in one paragraph

This is the **clean baseline** counterpart to the MACURA-Drone project. Where Project 1 is about
a sophisticated model-based algorithm, Project 2 is intentionally simple: train a car to drive
and overtake safely on a highway using **standard, model-free** reinforcement learning, and
keep the design so transparent that **Neuro-Symbolic (NeSy)** reasoning — explicit driving rules
— can be layered on top afterwards. No world models, no ensembles, no MACURA. The point is a
reproducible, interpretable starting line.

## 2. The task

- **Environment:** `highway-env` — a lightweight highway driving simulator with surrounding
  traffic.
- **Action space:** **discrete tactical meta-actions** — `LANE_LEFT, IDLE, LANE_RIGHT, FASTER,
  SLOWER`. The low-level controller handles steering and throttle; the agent only chooses the
  manoeuvre. This "simplified" framing is a deliberate choice (see §5 — it is what makes NeSy
  natural later).
- **Observation:** the kinematic state of the ego and the nearest surrounding vehicles.
- **Reward:** the simulator's native driving reward (progress / speed, stay in lane, penalty for
  collisions) plus a light shaping term so "overtake safely" is expressible — a small bonus per
  car passed and a small penalty for leaving the road.

## 3. The goal

Train a vehicle that **overtakes traffic while staying safe** (no collisions, stays on the road),
using simple standard algorithms, and produce a clean, reproducible baseline that:
1. can be evaluated with clear metrics (crash rate, overtakes, on-road %, return), and
2. is structured so the **NeSy** extension can be added and measured against it.

## 4. Algorithms — two standard baselines, one recommended

The project implements exactly **two** standard, model-free baselines:

| Algorithm | Type | Strengths | Weaknesses |
|---|---|---|---|
| **PPO** (Proximal Policy Optimization) | On-policy policy-gradient | Stable under stochastic traffic and reward shaping; good long-horizon credit assignment (overtaking is multi-step) | Lower sample efficiency |
| **DQN** (Deep Q-Network) | Off-policy value-based | More sample-efficient on discrete actions | Brittle in noisy multi-agent settings (value overestimation), sensitive to reward shaping |

### Recommended single best model: **PPO**

For an *overtake-and-stay-safe* task with **stochastic surrounding traffic** and a **shaped
reward**, PPO is the more reliable baseline:
- it is **on-policy**, so it does not compound value-overestimation through a replay buffer of
  stale, noisy transitions the way DQN can;
- its **clipped objective** is tolerant of reward shaping;
- its **advantage estimation** gives cleaner credit assignment for the multi-step overtaking
  manoeuvre.

DQN is kept as the second required baseline (and is more sample-efficient), but **PPO is the
model put forward as best for this task.**

## 5. Why discrete meta-actions (and why it matters for NeSy)

Choosing the discrete tactical action space is not just a simplification for learning — it is the
hinge for the Neuro-Symbolic stage. Symbolic driving rules are naturally written over
*manoeuvres* ("do not change to the left lane unless the left gap is safe"), not over raw torque.
By learning at the manoeuvre level, the later NeSy layer can reason over the *same* vocabulary
the policy acts in, so rules can mask, shape, or explain the policy's decisions cleanly.

## 6. Evaluation — the metrics to report

- **Crash rate** (% of episodes ending in a collision).
- **Off-road / on-road %**.
- **Overtakes** (distinct cars passed per episode).
- **Return** and episode length.
- PPO vs DQN, same seeds and same evaluation, so the comparison is fair.

## 7. Next steps — Neuro-Symbolic (NeSy) roadmap

The baseline is purely neural. NeSy is added in **stages**, each independently shippable and each
measured against the baseline:

1. **Symbolic predicates (perception → logic).** Ground the continuous observation into
   truth-valued predicates rules can use: `TTC_low(front)`, `safe_gap(left)`, `safe_gap(right)`,
   `blocked_ahead`, `speed_below_min`, `off_road`.
2. **Safety shield (do this first).** Wrap the trained policy: if the chosen manoeuvre violates a
   hard rule (e.g. lane-change into an unsafe gap, accelerate while time-to-collision is low),
   replace it with the safest legal fallback. Zero retraining, immediate safety gain, clean
   on/off ablation.
3. **Logic-shaped reward (training-time).** Turn the predicates into penalties added to the
   reward, so the policy *learns* to respect the rules instead of being corrected afterwards.
4. **Differentiable logic (end-to-end NeSy).** Replace hard masking with a differentiable logic
   module (e.g. fuzzy / Łukasiewicz t-norms, or a Logic Tensor Network) trained jointly with the
   neural policy, so symbolic knowledge becomes a gradient signal.
5. **Symbolic distillation / explanation.** Distil the trained policy into a small, human-readable
   rule set over the predicates for interpretability, verified by the Stage-2 shield.

**Headline NeSy result to aim for:** *the same or better overtaking, with provably fewer rule
violations than the pure-neural baseline.*

## 8. Honest scope

- This project **excludes model-based RL on purpose** — it is the simple, interpretable baseline,
  not a place to demonstrate MACURA (that is Project 1).
- The two baselines are *standard* algorithms; the project's value is the task framing, the
  evaluation, and the NeSy roadmap — not novel RL machinery.
- Keeping it separate from the drone project means each tells one clean story: **Project 1 =
  MACURA demonstrated on an unstable continuous-control system; Project 2 = a transparent driving
  baseline ready for Neuro-Symbolic reasoning.**
