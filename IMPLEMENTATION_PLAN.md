# Implementation Plan — `nesy-highway-driving`

A single, end-to-end plan for the project: train a **pure-neural driving baseline**, then add
**Neuro-Symbolic (NeSy) traffic rules** as a fine-tuning / correction layer, then **port to
MetaDrive** for a realistic, robotics-ready setting — now explicitly built on the **course lab
toolkit** (Labs 1–5). The plan is organised into the three parts requested, with the labs woven
through each as the engineering substrate.

1. **Part 1 — Train one model *without* logical rules** (baseline + evaluation + study).
2. **Part 2 — Apply the NeSy rules as a "fine-tune"** on the trained model.
3. **Part 3 — The MetaDrive part** (realistic sim + velocity action for robotics).

It reuses the conventions already fixed in [`README.md`](README.md), [`project.md`](project.md),
and [`ARCHITECTURE_nesy-highway-driving.md`](ARCHITECTURE_nesy-highway-driving.md): function-only
`.py` modules, one config YAML, training on Colab, results mirrored to Drive, fixed seeds for fair
comparison. No code here — just *what* to build and *why*, in the order to build it.

> **Note on the labs.** The lab PDFs themselves were not available when this plan was written; the
> integration below is derived from the lab titles (Lab 1 — intro + camera follow; Lab 2 — obstacle
> avoidance with LIDAR; Lab 3 — FSM planning; Lab 4 — MCTS + velocity obstacles; Lab 5 — RL + CBF)
> and the standard content those topics carry in a robotics course. If the lab handouts/code are
> shared, the relevant sections can be tightened to match their exact APIs and parameters.

---

## 0. The story in one paragraph

We first establish a clean, reproducible **model-free RL baseline** that learns to overtake traffic
on a highway using discrete tactical meta-actions (`LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER`).
That baseline knows nothing about traffic law — it only optimises reward (this is **Lab 5's RL**
half). We then inject the **temporal-logic traffic rules** (Maierhofer et al., TU Munich / fortiss)
in two complementary ways: a **hard safety layer** — realised as a manoeuvre **shield** and, in the
continuous setting, as **Control Barrier Functions (Lab 5)** and **velocity obstacles (Lab 4)** —
and a **logic-shaped reward fine-tune** for the soft rules. The behavioural glue that decides
*which* manoeuvre to consider is a **finite-state machine (Lab 3)**; the safety substrate that keeps
the robot from hitting anything reactively is **LIDAR avoidance (Lab 2)**; and the whole thing runs
on a robot driven by velocity commands through **`cmd_vel`, set up in Lab 1**. Finally we migrate to
**MetaDrive**, switching the action interface to **velocity (linear + angular)** so the symbolic
vocabulary maps onto a real robot and intersection rules come online. The headline target across all
three parts: **equal-or-better overtaking with provably fewer rule violations than the pure-neural
baseline.**

The source rules come from the two papers in [`paper/`](paper/):

- **Interstate rules** — *IEEE IV 2020*: safe distance `RG1`, no unnecessary braking `RG2`, speed
  limit `RG3`, preserve flow `RG4`, no stopping `RI1`, no passing on the right `RI2`, no
  U-turn/reverse `RI3`, emergency-lane `RI4`. → maps onto `highway-env`.
- **Intersection rules** — *IEEE IV 2022*: stop signs, traffic lights, right-before-left, priority,
  left-turn yielding. → reserved for **Part 3 (MetaDrive)**.

Safety rules become **hard constraints** (shield / CBF / VO); comfort-efficiency rules become **soft
heuristics** (reward), matching the papers' own finding (safety rules ≈100% human compliance,
comfort rules violated 20–35%).

---

## 0.1 How the labs map onto the project (read this first)

The five labs are not a side track — they are the **toolbox** each part is built from. This mapping
is the spine of the whole plan.

| Lab | Topic | Role in this project | Used in |
|---|---|---|---|
| **Lab 1** | Intro + camera follow | ROS substrate; perception→control loop; the `cmd_vel` (velocity) interface every later stage actuates through; a minimal "follow a target" controller to sanity-check the stack. | Part 3 (robotics base), and the action-interface design throughout. |
| **Lab 2** | Obstacle avoidance with LIDAR | Reactive last-resort safety; the lidar observation MetaDrive exposes; grounds the `off_road` / `too_close` predicates from raw range data. | Part 2 (predicates), Part 3 (observation + reactive safety net). |
| **Lab 3** | FSM planning | The **symbolic manoeuvre layer**: a finite-state machine over high-level states (`CRUISE`, `FOLLOW`, `OVERTAKE-LEFT`, `MERGE`, `EMERGENCY-STOP`) that is the natural home for the temporal-logic rules and the shield's fallbacks. | Part 2 (shield structure), Part 3 (behaviour layer). |
| **Lab 4** | MCTS + velocity obstacles | **Velocity Obstacles / RVO** give the continuous, geometric version of "is this gap safe?" — directly grounding `safe_gap(left/right)` and `keeps_safe_distance`. MCTS is an optional symbolic planner that can search manoeuvre sequences the shield then validates. | Part 2 (safe-gap predicates), Part 3 (continuous safe-gap + optional planner). |
| **Lab 5** | RL + CBF | **RL** is the Part-1 baseline learner. **Control Barrier Functions** are the continuous-control analog of the discrete safety shield — a provably-safe filter on the velocity command. This is the bridge that lets the same safety guarantees survive the move from discrete meta-actions to `cmd_vel`. | Part 1 (RL baseline), Part 2 (shield ↔ CBF correspondence), Part 3 (CBF on `cmd_vel`). |

The key conceptual through-line: **a hard traffic rule has one meaning expressed three ways** — as an
MTL formula (the papers), as a discrete **manoeuvre shield** (`highway-env`, Part 2), and as a
**CBF / velocity-obstacle constraint** on the velocity command (MetaDrive / robot, Lab 4 + Lab 5,
Part 3). Building all three and showing they agree is the project's strongest result.

---

# Part 1 — Train one model *without* logical rules

**Objective:** a transparent, reproducible neural baseline plus an honest evaluation/study. No
symbolic knowledge of any kind. This is the control group every later number is measured against.
This part exercises the **RL half of Lab 5** and reuses the **camera/perception-to-control framing
of Lab 1** as the conceptual shape of the agent (observe → decide → actuate).

## 1.1 Environment

- **Simulator:** `highway-env` (`highway-v0`; `highway-fast-v0` for quick iteration).
- **Action space:** `DiscreteMetaAction` — `LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER`. The
  low-level controller handles steering/throttle; the agent only chooses the manoeuvre. This is the
  deliberate hinge for Part 2 — rules and the **Lab 3 FSM** are written over manoeuvres, not torque.
- **Observation:** `Kinematics` — ego + the *N* nearest vehicles as `[presence, x, y, vx, vy]`,
  ego-relative, `normalize: true`, fixed `vehicles_count` so the tensor shape is constant.
- **Reward:** the simulator's native reward (speed/progress + lane-keeping − collision) plus a small
  shaping term so "overtake safely" is expressible: a bonus per car passed, a penalty for leaving
  the road. Keep shaping light — heavy shaping confounds the later NeSy comparison.

All quantities live in `configs/highway.yaml` (env block, PPO block, DQN block, and a pre-reserved
`rules:` block for Part 2). Nothing hard-coded.

## 1.2 Algorithms — two standard baselines, PPO recommended

| Algorithm | Type | Why it's here |
|---|---|---|
| **PPO** | On-policy policy-gradient | Recommended best model. On-policy → no replay of stale, noisy multi-agent transitions; clipped objective tolerates reward shaping; advantage estimation gives clean credit assignment for the multi-step overtake. |
| **DQN** | Off-policy value-based | Second required baseline. More sample-efficient on discrete actions, but more brittle in noisy traffic (value overestimation). Kept for a fair contrast. |

Both use the same observation, env config, and eval seeds. (This is the **Lab 5 RL** content applied
to driving, before any CBF safety is added.)

## 1.3 Code surface (function-only)

```
envs/highway_factory.py   make_env(cfg, render=False) -> env
agents/baselines.py       train_ppo(cfg, drive_dir) -> model
                          train_dqn(cfg, drive_dir) -> model
                          load_model(path, algo) -> model
eval/evaluate.py          evaluate(model, cfg, seeds) -> metrics
                          record_video(model, cfg, path)
```

The Colab notebook mounts Drive, runs `bash/setup_colab.sh` (clone + pip install + Drive folders),
loads the YAML, calls `train_ppo`/`train_dqn`, then `evaluate` + `record_video`; everything mirrors
to Drive.

## 1.4 Evaluation — metrics

Over a fixed set of held-out seeds (identical for PPO and DQN): **crash rate**, **on-road/off-road
%**, **overtakes** per episode, **return**, **episode length**, and **training curves**. Report PPO
vs. DQN side by side, mean ± std over seeds.

## 1.5 The "study" deliverable for Part 1

1. Does PPO beat DQN here, and on which axis (safety vs. throughput)?
2. What does the baseline get *wrong*? Catalogue failure modes — cutting in with too small a gap,
   passing on the right, tail-gating, braking abruptly. **These failures motivate Part 2** and are a
   free preview of the per-rule violation rate Part 2 measures properly.
3. Sensitivity to shaping weights and `vehicles_count`.

**Exit criterion:** frozen PPO + DQN checkpoints, a metrics table, training curves, two eval videos,
and a one-page study — all reproducible from the notebook with fixed seeds.

---

# Part 2 — Apply the NeSy rules as a "fine-tune"

**Objective:** take the frozen Part-1 policy and make it obey the temporal-logic traffic rules
*without* discarding the baseline. "Fine-tune" has layers of increasing intrusiveness: a
**zero-retraining shield** (constraints), then a **reward fine-tune** (heuristics), then optional
deeper NeSy. The shield is structured as an **FSM (Lab 3)**; its safe-gap tests come from **velocity
obstacles (Lab 4)**; its reactive floor is **LIDAR avoidance (Lab 2)**; and its continuous-control
twin is a **CBF (Lab 5)**, which makes the discrete guarantee transfer to `cmd_vel` in Part 3.

## 2.1 Step A — Symbolic predicates (perception → logic)

Ground the observation into truth-valued (later fuzzy) predicates the rules are written over.

```
nesy/roadmap.py   predicates(obs, cfg) -> dict[str, bool|float]
```

| Predicate | Meaning | Source rule | Grounded via |
|---|---|---|---|
| `keeps_safe_distance(front)` | gap to leader ≥ legal safe distance | `RG1` | kinematics / **VO (Lab 4)** |
| `safe_gap(left)` / `safe_gap(right)` | target-lane gap safe for a lane change | `RG1` lane-change | **velocity obstacles (Lab 4)** |
| `too_close` / `off_road` | imminent collision / leaving lanes | road geom / `RI4` | **LIDAR ranges (Lab 2)** |
| `unnecessary_braking` | braking harder than allowed without cause | `RG2` | ego accel |
| `speed_below_min` / `over_speed_limit` | speed outside the legal band | `RG3` | ego speed vs. `v_max` |
| `impedes_flow` | ego fails to preserve flow behind slow leader | `RG4` | relative speed |
| `must_not_stop` / `in_standstill` | stopping where forbidden | `RI1` | ego speed |
| `passing_on_right` / `faster_than_left` | overtaking on the right outside exceptions | `RI2` | lateral + relative speed |

`predicates()` is pure (obs → dict), unit-testable, and reused unchanged in Part 3. Parameters
(`t_c`, `a_min`, `v_max`, `d_near`, …) come from the YAML `rules:` block, traceable to the papers.

## 2.2 Step B — Safety shield as an FSM (hard constraints, do this first, zero retraining)

Wrap the frozen Part-1 policy in a **finite-state machine (Lab 3)**. States such as `CRUISE`,
`FOLLOW`, `OVERTAKE-LEFT`, `MERGE`, `EMERGENCY-STOP` define which manoeuvres are admissible; the
policy proposes a manoeuvre, the FSM + predicates check the hard constraints, and any violation is
replaced with the safest legal fallback (`IDLE`/`SLOWER`, or refuse the lane change → drop back to
`FOLLOW`/`EMERGENCY-STOP`).

```
nesy/roadmap.py   safety_shield(action, preds, fsm_state, cfg) -> (safe_action, fsm_state)
```

Hard constraints (humans almost never break — bound *safety*):

- **`RG1` safe distance / safe gap** — block lane changes into an unsafe **velocity-obstacle (Lab 4)**
  gap; block `FASTER` when the leader gap is unsafe.
- **`RG3` speed limit** — block `FASTER` above `v_max`.
- **`RI1` no stopping** — block manoeuvres that would force a standstill where forbidden.
- **Stay on road / don't hit anything** — block off-road manoeuvres; **LIDAR (Lab 2)** is the
  reactive floor that triggers `EMERGENCY-STOP` regardless of the policy.

Why first: **no retraining**, an **immediate safety gain**, and a **clean on/off ablation** (shielded
vs. unshielded, same checkpoint, same seeds). Crash rate and `RG1`/`RG3`/`RI1` violations should drop
to ~0 with overtaking throughput largely intact.

## 2.3 Step C — Logic-shaped reward (soft heuristics, the actual "fine-tune")

Turn the **heuristic** rules into reward penalties and *continue training* the Part-1 policy on the
augmented reward (warm-start, lower LR, fewer steps — the genuine fine-tune). The policy *learns*
compliance instead of being corrected after the fact.

Soft heuristics: `RI2` (passing on the right / faster than left, outside queue/congestion
exceptions), `RG4` (impeding flow), `RG2` (abrupt braking), and accelerating while being overtaken.

```
envs/highway_factory.py   reward wrapper adds  − Σ λ_i · violation_i(preds)   from cfg.rules
agents/baselines.py       finetune_logic_reward(model, cfg, drive_dir) -> model
```

Penalty weights `λ_i` live in the YAML; ablate them to plot the throughput ↔ compliance trade-off.

## 2.4 Steps D & E — optional deeper NeSy (stretch)

- **D. Differentiable logic.** Replace hard masking with a differentiable logic module (fuzzy /
  Łukasiewicz t-norms, or a small Logic Tensor Network) so predicates become a smooth gradient signal
  trained jointly with the policy. (`predicates()` already returns floats where natural.)
- **E. Symbolic distillation / explanation.** Distil the policy into a small human-readable rule set
  over the predicates, and run an **independent MTL runtime monitor** built straight from `RG1`–`RI4`
  to audit every episode — this is what turns "fewer violations" into "*provably* fewer violations".

## 2.5 Evaluation for Part 2

Re-run the Part-1 harness on the same seeds and **add a per-rule violation rate**, counted by the
independent monitor (not by the shield/reward the agent sees — otherwise it is circular). Compare:

| Config | Crash | On-road % | Overtakes | Return | `RG1` | `RG3` | `RI1` | `RI2` | `RG4` |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (Part 1) | | | | | | | | | |
| + FSM shield (B) | | | | | | | | | |
| + Logic reward (C) | | | | | | | | | |
| + Shield + reward | | | | | | | | | |

**Exit criterion (headline result):** the shielded + fine-tuned policy achieves **equal-or-better
overtaking with provably fewer rule violations** than the pure-neural baseline.

---

# Part 3 — The MetaDrive part (and the robotics bridge)

**Objective:** move the validated pipeline to **MetaDrive** — more realistic, and the gateway to
**robotics** — reusing everything from Parts 1–2 and changing only simulator-specific pieces. This is
where the labs pay off most: the **velocity action** is the **Lab 1 `cmd_vel`** interface, the
**CBF (Lab 5)** becomes the continuous safety filter, **velocity obstacles (Lab 4)** ground safe
gaps from continuous geometry, and **LIDAR (Lab 2)** is the native MetaDrive observation.

## 3.1 What changes (three things) and what stays

### 3.1.1 Action interface — velocity (linear + angular), for the robot (Lab 1)

Set the MetaDrive action to **velocity: linear `v` + angular `ω`** rather than discrete meta-actions,
because a real robot is driven via `cmd_vel`, which expects velocity commands.

```
envs/metadrive_factory.py   make_env_md(cfg) -> env        # continuous (v, ω) action
```

`(v, ω)` maps directly to ROS `geometry_msgs/Twist` (`linear.x = v`, `angular.z = ω`) — exactly the
interface set up in **Lab 1's camera-follow** controller. The **discrete meta-actions remain the
symbolic vocabulary**: the **FSM shield (Lab 3)** still reasons over manoeuvres, and a thin layer
translates the chosen manoeuvre into a `(v, ω)` setpoint, so all of Part 2's symbolic logic is
preserved; only the actuator changes.

### 3.1.2 Safety on continuous control — CBF + velocity obstacles (Labs 4 & 5)

On `cmd_vel` the discrete shield's guarantee is re-expressed continuously:

- **Control Barrier Function (Lab 5)** — a safety filter that takes the policy's desired `(v, ω)` and
  projects it to the nearest command that keeps a safety function `h(x) ≥ 0` (safe distance, on-road).
  This is the continuous twin of the discrete shield; showing the two agree on the same scenario is
  the project's cleanest "one rule, three encodings" demonstration.
- **Velocity Obstacles / RVO (Lab 4)** — the geometric admissible-velocity set used both to ground
  `safe_gap` and as a fallback collision-avoidance layer when the CBF's model is too coarse.
- **LIDAR reactive avoidance (Lab 2)** — the last-resort floor, straight from raw ranges.

### 3.1.3 New rules — intersections (the 2022 paper)

MetaDrive has intersections, so the **intersection rules** (stop signs, traffic lights,
right-before-left, priority, left-turn yielding) come online, added to `nesy/RULES.md` as new
predicates + constraints/heuristics and handled by the *same* FSM-shield-then-reward machinery. New
FSM states appear here (`STOP-SIGN-WAIT`, `YIELD`, `LIGHT-STOP`).

### 3.1.4 What stays the same

Function-only contract; single config YAML; PPO/DQN training (PPO now over a continuous head);
Part-2 predicate library, shield, and logic-shaped reward; the evaluation metrics + independent MTL
monitor; the five-stage NeSy roadmap.

## 3.2 Migration steps

1. `make_env_md(cfg)` with a continuous `(v, ω)` action and a `metadrive:` config block.
2. **Manoeuvre → `(v, ω)`** translation so the FSM shield drives the continuous actuator (and a
   robot's `cmd_vel`).
3. A **CBF safety filter (Lab 5)** + **VO layer (Lab 4)** on the velocity command.
4. A MetaDrive **observation adapter** (kinematics + **LIDAR, Lab 2**) feeding the existing
   `predicates()`.
5. Re-train PPO/DQN; re-apply shield + CBF + logic-reward fine-tune.
6. Add **intersection** predicates/rules and extend the violation-rate evaluation.
7. (Robotics stretch) bridge the velocity setpoint to a real/simulated robot over `cmd_vel`, reusing
   the **Lab 1** stack.

## 3.3 Evaluation for Part 3

Same metric table as Part 2 (crash, on-road %, overtakes, return, per-rule violations) **plus the
intersection rules**, on MetaDrive scenarios, **plus** a CBF-intervention rate and a check that the
CBF and the discrete shield agree. The narrative: the NeSy benefit shown on `highway-env`
**transfers to a realistic simulator and a robotics velocity interface**, with safety guaranteed by
the CBF rather than only enforced by discrete masking.

---

## 4. Revised repository structure (labs included)

```
nesy-highway-driving/
├── README.md
├── project.md
├── IMPLEMENTATION_PLAN.md          # this document
├── ARCHITECTURE_nesy-highway-driving.md
├── requirements.txt
├── colab.ipynb                     # clone → train → evaluate → NeSy
├── bash/setup_colab.sh
├── configs/
│   └── highway.yaml                # env + PPO + DQN + rules{} + metadrive{} + cbf{}
├── envs/
│   ├── highway_factory.py          # make_env(cfg)              [Parts 1–2]
│   └── metadrive_factory.py        # make_env_md(cfg), (v,ω)    [Part 3]
├── agents/
│   └── baselines.py                # train_ppo/dqn, load, finetune_logic_reward   [Lab 5 RL]
├── eval/
│   └── evaluate.py                 # evaluate(), record_video()
├── nesy/
│   ├── ROADMAP.md                  # five-stage NeSy plan
│   ├── RULES.md                    # rule → predicate → constraint/heuristic
│   └── roadmap.py                  # predicates(), safety_shield()   [functions only]
├── labs/                           # the course toolkit, as reusable functions
│   ├── lab1_cmd_vel.py             # velocity/cmd_vel interface + camera-follow controller
│   ├── lab2_lidar_avoidance.py     # reactive LIDAR avoidance + range→predicate helpers
│   ├── lab3_fsm.py                 # finite-state-machine behaviour layer (shield host)
│   ├── lab4_velocity_obstacles.py  # VO/RVO safe-gap + optional MCTS manoeuvre search
│   └── lab5_cbf.py                 # Control Barrier Function safety filter on (v, ω)
└── paper/                          # temporal-logic rule papers + notes
```

Each `labs/*.py` stays **function-only** (consistent with the architecture contract) and is imported
by the shield, the predicate library, and the MetaDrive factory.

---

## 5. Sequencing, risks, and definition of done

**Order of work.** Part 1 fully (frozen checkpoints + study) → Part 2 Step A (predicates, grounded
with **Lab 2/Lab 4** helpers) → Step B (FSM shield, **Lab 3**, zero-retrain) → Step C (reward
fine-tune) → optional D/E → Part 3 (velocity action **Lab 1**, CBF **Lab 5**, VO **Lab 4**,
intersections). Never start a layer before the previous exit criterion is met, or the A/B
comparisons stop being fair.

**Key risks & mitigations.**

- *Reward shaping confounds the comparison* → keep Part-1 shaping minimal; measure violations with
  the **independent MTL monitor**, never the reward the agent optimises.
- *Predicate grounding is wrong* → unit-test `predicates()` on hand-built scenes (and on recorded
  **LIDAR** frames) before trusting any downstream number.
- *Shield too aggressive (kills throughput)* → report intervention rate; it should rarely fire once
  Step C has trained compliance in.
- *Discrete guarantee doesn't transfer to `cmd_vel`* → that is exactly what the **CBF (Lab 5)**
  solves; validate by showing CBF and discrete shield agree on shared scenarios.
- *MetaDrive continuous control is harder to train* → reuse PPO, short curriculum, keep VO + CBF as a
  safety net.

**Definition of done.** A reproducible notebook that, from fixed seeds, produces: (1) the PPO/DQN
baseline and study; (2) the four-config NeSy comparison table on `highway-env` showing equal-or-
better overtaking with provably fewer violations; (3) the same pipeline on MetaDrive with a velocity
action, a CBF safety filter, and intersection rules. Everything mirrored to Drive, every parameter in
the YAML, every rule traceable to a formula in [`paper/`](paper/) and every safety mechanism
traceable to a lab.
