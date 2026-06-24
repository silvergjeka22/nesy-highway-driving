# Implementation Plan — `nesy-highway-driving`

A single, end-to-end plan organised as **four Colab notebooks**, one per part. Each notebook follows
the **same structure and conventions** (function-only `.py` modules imported by the notebook, one
config YAML, clone-from-GitHub + mount Drive, fixed seeds), and **each part ends by saving an `.mp4`
to Google Drive** that shows how the model(s) perform — the visual result for the report and the exam
discussion.

| Notebook | Part | What it does | Saved to Drive at the end |
|---|---|---|---|
| `colab_1_baseline.ipynb` | **Part 1** | Study the environment, train **PPO vs DQN**, evaluate, pick the **best** model | best checkpoint **+ `part1_best.mp4`** |
| `colab_2_nesy.ipynb` | **Part 2** | Load the best model, **fine-tune it with NeSy + labs** | NeSy checkpoint **+ `part2_nesy.mp4`** |
| `colab_3_metadrive.ipynb` | **Part 3** | Port to **MetaDrive** (velocity action, CBF/VO, intersections) | MetaDrive checkpoint **+ `part3_metadrive.mp4`** |
| `colab_4_race.ipynb` | **Part 4** | **Race** the NeSy agent vs the no-NeSy baseline | race result **+ `part4_race.mp4`** |

This plan reuses the conventions in [`README.md`](README.md), [`project.md`](project.md), and
[`ARCHITECTURE_nesy-highway-driving.md`](ARCHITECTURE_nesy-highway-driving.md). No code here — just
*what* each notebook builds and *why*, in build order.

> **Note on the labs.** The lab PDFs were not available when this plan was written; the integration
> is derived from the lab titles (Lab 1 — intro + camera follow; Lab 2 — obstacle avoidance with
> LIDAR; Lab 3 — FSM planning; Lab 4 — MCTS + velocity obstacles; Lab 5 — RL + CBF) and the standard
> content those topics carry. If the lab handouts/code are shared, the lab sections can be tightened.

---

## 0. Conventions shared by all four notebooks

- **Function-only `.py` files.** No top-level execution; the notebook imports functions and drives
  everything. Same contract as the architecture doc.
- **One config YAML** (`configs/highway.yaml`) holds every parameter — env, PPO, DQN, the `rules:`
  block (paper parameters `t_c`, `a_min`, `v_max`, `d_near`, …), the `metadrive:` block, the `cbf:`
  block, and the `race:` block. Nothing hard-coded.
- **Colab workflow (identical in every notebook):** mount Google Drive → run `bash/setup_colab.sh`
  (clone/`git pull` the repo + `pip install -r requirements.txt` + create Drive folders) → import the
  project functions → run → **mirror checkpoints, metrics, and the `.mp4` to Drive.**
- **Drive layout:**
  `/content/drive/MyDrive/nesy-highway-driving/{checkpoints,metrics,videos}/`. The `videos/` folder
  collects `part1_best.mp4`, `part2_nesy.mp4`, `part3_metadrive.mp4`, `part4_race.mp4`.
- **Fixed seeds**, identical eval seeds across algorithms and parts, so every comparison is fair.
- **Reproducibility of the chain:** Part 2 loads Part 1's best checkpoint from Drive; Part 3 starts
  from Part 2; Part 4 races Part 1's baseline against Part 2/3's NeSy agent. The Drive checkpoints are
  the hand-off between notebooks.

### Where the labs are used (the toolbox)

| Lab | Topic | Role | First used in |
|---|---|---|---|
| **Lab 1** | Intro + camera follow | `cmd_vel` / velocity interface; perception→control loop | Part 3 |
| **Lab 2** | Obstacle avoidance with LIDAR | reactive safety floor; range→predicate grounding | Part 2 |
| **Lab 3** | FSM planning | finite-state-machine behaviour layer that hosts the safety shield | Part 2 |
| **Lab 4** | MCTS + velocity obstacles | VO/RVO safe-gap grounding; optional manoeuvre search | Part 2 / Part 3 |
| **Lab 5** | RL + CBF | the RL learner (Part 1) + Control Barrier Function safety filter (Part 3) | Part 1 / Part 3 |

### The source rules (papers in [`paper/`](paper/))

- **Interstate rules** — *IEEE IV 2020*: safe distance `RG1`, no unnecessary braking `RG2`, speed
  limit `RG3`, preserve flow `RG4`, no stopping `RI1`, no passing on the right `RI2`, no
  U-turn/reverse `RI3`, emergency-lane `RI4`. → `highway-env` (Parts 1–2, 4).
- **Intersection rules** — *IEEE IV 2022*: stop signs, traffic lights, right-before-left, priority,
  left-turn yielding. → MetaDrive (Part 3).

Safety rules become **hard constraints** (shield / CBF / VO); comfort-efficiency rules become **soft
heuristics** (reward), matching the papers' finding (safety ≈100% human compliance; comfort violated
20–35%).

---

# Part 1 — Baseline: compare the two algorithms, save the best model

**Notebook:** `colab_1_baseline.ipynb` — **must be the complete, self-explanatory notebook**: it
studies and explains the environment, then trains and compares the two algorithms, then saves the
**best** model and a performance `.mp4` to Drive. This is the entry point a reader opens first.

## 1.1 Notebook contents (in order)

1. **Setup.** Mount Drive, run `setup_colab.sh`, load `configs/highway.yaml`.
2. **Study & explain the environment** (this is required — the notebook teaches the environment):
   - what `highway-env` is, the `highway-v0` scenario, and why discrete tactical actions were chosen;
   - the **action space** `LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER` and what the low-level
     controller does;
   - the **observation** (`Kinematics`: ego + N nearest vehicles, ego-relative, normalised, fixed
     `vehicles_count`), shown with a rendered frame and a printed example observation;
   - the **reward** (native speed/lane-keeping − collision, plus the light overtake/off-road shaping)
     written out explicitly;
   - a short **random-policy rollout** + rendered clip so the reader sees the task before any learning.
3. **Train both baselines** on the same env/seeds:

   | Algorithm | Type | Why |
   |---|---|---|
   | **PPO** | On-policy policy-gradient | Recommended best; on-policy avoids replaying stale noisy multi-agent transitions; clipped objective tolerates shaping; clean credit assignment for multi-step overtakes. |
   | **DQN** | Off-policy value-based | Second baseline; more sample-efficient on discrete actions but more brittle in noisy traffic. |

   (This is the **RL half of Lab 5**, before any safety is added.)
4. **Evaluate both** on the same held-out seeds and **compare**: crash rate, on-road/off-road %,
   overtakes per episode, return, episode length, plus training curves — reported mean ± std, side by
   side.
5. **Pick the best model** by a stated rule (default: lowest crash rate among models within X% of the
   top return — safety-first; the rule lives in the YAML so it is explicit, not arbitrary).
6. **Save to Drive:** the **best checkpoint** → `checkpoints/part1_best_{ppo|dqn}.zip`, the metrics
   table, the training curves, **and `videos/part1_best.mp4`** — a recorded rollout of the best model
   driving/overtaking.

## 1.2 Code surface (function-only)

```
envs/highway_factory.py   make_env(cfg, render=False) -> env
agents/baselines.py       train_ppo(cfg, drive_dir) -> model
                          train_dqn(cfg, drive_dir) -> model
                          load_model(path, algo) -> model
eval/evaluate.py          evaluate(model, cfg, seeds) -> metrics
                          select_best(metrics_ppo, metrics_dqn, cfg) -> (best_model, tag)
                          record_video(model, cfg, path)        # writes the .mp4
```

## 1.3 Exit criterion

A frozen **best** checkpoint on Drive, the PPO-vs-DQN comparison table + curves, a documented
selection rule, and **`part1_best.mp4` saved to Drive**. Part 2 will load exactly this checkpoint.

---

# Part 2 — Fine-tune the best model with NeSy + labs

**Notebook:** `colab_2_nesy.ipynb` — loads the **best Part-1 checkpoint from Drive** and makes it obey
the temporal-logic traffic rules, using the NeSy layer and the labs. It is also the **initialisation
for Part 3** (the predicates, FSM shield, and reward wrapper built here are reused there).

## 2.1 Notebook contents (in order)

1. **Setup** + load `part1_best` from Drive.
2. **Step A — Predicates (perception → logic).** Ground the observation into truth-valued (later
   fuzzy) predicates the rules are written over: `keeps_safe_distance(front)`, `safe_gap(left/right)`,
   `too_close`/`off_road`, `unnecessary_braking`, `over_speed_limit`/`speed_below_min`,
   `impedes_flow`, `must_not_stop`, `passing_on_right`. Safe-gap predicates are grounded with
   **velocity obstacles (Lab 4)**; proximity/off-road with **LIDAR helpers (Lab 2)**. Parameters from
   the YAML `rules:` block, traceable to the papers.
3. **Step B — Safety shield as an FSM (Lab 3), zero retraining.** Wrap the frozen best model in a
   finite-state machine (`CRUISE`, `FOLLOW`, `OVERTAKE-LEFT`, `MERGE`, `EMERGENCY-STOP`); if the
   proposed manoeuvre violates a **hard constraint** (`RG1` safe distance/gap, `RG3` speed limit,
   `RI1` no stopping, stay-on-road), replace it with the safest legal fallback. Immediate safety gain,
   clean on/off ablation, no training.
4. **Step C — Logic-shaped reward fine-tune (the actual fine-tune).** Warm-start from the best
   checkpoint and **continue training** on the reward augmented with **soft-heuristic** penalties
   (`RI2` passing on the right, `RG4` impeding flow, `RG2` abrupt braking, accelerating while being
   overtaken). Lower LR, fewer steps. Penalty weights `λ_i` in the YAML.
5. **(Optional, stretch) Step D/E** — differentiable logic (fuzzy t-norms / LTN) and an independent
   MTL runtime monitor for *provable* violation counting.
6. **Evaluate vs. the baseline** on the same seeds, adding a **per-rule violation rate** measured by
   the independent monitor (not the reward the agent sees). Four configs: baseline, +shield,
   +logic-reward, +shield+reward.
7. **Save to Drive:** the NeSy checkpoint → `checkpoints/part2_nesy.zip`, the comparison table, **and
   `videos/part2_nesy.mp4`** — the fine-tuned agent driving, with the shield's interventions visible.

## 2.2 Code surface (function-only)

```
nesy/roadmap.py           predicates(obs, cfg) -> dict
                          safety_shield(action, preds, fsm_state, cfg) -> (safe_action, fsm_state)
labs/lab2_lidar_avoidance.py   range helpers -> too_close/off_road
labs/lab3_fsm.py               fsm_step(state, preds, cfg) -> state
labs/lab4_velocity_obstacles.py  safe_gap(...) -> bool/float
agents/baselines.py       finetune_logic_reward(model, cfg, drive_dir) -> model
eval/evaluate.py          evaluate(model, cfg, seeds) -> metrics   # + record_video(...) -> .mp4
```

## 2.3 Exit criterion

A NeSy checkpoint on Drive, the four-config comparison (equal-or-better overtaking with provably
fewer violations than baseline), **`part2_nesy.mp4` saved to Drive**, and the predicate/shield/reward
modules ready to be imported by Part 3.

---

# Part 3 — MetaDrive (realistic sim + velocity action)

**Notebook:** `colab_3_metadrive.ipynb` — ports the validated NeSy pipeline to **MetaDrive**, the
more realistic and robotics-ready simulator, reusing Part 2's modules and changing only the
simulator-specific pieces.

## 3.1 Notebook contents (in order)

1. **Setup** + load the Part-2 NeSy checkpoint/modules from Drive.
2. **Velocity action (Lab 1).** `make_env_md(cfg)` exposes a continuous **`(v, ω)`** action (linear +
   angular), mapping directly to a robot's `cmd_vel` (`geometry_msgs/Twist`: `linear.x=v`,
   `angular.z=ω`). The discrete meta-actions stay as the **symbolic vocabulary** — the FSM shield
   still reasons over manoeuvres and a thin layer translates them to `(v, ω)`.
3. **Continuous safety (Labs 4 & 5).** A **Control Barrier Function** filter projects the policy's
   desired `(v, ω)` to the nearest safe command (`h(x) ≥ 0`: safe distance, on-road) — the continuous
   twin of the discrete shield; **velocity obstacles (Lab 4)** and **LIDAR (Lab 2)** as fallback
   layers.
4. **Observation adapter.** Feed MetaDrive's kinematics + LIDAR into the **same** `predicates()`.
5. **Intersection rules (2022 paper).** Add stop signs, traffic lights, right-before-left, priority,
   left-turn yielding as new predicates + FSM states (`STOP-SIGN-WAIT`, `YIELD`, `LIGHT-STOP`).
6. **Train / fine-tune** (PPO over a continuous head) and **evaluate** with the same metrics + a
   CBF-intervention rate + a check that CBF and discrete shield agree.
7. **Save to Drive:** the MetaDrive checkpoint → `checkpoints/part3_metadrive.zip`, metrics, **and
   `videos/part3_metadrive.mp4`** — the agent driving in MetaDrive (incl. an intersection).

## 3.2 What stays vs. changes

Reused: function-only contract, the YAML, PPO/DQN, the Part-2 predicates/shield/logic-reward, the
metrics + independent monitor. Changed: `envs/metadrive_factory.py`, the manoeuvre→`(v,ω)` mapping,
the CBF/VO filter, the observation adapter, the intersection rules.

## 3.3 Exit criterion

A MetaDrive checkpoint on Drive, the metric table (incl. intersection rules + CBF agreement), and
**`part3_metadrive.mp4` saved to Drive**.

---

# Part 4 — Race: NeSy vs no-NeSy

**Notebook:** `colab_4_race.ipynb` — the capstone: put the two agents in the **same scene** and let
them **race** — each overtakes the background traffic and tries to surpass the other — scoring both
*who finishes first* **and** *who stays safe and rule-compliant under competitive pressure*.

## 4.1 The matchup

- **Agent A — no-NeSy:** the frozen Part-1 baseline (optimises reward, no traffic law).
- **Agent B — NeSy:** the Part-2 (highway-env) or Part-3 (MetaDrive) shielded + fine-tuned agent.

Same start, same background traffic, same seed; lane assignment swapped across races to cancel
positional bias. Each agent runs its **own** policy on its **own** local observation through the
**same** `predicates()` / shield / CBF — no shared weights.

## 4.2 Notebook contents (in order)

1. **Setup** + load `part1_best` (Agent A) and `part2_nesy` / `part3_metadrive` (Agent B) from Drive.
2. **Build the multi-agent race env.** `make_race_env(cfg, n_agents=2)` — MetaDrive native MARL
   (preferred) or `highway-env` with `controlled_vehicles=2`.
3. **Define the race** (in the YAML `race:` block): winner = most longitudinal progress in a fixed
   time budget (or first past a virtual finish line / MetaDrive destination); rule for **agent–agent
   collisions** (logged separately from hitting background traffic).
4. **Run N races** (fixed seeds) and score per agent: win rate / finishing position, time-to-finish /
   progress, overtakes of background traffic, crash rate (incl. agent–agent), per-rule violation rate
   (independent monitor), shield/CBF intervention rate.
5. **Save to Drive:** the race scorecard **and `videos/part4_race.mp4`** — a **side-by-side race
   video** of the two agents, the single most legible artefact for the exam.

## 4.3 Honest caveat

A pure "be ahead of the other" reward incentivises aggressive, rule-breaking driving — so the
scorecard must **always** pair finishing position with crash + violation metrics, never the winner
alone. Competitive self-play training (MARL via PettingZoo/RLlib) stays an explicit stretch goal; the
deliverable is the **evaluation-time** race using already-trained policies.

## 4.4 Exit criterion (capstone result)

In head-to-head racing, the NeSy agent finishes **as fast or faster** than the no-NeSy baseline while
recording **fewer crashes and provably fewer rule violations**, with **`part4_race.mp4` saved to
Drive** as the visual proof.

---

## 5. Repository structure

```
nesy-highway-driving/
├── README.md
├── project.md
├── IMPLEMENTATION_PLAN.md              # this document
├── ARCHITECTURE_nesy-highway-driving.md
├── requirements.txt
├── bash/setup_colab.sh                 # clone repo + pip install + make Drive folders
├── configs/
│   └── highway.yaml                    # env + PPO + DQN + rules{} + metadrive{} + cbf{} + race{}
├── notebooks/
│   ├── colab_1_baseline.ipynb          # Part 1  -> part1_best.mp4
│   ├── colab_2_nesy.ipynb              # Part 2  -> part2_nesy.mp4
│   ├── colab_3_metadrive.ipynb         # Part 3  -> part3_metadrive.mp4
│   └── colab_4_race.ipynb              # Part 4  -> part4_race.mp4
├── envs/
│   ├── highway_factory.py              # make_env(cfg)                     [Parts 1–2, 4]
│   └── metadrive_factory.py            # make_env_md(cfg), (v,ω)           [Part 3]
├── agents/
│   └── baselines.py                    # train_ppo/dqn, load, select_best, finetune_logic_reward
├── eval/
│   ├── evaluate.py                     # evaluate(), record_video()  -> .mp4
│   └── race.py                         # make_race_env(), race(), record_race_video() -> .mp4
├── nesy/
│   ├── ROADMAP.md
│   ├── RULES.md                        # rule -> predicate -> constraint/heuristic
│   └── roadmap.py                      # predicates(), safety_shield()     [functions only]
├── labs/
│   ├── lab1_cmd_vel.py                 # velocity/cmd_vel + camera-follow
│   ├── lab2_lidar_avoidance.py         # reactive LIDAR + range->predicate
│   ├── lab3_fsm.py                     # FSM behaviour layer (shield host)
│   ├── lab4_velocity_obstacles.py      # VO/RVO safe-gap + optional MCTS
│   └── lab5_cbf.py                     # CBF safety filter on (v, ω)
└── paper/                              # temporal-logic rule papers + notes
```

All `.py` files are **function-only**; the four notebooks are the only place code executes.

---

## 6. Sequencing, risks, and definition of done

**Order.** Part 1 (save best to Drive + `part1_best.mp4`) → Part 2 (load best, NeSy fine-tune, save
`part2_nesy.mp4`) → Part 3 (MetaDrive, save `part3_metadrive.mp4`) → Part 4 (race, save
`part4_race.mp4`). Each notebook loads the previous notebook's Drive checkpoint; never start a part
before the previous exit criterion (including its saved `.mp4`) is met.

**Key risks & mitigations.**

- *Reward shaping confounds the comparison* → keep Part-1 shaping light; count violations with the
  **independent MTL monitor**, never the reward the agent optimises.
- *Predicate grounding is wrong* → unit-test `predicates()` on hand-built scenes and recorded LIDAR
  frames before trusting downstream numbers.
- *Shield too aggressive (kills throughput)* → report intervention rate; it should rarely fire after
  the Step-C fine-tune.
- *Discrete guarantee doesn't transfer to `cmd_vel`* → that is what the **CBF (Lab 5)** solves;
  validate by showing CBF and discrete shield agree.
- *Race reward rewards aggression* → always pair finishing position with crash + violation metrics.
- *Video recording fails on headless Colab* → use an offscreen/virtual display for rendering so every
  part reliably writes its `.mp4` to Drive.

**Definition of done.** Four reproducible notebooks that, from fixed seeds, produce: (1) the PPO-vs-DQN
study with the best model saved to Drive; (2) the NeSy fine-tune with the four-config comparison; (3)
the MetaDrive pipeline with velocity action, CBF, and intersection rules; (4) the head-to-head race —
**and an `.mp4` saved to Drive at the end of every part** (`part1_best`, `part2_nesy`,
`part3_metadrive`, `part4_race`). Every parameter in the YAML, every rule traceable to a formula in
[`paper/`](paper/), every safety mechanism traceable to a lab.
