# nesy-highway-driving

A **transparent, model-free autonomous-driving baseline** on `highway-env`, deliberately kept simple
so that **Neuro-Symbolic (NeSy)** reasoning — explicit, formally-specified driving rules — can be
layered on top and *measured* against it. The car learns to **overtake traffic while staying safe**;
the symbolic layer then makes that safety **provable** and **explainable (XAI)** using real traffic
law formalised in temporal logic.

> **This README is the single source of truth for the whole plan.** Progress and open items live in
> [`TODO.md`](TODO.md). There are no other planning docs — everything (architecture, rule catalog,
> NeSy roadmap, XAI framing) is consolidated here.

---

## 1. The idea in one paragraph

Train a car to drive and overtake safely on a highway using **standard model-free RL** — **PPO**
(recommended) with **DQN** as a second baseline — over a **discrete tactical action space**
(`LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER`). The low-level controller handles steering/throttle;
the agent only picks the manoeuvre. That choice is the hinge: symbolic driving rules are naturally
written over *manoeuvres* ("don't change left unless the left gap is safe"), so the NeSy layer can
reason in the **same vocabulary** the policy acts in — **shielding**, **shaping**, or **explaining**
its decisions cleanly. The headline result to aim for: *the same or better overtaking, with provably
fewer rule violations than the pure-neural baseline.*

---

## 2. The XAI / Neuro-Symbolic framing (from the lecture)

This project is the practical companion to the **XAI · Neurosymbolic AI** lecture (Dr. Daniele Meli).
Two ideas from the lecture drive the design:

- **System 1 / System 2** (Kahneman). The trained neural policy is the fast, associative **System 1**
  (a black box). The temporal-logic traffic rules are the slow, inspectable **System 2**. NeSy is how
  we connect them.
- **Kautz's taxonomy.** We sit in the **Neuro[Symbolic]** regime — System 1 (the policy) is in
  control, and System 2 (logic) is invoked when needed (to veto, to penalise, to explain).

The lecture contrasts **two canonical ways to inject logic into RL**, and *this project implements and
compares both* (Part 2):

| Method | Lecture (slides) | Mechanism | Trade-off the lecture highlights |
|---|---|---|---|
| **Shielding** | *Action pruning*, 92–96 | hard: replace an unsafe manoeuvre at run time (zero retraining) | safety is **guaranteed**, but pruning *"requires perfect domain knowledge"* and can cost performance / make a goal unreachable |
| **Reward shaping** | *Logical reward*, 84–91 | soft: `reward − Σ λ·violation`, the policy *learns* to comply | keeps performance, but *"reward is not enough"* — compliance is only soft/sub-optimal |

**One rule, three encodings** (the project's headline): a single hard rule is (a) an **MTL/temporal-
logic formula** from the papers, (b) a **discrete manoeuvre shield** (Part 2), and (c) a **CBF /
velocity-obstacle** constraint on `(v, ω)` (Part 3). Showing the discrete shield and the continuous
CBF agree on the same scene is the strongest evidence the rule is encoded faithfully.

**Explainability deliverable.** Every config is scored by an **independent temporal-logic monitor**
(`nesy.roadmap.rule_violations`), *not* by the reward/shield the agent optimises — so "fewer
violations" is an honest, auditable, *simulatable* measurement, exactly the XAI point of the lecture.

---

## 3. The four parts (one Colab notebook each)

Each notebook follows the **same conventions** (function-only `.py` modules imported by the notebook,
one config YAML, clone-from-GitHub + mount Drive, fixed seeds) and **ends by saving an `.mp4` to
Drive**. The Drive checkpoints are the hand-off between notebooks.

| Notebook | Part | What it does | Saved to Drive |
|---|---|---|---|
| [`colab_1_baseline.ipynb`](notebooks/colab_1_baseline.ipynb) | **Part 1** | study the env, train **PPO vs DQN**, log training curves, evaluate + compare, pick the **best** | `part1_best_{tag}.zip` **+ `part1_best.mp4`** + plots |
| [`colab_2_nesy.ipynb`](notebooks/colab_2_nesy.ipynb) | **Part 2 (XAI)** | load the best model, add NeSy: **predicates → shield → logic-reward**, **compare shield vs no-shield**, pick best method | `part2_nesy.zip` **+ `part2_nesy.mp4`** + violation plots |
| [`colab_3_metadrive.ipynb`](notebooks/colab_3_metadrive.ipynb) | **Part 3** | port to **MetaDrive** (velocity action, CBF/VO, intersections) | `part3_metadrive.zip` **+ `part3_metadrive.mp4`** |
| [`colab_4_race.ipynb`](notebooks/colab_4_race.ipynb) | **Part 4** | **race** the NeSy agent vs the no-NeSy baseline in one scene | race scorecard **+ `part4_race.mp4`** |

> **Current focus: Parts 1 & 2** (the highway-env baseline + the NeSy/XAI layer). They are fully
> implemented and runnable end-to-end. Parts 3 & 4 are planned and scaffolded (see
> [`TODO.md`](TODO.md) for the exact open items).

### Part 1 — baseline: compare PPO vs DQN, save the best (`colab_1_baseline.ipynb`)

1. **Setup** — mount Drive, run `bash/setup_colab.sh`, load `configs/highway.yaml`.
2. **Study & explain the environment** — `highway-v0`, the discrete meta-actions, the `Kinematics`
   observation (ego + N nearest vehicles, ego-relative, normalised), the reward (native
   speed/lane-keeping − collision **plus light shaping**: small overtake bonus, small off-road
   penalty — kept light so it doesn't confound Part 2), and a random-policy clip.
3. **Train both baselines** on the same env/seeds. Training curves (`ep_rew_mean`, `ep_len_mean` vs
   timesteps) are logged to `metrics/curves/<algo>/progress.csv` and **plotted PPO-vs-DQN**.

   | Algorithm | Type | Why |
   |---|---|---|
   | **PPO** | on-policy policy-gradient | **recommended**; on-policy avoids replaying stale noisy transitions; clipped objective tolerates shaping; clean credit assignment for multi-step overtakes |
   | **DQN** | off-policy value-based | second baseline; more sample-efficient on discrete actions but more brittle in noisy traffic |

4. **Evaluate + compare** on the same held-out seeds: crash rate, on-road %, overtakes/episode,
   return, length — mean ± std, side by side, with a **comparison bar chart**.
5. **Pick the best** by the explicit YAML rule (`select:` — lowest crash rate among models within
   `within_return_pct` of the top return; safety-first).
6. **Save** the best checkpoint, the metrics, the plots, and `part1_best.mp4` to Drive.

### Part 2 — NeSy + XAI: shield vs reward shaping (`colab_2_nesy.ipynb`)

Loads the best Part-1 checkpoint and makes it obey the traffic rules, the **two ways the lecture
contrasts**, then compares them:

- **Step A — Predicates (perception → logic).** Ground the SI-unit scene into truth-valued
  predicates: `keeps_safe_distance`, `safe_gap_left/right`, `too_close`/`off_road`,
  `over_speed_limit`/`speed_below_min`, `unnecessary_braking`, `impedes_flow`, `passing_on_right`,
  `in_standstill`/`must_not_stop`. Parameters from the YAML `rules:` block, traceable to the papers.
- **Step B — Safety shield (FSM, Lab 3), zero retraining.** Wrap the frozen policy in an FSM
  (`CRUISE/FOLLOW/OVERTAKE_LEFT/MERGE/EMERGENCY_STOP`); if a manoeuvre violates a **hard constraint**
  (RG1 safe gap, RG3 speed limit, RI1 no-stop, stay-on-road) it's replaced with the safest legal
  fallback. *This is the lecture's shielding.*
- **Step C — Logic-shaped reward fine-tune.** Warm-start from the best checkpoint and continue
  training (lower LR, fewer steps) on `reward − Σ λ_i·violation_i` over the **soft heuristics**
  (RI2 passing-right, RG4 impeding flow, RG2 abrupt braking). *This is the lecture's reward shaping.*
- **Compare four configs** — *baseline*, *+shield*, *+logic-reward*, *+shield+reward* — with per-rule
  violation rates from the **independent monitor**, plotted, and an explicit **best-method pick**
  (`eval.evaluate.select_nesy_method`: fewest total violations among configs that keep overtaking and
  don't worsen crashes). Save the NeSy checkpoint, plots, and `part2_nesy.mp4`.

### Part 3 — MetaDrive (planned: realistic sim + velocity action)

Ports the validated pipeline to **MetaDrive**: a continuous **velocity `(v, ω)`** action (maps to a
robot's ROS `cmd_vel`), a **Control Barrier Function** safety filter (Lab 5) + **velocity obstacles**
(Lab 4) on the command, the **same `predicates()`** via a MetaDrive observation adapter, and
**intersection rules** (2022 paper). The discrete meta-actions stay the *symbolic vocabulary* — the
shield reasons over manoeuvres, translated to `(v, ω)`. MetaDrive-specific APIs are scaffolded with
explicit TODOs; the **CBF ↔ discrete-shield agreement check runs without MetaDrive**.

### Part 4 — race: NeSy vs no-NeSy (planned capstone)

Put both agents in the **same** multi-agent scene and let them race — each overtakes background
traffic and tries to surpass the other — scoring *who finishes first* **and** *who stays safe and
rule-compliant under competitive pressure*. **Honest caveat:** a "be ahead" incentive rewards
aggression, so the scorecard **always** pairs finishing progress with crash + violation metrics,
never the winner alone.

---

## 4. The rule catalog (papers → predicate → constraint/heuristic)

Rules come from **Maierhofer et al., *Formalization of Interstate Traffic Rules in Temporal Logic*,
IEEE IV 2020** (Table II; PDF in [`paper/`](paper/)); intersection rules from the **2022** companion
paper. Parameters live in `configs/highway.yaml` under `rules:`.

| Rule | Meaning | Predicate(s) | Tag | Key params |
|---|---|---|---|---|
| **RG1** | safe distance to leader; no lane change into an unsafe gap | `keeps_safe_distance`, `safe_gap_left/right`, `too_close` | **constraint** (shield) | `t_d=0.3`, `a_min_ego=-10.0`, `a_min_other=-10.5` |
| **RG2** | no unnecessary (abrupt) braking | `unnecessary_braking` | heuristic (reward) | `a_abrupt=-2.0` |
| **RG3** | obey the speed limit | `over_speed_limit`, `speed_below_min` | **constraint** (upper) / heuristic (lower) | `v_max=30.0` (sim) |
| **RG4** | preserve traffic flow behind a slow leader | `impedes_flow` | heuristic (reward) | `delta_v_fl=15.0` |
| **RI1** | no stopping where forbidden | `in_standstill`, `must_not_stop` | **constraint** (shield) | `v_err=0.01` |
| **RI2** | no passing on the right (outside queue/slow/congestion) | `passing_on_right` | heuristic (reward) | `v_qv`, `slightly_higher_speed` |
| **RI3** | no U-turn / reversing | `makes_uturn` *(needs heading; stubbed)* | constraint | `delta_theta_uturn=1.57` |
| **RI4** | keep the emergency lane clear in congestion | *(needs lane-type map; stubbed)* | constraint | — |

**Safe distance (RG1):** the legal RSS-style gap is
`d_safe = v_ego·t_d + v_ego²/(2|a_min_ego|) − v_lead²/(2|a_min_other|)`, and
`keeps_safe_distance ⇔ (leader_x − ego_x − car_length) ≥ d_safe`.

**Why the constraint/heuristic split** (and why shield *and* reward): safety rules humans almost never
break (RG1, RG3-upper, RI1, stay-on-road) → **hard shield** (≈0 violations, no retraining); comfort/
efficiency rules humans trade off 20–35% of the time (RG2, RG4, RI2) → **soft reward penalties** the
policy *learns*. This mirrors the papers' own human-compliance findings.

---

## 5. The labs (the robotics toolbox)

| Lab | Topic | Role in this project | Used in |
|---|---|---|---|
| **Lab 1** | intro + camera follow | `cmd_vel` / velocity interface; manoeuvre→`(v,ω)` | Part 3 |
| **Lab 2** | LIDAR obstacle avoidance | reactive safety floor; range→predicate grounding | Part 2/3 |
| **Lab 3** | FSM planning | the finite-state-machine behaviour layer that **hosts the shield** | Part 2 |
| **Lab 4** | MCTS + velocity obstacles | VO/RVO safe-gap grounding | Part 2/3 |
| **Lab 5** | RL + CBF | the RL learner (Part 1) + **Control Barrier Function** filter on `(v, ω)` (Part 3) | Part 1/3 |

---

## 6. Repository structure & conventions

```
nesy-highway-driving/
├── README.md                     # this file — the full plan
├── TODO.md                       # done / remaining / notes
├── requirements.txt
├── utils.py                      # config, seeds, Drive paths, save_mp4, curve_dir
├── bash/setup_colab.sh           # clone repo + pip install + NumPy 2.x + Drive folders
├── configs/highway.yaml          # the ONE config: env + ppo + dqn + rules{} + fsm{} + metadrive{} + cbf{} + vo{} + race{}
├── notebooks/                    # the only place code executes
│   ├── colab_1_baseline.ipynb    # Part 1  -> part1_best.mp4
│   ├── colab_2_nesy.ipynb        # Part 2  -> part2_nesy.mp4   (XAI)
│   ├── colab_3_metadrive.ipynb   # Part 3  -> part3_metadrive.mp4
│   └── colab_4_race.ipynb        # Part 4  -> part4_race.mp4
├── envs/
│   ├── highway_factory.py        # make_env(cfg), read_scene(env), reward wrappers   [Parts 1-2,4]
│   └── metadrive_factory.py      # make_env_md(cfg), read_scene_md(env), (v,ω)        [Part 3]
├── agents/baselines.py           # train_ppo/dqn, load_model, select_best, finetune_logic_reward, train_ppo_md
├── eval/
│   ├── evaluate.py               # evaluate(), record_video(), select_nesy_method()
│   ├── plots.py                  # plot_training_curves(), plot_eval_comparison(), plot_violation_rates()
│   └── race.py                   # make_race_env(), race(), record_race_video()
├── nesy/roadmap.py               # predicates(), safety_shield(), logic_penalty(), rule_violations()
├── labs/                         # lab1_cmd_vel, lab2_lidar_avoidance, lab3_fsm, lab4_velocity_obstacles, lab5_cbf
└── paper/                        # the temporal-logic traffic-rule PDFs + the XAI/NeSy lecture (.pptx)
```

**Conventions**
- **Function-only `.py` files** — no top-level execution; the four notebooks are the only orchestration.
- **One config YAML** — every quantity lives in `configs/highway.yaml` (incl. all rule parameters);
  nothing is hard-coded.
- **Colab workflow** — mount Drive → `setup_colab.sh` (clone/pull + `pip install` + Drive folders) →
  import functions → run → mirror checkpoints/metrics/videos to
  `/content/drive/MyDrive/nesy-highway-driving/{checkpoints,metrics,videos,metrics/curves}/`.
- **Fixed seeds**, identical eval seeds across algorithms and parts → every comparison is fair.

**Evaluation metrics:** crash rate · on-road % · overtakes/episode · return · episode length · **plus**
per-rule violation rate (independent monitor). PPO vs DQN and every NeSy config on the same seeds.

---

## 7. How to run (Colab)

1. Open a notebook in Colab and **Runtime → Run all**.
2. Provide a **GitHub token** (Colab `userdata` `GITHUB_TOKEN`, or the hidden prompt) — the repo is
   private; `setup_colab.sh` clones it, installs deps, and creates the Drive folders.
3. **NumPy note:** Colab ships NumPy 2.x. The setup keeps NumPy on 2.x; if a stale build was loaded,
   the first setup cell **restarts the runtime once** automatically — just run all again. (Do *not*
   pin `numpy<2`; that triggers the `numpy.dtype size changed` ABI error.)
4. Run Part 1 → Part 2 in order (Part 2 loads Part 1's best checkpoint from Drive). Parts 3–4 are
   optional/planned.

---

## 8. Papers

- **Interstate rules** — Maierhofer et al., *IEEE IV 2020* — safe distance, braking, speed, flow, no
  stopping, no passing-right, emergency lane. → `highway-env` (Parts 1–2, 4). PDF in [`paper/`](paper/).
- **Intersection rules** — Maierhofer et al., *IEEE IV 2022* — stop signs, lights, right-before-left,
  priority, left-turn yielding. → MetaDrive (Part 3). PDF in [`paper/`](paper/).
- **XAI · Neurosymbolic AI lecture** — Dr. Daniele Meli (`paper/XAI_NeSy.pptx`) — System 1/2, Kautz's
  taxonomy, shielding vs reward shaping, informed exploration. The conceptual basis for §2.
