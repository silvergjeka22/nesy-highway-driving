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

Train a car to drive and overtake safely on a highway using **standard model-free RL** —
**RecurrentPPO** (PPO-LSTM), **DQN** and **QR-DQN** — over a **discrete tactical action space**
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
| [`colab_1_baseline.ipynb`](notebooks/colab_1_baseline.ipynb) | **Part 1** | study the env, train **RecurrentPPO vs DQN vs QR-DQN** to overtake aggressively, evaluate + compare, pick the **best** | `rppo.zip`/`dqn.zip`/`qrdqn.zip`/`part1_best.zip` **+ `part1_best.mp4`** + plots |
| [`colab_2_nesy.ipynb`](notebooks/colab_2_nesy.ipynb) | **Part 2 (XAI)** | load the best model, add NeSy: **predicates → shield → logic-reward**, compare the four configs, pick the best method | `part2_nesy.zip` **+ `part2_nesy.mp4`** + violation plots |
| [`colab_3_metadrive.ipynb`](notebooks/colab_3_metadrive.ipynb) | **Part 3** | run the **second-best** algorithm on **MetaDrive** via the labs (velocity action, CBF/VO) | metrics + **`part3_metadrive.mp4`** (3D) |
| [`colab_4_race.ipynb`](notebooks/colab_4_race.ipynb) | **Part 4** | **race** the NeSy agent vs the no-NeSy baseline in one scene | race scorecard **+ `part4_race.mp4`** |

### Part 1 — baseline: RecurrentPPO vs DQN vs QR-DQN, save the best (`colab_1_baseline.ipynb`)

1. **Setup** — mount Drive, clone the repo (token prompt), `pip install`, load `configs/highway.yaml`.
2. **Study & explain the environment** — `highway-v0`, the discrete meta-actions, the `Kinematics`
   observation (ego + N nearest vehicles, ego-relative, normalised), a rendered frame, and a
   random-policy statistics cell (speeds, rewards, crashes) so the learning problem is concrete.
3. **The reward is tuned for aggressive overtaking**: `normalize_reward: false` makes the native
   reward pay *speed only* (≈0 at 20 m/s, 0.5/step at 30 m/s); shaping adds `overtake_bonus` per
   car passed and a one-off `collision_penalty` per crash. Getting ahead is the only way to earn,
   crashing the only big cost — the baseline learns to pass traffic and take risks, which is
   exactly the rule-breaking raw material Part 2 needs.
4. **Train all three baselines** on the same env/seeds and the ONE shared budget
   (`train.total_timesteps` — 3k as a fast sanity check, 30k for the real run; a single number
   switches all three). The live progress lines and the training curves show **overtakes/episode
   and crash rate over time** next to the reward, so learning-to-pass is visible as it happens
   (`metrics/curves/<algo>/progress.csv`).

   | Algorithm | Type | Why |
   |---|---|---|
   | **RecurrentPPO** (PPO-LSTM, sb3-contrib) | on-policy policy-gradient | the LSTM carries intent across the identity-swapping sorted observation rows, so it commits to multi-step overtakes |
   | **DQN** | off-policy value-based | sample-efficient on discrete actions but more brittle in noisy traffic |
   | **QR-DQN** (sb3-contrib) | off-policy distributional | learns return quantiles, so the rare crash spike isn't averaged away — DQN's exact ablation twin |

5. **Evaluate + compare** on the same held-out seeds: crash rate, forward distance / steps
   survived, overtakes/episode, lane-changes/episode, on-road %, return — mean ± std, side by
   side, with a **comparison bar chart** and overtaking diagnostics (overtakes per 100 steps,
   % episodes with ≥1 pass, max passes) so a crash-shortened episode isn't mistaken for "the car
   never overtakes".
6. **Pick the winner** (most overtakes, tie → distance, among models that actually learn to pass)
   → save `part1_best.zip` + `part1_best.json` (which algorithm won — Parts 2-4 read these), then
   record the ≥30s **`part1_best.mp4`** with `demo/demo.py` (separate process, prints overtakes
   while recording).

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
  training (lower LR, `finetune.total_timesteps` ≈ 20k — enough to actually converge) on
  `reward − Σ λ_i·violation_i` over the **soft heuristics** (RI2 passing-right, RG4 impeding flow,
  RG2 abrupt braking). *This is the lecture's reward shaping.*
- **Compare four configs** — *baseline*, *+shield*, *+logic-reward*, *+shield+reward* — with per-rule
  violation rates from the **independent monitor**, plotted, and an explicit **best-method pick**
  (`eval.evaluate.select_nesy_method`: fewest total violations among configs that keep overtaking and
  don't worsen crashes). The aggressive Part-1 baseline violates the rules *heavily* — the shield
  crushes the hard-rule rows with zero retraining; the fine-tune teaches the soft ones. Save the
  NeSy checkpoint, plots, and `part2_nesy.mp4` (`demo/demo.py --shield`).

### Part 3 — MetaDrive robotics lab: the second-best algorithm, made safe by the labs

Ports the pipeline to **MetaDrive** (continuous control, 3D physics) by running a discrete Part-1
policy through the robotics labs — deliberately the **second-best** algorithm from Part 1 (DQN if
PPO won), because the showcase's point is that **the symbolic safety layer, not the policy,
provides the safety**. The bridge: `read_scene_md` (SI scene adapter) → `read_kin_obs_md`
(reconstructed highway observation) → model picks a manoeuvre → **FSM shield** → Lab-1
`manoeuvre_to_cmd_vel` → continuous **velocity `(v, ω)`** (a robot's ROS `cmd_vel`) → **CBF (Lab 5)
+ velocity obstacles (Lab 4)**. The same `predicates()` and rules are reused unchanged; the
**"one rule, three encodings" agreement check** (MTL predicate ↔ discrete shield mask ↔ CBF
barrier, all reducing to `gap < safe_distance`) **runs without MetaDrive**. Ends with a ≥30s **3D
chase-camera video** recorded offscreen (`demo/demo_md.py`, GPU runtime; auto-falls back to
top-down on CPU). MetaDrive needs Python ≤ 3.11 → the notebook installs the **condacolab
Python-3.10 runtime** (one kernel restart, then Run all again).

### Part 4 — race: NeSy vs no-NeSy (capstone)

Put both agents in the **same** multi-agent scene and let them race — each overtakes background
traffic and tries to surpass the other — scoring *who gets ahead* (distance, per-agent overtakes)
**and** *who stays safe and rule-compliant under competitive pressure* (crashes, per-rule violation
rates, shield interventions). Start slots swap on alternate seeds to cancel positional bias.
**Honest caveat:** a "be ahead" incentive rewards aggression, so the scorecard **always** pairs
finishing progress with crash + violation metrics, never the winner alone.

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
├── configs/highway.yaml          # the ONE config: env + shaping + ppo + dqn + rules{} + fsm{} + metadrive{} + cbf{} + vo{} + race{}
├── notebooks/                    # the only place code executes
│   ├── colab_1_baseline.ipynb    # Part 1  -> part1_best.mp4
│   ├── colab_2_nesy.ipynb        # Part 2  -> part2_nesy.mp4   (XAI)
│   ├── colab_3_metadrive.ipynb   # Part 3  -> part3_metadrive.mp4 (3D)
│   └── colab_4_race.ipynb        # Part 4  -> part4_race.mp4
├── envs/
│   ├── highway_factory.py        # create_environment(cfg), read_scene(env), reward wrappers [Parts 1-2,4]
│   └── metadrive_factory.py      # make_env_md(cfg), read_scene_md(env), the Lab-1 bridge     [Part 3]
├── agents/baselines.py           # train_rppo/dqn/qrdqn, load_model, finetune_logic_reward
├── eval/
│   ├── evaluate.py               # evaluate(), evaluate_nesy_md(), select_nesy_method()
│   ├── plots.py                  # plot_training_curves(), plot_eval_comparison(), plot_violation_rates()
│   └── race.py                   # make_race_env(), race(), record_race_video()
├── demo/
│   ├── demo.py                   # standalone ≥30s video: highway (Parts 1-2, 4-ready), prints overtakes + OK/FAIL
│   └── demo_md.py                # standalone ≥30s video: MetaDrive 3D / top-down (Part 3)
├── nesy/roadmap.py               # predicates(), safety_shield(), continuous_shield(), rule_violations()
├── labs/                         # lab1_cmd_vel, lab2_lidar_avoidance, lab3_fsm, lab4_velocity_obstacles, lab5_cbf
└── paper/                        # the temporal-logic traffic-rule PDFs + the XAI/NeSy lecture (.pptx)
```

**Conventions**
- **Function-only `.py` files** — no top-level execution; the four notebooks are the only orchestration.
- **One config YAML** — every quantity lives in `configs/highway.yaml` (incl. all rule parameters);
  nothing is hard-coded.
- **Colab workflow** — each notebook mounts Drive, clones the repo (GitHub token prompt), installs
  `requirements.txt`, then mirrors checkpoints/metrics/videos to
  `/content/drive/MyDrive/nesy-highway-driving/{checkpoints,metrics,videos,metrics/curves}/`.
- **Videos are standalone scripts** (`demo/demo.py`, `demo/demo_md.py`) run as a subprocess —
  pygame/panda3d can never crash a notebook kernel; every clip is ≥30 s, prints the overtakes it
  recorded and ends with `[demo] OK/FAIL`.
- **Warnings are fixed at the source, never suppressed** — no `filterwarnings('ignore')` anywhere:
  the legacy `gym` package is uninstalled (its stderr notice), pygame's `pkg_resources` import is
  blocked before gymnasium loads (its DeprecationWarning), Part 3 installs `pygame-ce` +
  `jupyter_client>=8.6.2` (their deprecation warnings).
- **Fixed seeds**, identical eval seeds across algorithms and parts → every comparison is fair.

**Evaluation metrics:** crash rate · on-road % · overtakes/episode (+ overtakes per 100 steps, %
episodes with an overtake, max overtakes) · return · episode length · **plus** per-rule violation
rate (independent monitor). All three Part-1 algorithms and every NeSy config on the same seeds.

---

## 7. How to run (Colab)

1. Open a notebook in Colab and **Runtime → Run all**.
2. Provide a **GitHub token** at the hidden prompt (the repo is private); the setup cell clones it
   and installs the dependencies.
3. Run the parts in order — each loads the previous part's checkpoints from Drive:
   **Part 1** (trains `rppo.zip`/`dqn.zip`/`qrdqn.zip`, saves `part1_best.zip` + `part1_best.json`) →
   **Part 2** (loads `part1_best`, saves `part2_nesy.zip`) → **Part 3** / **Part 4**.
4. **Part 3 only:** MetaDrive needs Python ≤ 3.11, so its first cell installs **condacolab
   (Python 3.10)** and restarts the kernel once — expected; just Run all again. The **3D video
   needs a GPU runtime** (Runtime → Change runtime type → GPU); on CPU it falls back to top-down.

---

## 8. Papers

- **Interstate rules** — Maierhofer et al., *IEEE IV 2020* — safe distance, braking, speed, flow, no
  stopping, no passing-right, emergency lane. → `highway-env` (Parts 1–2, 4). PDF in [`paper/`](paper/).
- **Intersection rules** — Maierhofer et al., *IEEE IV 2022* — stop signs, lights, right-before-left,
  priority, left-turn yielding. → MetaDrive (Part 3). PDF in [`paper/`](paper/).
- **XAI · Neurosymbolic AI lecture** — Dr. Daniele Meli (`paper/XAI_NeSy.pptx`) — System 1/2, Kautz's
  taxonomy, shielding vs reward shaping, informed exploration. The conceptual basis for §2.
