# nesy-highway-driving

A small research project for the **Explainable AI · Neuro-Symbolic AI** course (Dr. Daniele Meli).

**The idea in one sentence:** a neural network learns to drive and overtake on a highway, and then
real traffic rules — taken from published papers and written as logic — are added on top to make the
car safe, and to *measure* and *explain* how safe it is.

**Neuro-Symbolic (NeSy)** means combining a **neural network** (learns from data, fast, but a black
box) with **symbolic rules** (written by humans, clear and checkable). This project does exactly that,
in three steps — one Colab notebook each.

> Every rule violation is counted by a separate, **independent checker** — never by the reward or the
> shield the car is trained with. So the numbers below are honest and can be audited. All results
> shown are the real measured results of the project.

---

## The three parts

The notebooks run in order. Each one saves its result to Google Drive, and the next one picks it up.

| Notebook | Part | What it does |
|---|---|---|
| `notebooks/colab_1_baseline.ipynb` | **Part 1** | train three algorithms to drive, pick the best |
| `notebooks/colab_2_nesy.ipynb` | **Part 2** | add the traffic rules three ways, compare them |
| `notebooks/colab_3_metadrive.ipynb` | **Part 3** | move the best driver to a 3D robot simulator, keep it safe with the labs |

---

## The main idea (from the lecture)

Two ideas from the course shape the whole project:

- **System 1 / System 2.** The trained neural network is fast, automatic thinking (System 1) — but a
  black box. The traffic rules are slow, careful thinking (System 2) — clear and checkable.
  Neuro-Symbolic AI is how we connect the two.
- **Two ways to add rules to a learner.** *Shielding* blocks an unsafe action at the last moment
  (safe, but can cost performance). *Reward shaping* teaches the rules during training (keeps
  performance, but gives no guarantee). The project implements **both** and compares them.

---

## Part 1 — teach a car to drive

The simulator is **highway-env**: a 4-lane road with traffic. Our car (the "ego"):

- **sees** the position and speed of itself and the 4 nearest cars (just numbers, not images);
- **does** one of 5 moves — left, right, faster, slower, stay — twice a second;
- **is rewarded** for speed and for passing cars, and punished for crashing.

The reward is made **aggressive on purpose**, so the car learns to drive fast and take risks. This
gives us a rule-breaking driver that Part 2 can then fix.

We train three standard reinforcement-learning algorithms and compare them fairly (same road, same
settings, same amount of training):

- **DQN** — learns how good each move is *on average*.
- **QR-DQN** — the same as DQN, but learns the *whole range* of outcomes, so it can see rare crashes.
- **RecurrentPPO** — learns the behaviour directly, and has a small memory.

**Result** (50 test runs each):

| Algorithm | Crashes | Overtakes per run | |
|---|---|---|---|
| RecurrentPPO | 100% | 3.5 | slow to learn in this test |
| DQN | 100% | 6.3 | fastest, but always crashes |
| **QR-DQN** | **78%** | **5.4** | **survives longest → winner** |

QR-DQN wins because it is the only one that "sees" the rare crash instead of hiding it inside an
average. 78% crashes is still a bad driver — on purpose; that is what Part 2 fixes.

---

## The traffic rules

The rules come from two published papers (Maierhofer et al., IEEE 2020 and 2022) that rewrite real
German/EU traffic law as **temporal logic** — logic with a sense of time, e.g. "*always* keep a safe
distance". Six rules are used:

| Rule | Plain meaning | Kind |
|---|---|---|
| **RG1** | keep a safe distance from the car ahead | hard |
| **RG3** | obey the speed limit | hard |
| **RI1** | don't stop in the middle of the road | hard |
| **RG2** | don't brake harshly for no reason | soft |
| **RG4** | don't block the traffic behind you | soft |
| **RI2** | don't overtake on the right (overtaking must be on the left) | soft |

**Hard vs soft.** Hard rules are safety rules humans almost never break, so we **forbid** breaking
them (a veto — the shield). Soft rules are comfort rules humans sometimes bend, so we **penalise**
them, and the car learns to avoid them. This split copies how the papers measured real drivers.

**Safe distance (RG1)** uses the standard stopping-distance idea: my reaction distance, plus my
braking distance, minus the distance the car ahead gains by braking. If the real gap is smaller than
that, RG1 is broken. All rule numbers live in one config file (`configs/highway.yaml`) and are
traceable to the papers.

---

## Part 2 — add the rules

Part 2 takes the trained QR-DQN and makes it obey the rules, in three ways, then compares them. First
the scene is turned into simple true/false facts (called **predicates**), e.g. "am I too close?".
Then:

- **Shield** — checks the chosen move and swaps it for a safe one if it breaks a hard rule. No
  re-training.
- **Reward shaping** — during a short re-training, the car loses reward when it breaks a soft rule,
  so it learns better habits.
- A third method, **MCTS planning**, is also included (it is used mainly in Part 3).

**Result** (same independent checker, on QR-DQN):

| Setup | Crashes | Overtakes per run | Rule-breaking |
|---|---|---|---|
| neural only | 78% | 5.4 | high |
| + shield | 12% | 3.6 | lower |
| + reward | 90% | 4.3 | medium |
| **+ shield + reward** | **24%** | **3.3** | **lowest → winner** |

The shield fixes the hard rules (safe distance, speed) with no re-training; the reward fixes the soft
habits (overtaking on the right, blocking traffic). Together they work best.

---

## The robotics labs

The safety layer is built from the course labs, connected together:

| Lab | What it gives | Used in |
|---|---|---|
| **Lab 1** | turns a move into a robot speed command (forward speed + turn rate) | Part 3 |
| **Lab 2** | LIDAR: brake for close obstacles | Part 3 |
| **Lab 3** | state machine that hosts the safety shield | Part 2 |
| **Lab 4** | MCTS planner + a "is this gap safe?" test | Parts 2 & 3 |
| **Lab 5** | the RL learner + a safety filter (Control Barrier Function) | Parts 1 & 3 |

---

## Part 3 — a new simulator

Part 3 takes the trained QR-DQN (which has never seen this world) and drops it into **MetaDrive**, a
3D simulator with realistic physics. It drives there through the labs: the model picks a move → the
shield checks it → Lab 1 turns it into a robot speed command → the safety filters (barrier function,
velocity obstacles, LIDAR) clean it → the car moves.

This tests whether the safety comes from the **rules** or from the **network**. Three setups are
compared:

| Setup | Crashes | Overtakes per run |
|---|---|---|
| brain only (no rules) | 35% | 2.15 |
| + shield | 5% | 1.40 |
| **MCTS + shield** | **5%** | **2.15 → best** |

MCTS wins here: it uses the rules *before* acting (so its moves are already legal), and it plans on
the real scene (so a world it never trained on does not confuse it). This gives the safety of the
shield **and** the overtaking of the bare model.

**One rule, three encodings.** The same safe-distance rule appears as (1) a logic fact, (2) a blocked
move in the shield, and (3) a math safety filter — and all three always agree. This shows the rule
survived the move to the new world unchanged. (This check also runs on its own, without MetaDrive.)

---

## How to run (Google Colab)

1. Open a notebook in Colab and choose **Runtime → Run all**.
2. When asked, paste a **GitHub token** (the repo is private); the setup cell clones the code and
   installs the libraries.
3. Run the parts **in order**: Part 1 → Part 2 → Part 3. Each part loads the previous part's result
   from Drive.
4. **Part 3 only:** MetaDrive needs Python ≤ 3.11, so its first cell installs a Python-3.10 runtime
   and restarts the kernel once. This is expected — just choose **Run all** again. The 3D video needs
   a GPU runtime; on CPU it uses a top-down view instead.

---

## Project structure

```
nesy-highway-driving/
├── README.md              # this file
├── TODO.md                # short status: what is done
├── requirements.txt       # Python libraries
├── configs/highway.yaml   # one config file: all settings and rule numbers
├── notebooks/             # the three Colab notebooks (the only place code runs)
├── envs/                  # build the two simulators + read the scene
├── agents/baselines.py    # train the RL algorithms, load a model
├── nesy/roadmap.py        # the rules: predicates, shield, independent checker
├── labs/                  # the five course labs
├── eval/                  # evaluation + plots
├── demo/                  # standalone scripts that record the videos
└── paper/                 # the traffic-rule papers + the lecture slides
```

**Conventions.** The `.py` files hold functions only — the notebooks are the only place code runs.
Everything is set in one config file (`configs/highway.yaml`), nothing is hard-coded. The same fixed
test runs are used for every algorithm and every setup, so all comparisons are fair.

---

## What is simplified (kept honest)

To keep the project focused, a few things are deliberately left simple or out — and are noted openly:

- Two rules from the papers (**RI3** "no U-turn" and **RI4** "keep the emergency lane clear") are
  **not** implemented — they need extra map information the simulator does not provide.
- On MetaDrive, the "no stopping" (RI1) and "no overtaking on the right" (RI2) counts look very high.
  This is a **units issue, not a real failure**: those rules' thresholds were set for highway speeds
  (~30 m/s) and the robot drives much slower (~8 m/s). The rule text is the same; only the numbers
  would need re-tuning for the slower world.
- The safety filter (Control Barrier Function) is a simple one-dimensional version, not the full
  optimisation.

---

## Papers

- **Interstate traffic rules** — Maierhofer et al., *IEEE IV 2020* — safe distance, braking, speed,
  flow, no stopping, no overtaking on the right. Used in Parts 1–2. (PDF in `paper/`.)
- **Intersection traffic rules** — Maierhofer et al., *IEEE IV 2022* — used as the basis for Part 3.
  (PDF in `paper/`.)
- **XAI · Neuro-Symbolic AI lecture** — Dr. Daniele Meli — System 1/2, shielding vs reward shaping.
  The conceptual basis of the project.
