# Architecture — `nesy-highway-driving`

**Goal:** a simple model-free driving baseline (PPO recommended + DQN) on `highway-env`, ready
for a later Neuro-Symbolic (NeSy) layer. Trained on **Google Colab**, results mirrored to
**Google Drive**.

## Convention
- **Every `.py` file is a library of plain functions** — no top-level execution, no `argparse`
  main. All orchestration happens in the **Colab notebook**, which imports these functions.
- **Training runs on Colab** (free GPU). The notebook clones this GitHub repo into the Colab
  workspace with a small bash script, imports the modules, and runs training.
- **Results (checkpoints, videos, metrics) mirror to Google Drive** so they survive a session
  reset.

## Colab workflow
```
GitHub repo ──(git clone via bash)──> Colab workspace ──import──> notebook runs functions
                                            │
                                            └── results ──> Google Drive (mounted)
```
1. Mount Google Drive in the notebook.
2. Run `bash/setup_colab.sh` → clone the repo into `/content/nesy-highway-driving` (or
   `git pull`), `pip install -r requirements.txt`, create the Drive results folders, print paths.
3. `import` the project's functions.
4. Call the training functions; checkpoints/videos/metrics are written to Drive.

## Structure
```
nesy-highway-driving/
├── README.md                   # project overview (start here)
├── project.md                  # the original project brief / idea
├── requirements.txt
├── colab.ipynb                 # clone → import → train PPO/DQN → evaluate → (later) NeSy
├── bash/
│   └── setup_colab.sh          # clone repo into Colab + pip install + make Drive folders
├── configs/
│   └── highway.yaml            # env + PPO + DQN hyperparameters + rule parameters
├── envs/
│   ├── highway_factory.py      # make_env(cfg) → env                    [functions only]
│   └── metadrive_factory.py    # (later) make_env_md(cfg) → env, velocity action
├── agents/
│   └── baselines.py            # train_ppo(cfg), train_dqn(cfg), load_model()
├── eval/
│   └── evaluate.py             # evaluate(model, cfg, seeds), record_video()
├── paper/                      # temporal-logic traffic-rule papers + extracted notes
│   ├── README.md               # index, citations, open-access links, BibTeX
│   ├── interstate-rules-maierhofer-2020.md
│   ├── intersection-rules-maierhofer-2022.md
│   └── highway-rules-esterle-2020.md
└── nesy/
    ├── ROADMAP.md              # staged Neuro-Symbolic plan (5 stages)
    ├── RULES.md                # rule catalog: predicates + constraint/heuristic mapping
    └── roadmap.py              # predicates(), safety_shield() stubs   [functions only]
```

## Function-only contract (examples)
- `envs/highway_factory.py`: `make_env(cfg, render) -> env`
- `agents/baselines.py`: `train_ppo(cfg, drive_dir) -> model`, `train_dqn(cfg, drive_dir) -> model`,
  `load_model(path, algo) -> model`
- `eval/evaluate.py`: `evaluate(model, cfg, seeds) -> metrics`, `record_video(model, cfg, path)`
- `nesy/roadmap.py`: `predicates(obs) -> dict`, `safety_shield(action, preds) -> safe_action`

## Notebook sections (`colab.ipynb`)
1. Mount Drive + run `setup_colab.sh`.
2. Load `configs/highway.yaml`.
3. `train_ppo(...)` and `train_dqn(...)` (save to Drive).
4. `evaluate(...)` + `record_video(...)` for both.
5. (Later) NeSy: wrap the trained model with `predicates()` + `safety_shield()` and re-evaluate.

## Conventions
- No execution in `.py` files — only definitions; the notebook drives them.
- One config YAML — all quantities there, nothing hard-coded (including the **rule parameters**
  from the papers: `t_d`, `a_min`, `v_so`, `d_near`, …; see `nesy/RULES.md`).
- Drive mirroring — every checkpoint/video/metric copied to
  `/content/drive/MyDrive/nesy-highway-driving/...`.
- Reproducibility — fixed seeds from the notebook; same eval seeds for PPO and DQN.
- The two repos share **no** code — only this structure and the Colab workflow.

## Simulator path: `highway-env` now, MetaDrive later
The baseline targets **`highway-env`** because its discrete meta-actions and compact kinematic
observation make the symbolic predicates trivial to ground (see `nesy/RULES.md`). The supervisor
suggested **MetaDrive** as a more realistic option, especially for a robotics integration. The
structure is kept simulator-agnostic so the migration is additive, not a rewrite:

- **Action interface.** For robotics, set the action to **velocity (linear + angular)** rather
  than discrete meta-actions, because a real robot is driven through the `cmd_vel` topic, which
  expects velocity commands. MetaDrive supports continuous control, so `metadrive_factory.py`
  exposes a `(v, ω)` action that maps directly to `cmd_vel` (`geometry_msgs/Twist`:
  `linear.x = v`, `angular.z = ω`). The discrete meta-actions remain the *symbolic* vocabulary —
  the shield still reasons over manoeuvres and is translated to a velocity setpoint.
- **Observation.** MetaDrive exposes ego + neighbour kinematics (and lidar), so the same
  predicate library applies; intersections additionally enable the **P2 intersection rules**
  (stop sign, traffic light, right-before-left, priority, left-turn) catalogued in `nesy/RULES.md`.
- **What stays the same.** The function-only contract, the config-YAML discipline, the
  PPO/DQN baselines, the evaluation metrics, and the five-stage NeSy roadmap are all reused; only
  `envs/` and the action mapping change.

## Where the symbolic knowledge lives
- `paper/` — the source temporal-logic rule papers (notes + open-access links).
- `nesy/RULES.md` — every rule mapped to a predicate over the observation and tagged as a hard
  **constraint** (shield) or a soft **heuristic** (reward shaping).
- `nesy/ROADMAP.md` — the staged plan that turns those rules into shield → reward → differentiable
  logic → explanation.
