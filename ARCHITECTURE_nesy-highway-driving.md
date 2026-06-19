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
├── README.md
├── requirements.txt
├── colab.ipynb                 # clone → import → train PPO/DQN → evaluate → (later) NeSy
├── bash/
│   └── setup_colab.sh          # clone repo into Colab + pip install + make Drive folders
├── configs/
│   └── highway.yaml            # env + PPO + DQN hyperparameters
├── envs/
│   └── highway_factory.py      # make_env(cfg) → env    [functions only]
├── agents/
│   └── baselines.py            # train_ppo(cfg), train_dqn(cfg), load_model()
├── eval/
│   └── evaluate.py             # evaluate(model, cfg, seeds), record_video()
└── nesy/
    ├── ROADMAP.md              # staged Neuro-Symbolic plan
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
- One config YAML — all quantities there, nothing hard-coded.
- Drive mirroring — every checkpoint/video/metric copied to
  `/content/drive/MyDrive/nesy-highway-driving/...`.
- Reproducibility — fixed seeds from the notebook; same eval seeds for PPO and DQN.
- The two repos share **no** code — only this structure and the Colab workflow.
