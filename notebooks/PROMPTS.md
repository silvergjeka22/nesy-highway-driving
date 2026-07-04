# Notebook rebuild prompts — `nesy-highway-driving`

Four Colab notebooks, one prompt each. They all share the **same preamble** (role, files to
read, real APIs, hard constraints, structure, verification) so the four notebooks come out
consistent. Copy the **SHARED PREAMBLE** *plus* the one **PER-NOTEBOOK** block into the coding
agent. The agent must read the listed files before writing a single cell — do not invent
function signatures; every signature below is real and lives in this repo.

> House style for all four: **explain a lot, plot a little.** Prefer prose + printed tables +
> one or two decisive figures over a wall of charts. `README.md` is the single source of truth
> for the plan; `configs/highway.yaml` is the single source of truth for every number.

---

## SHARED PREAMBLE (prepend to every notebook prompt)

### ROLE
You are a senior RL research engineer rebuilding one of the four `notebooks/colab_*.ipynb`
into a single clean, correct, top-to-bottom Colab notebook. The notebook (a) explains the idea
in plain language, (b) runs the real pipeline through this repo's `.py` modules, (c) ends by
saving an `.mp4` demo to Google Drive, and (d) hands its checkpoint to the next notebook.

### STEP 1 — READ THESE FIRST (do not skip)
Read and stay faithful to the real APIs and plan in:

* `README.md` (the whole plan, the rule catalog §4, the conventions §6), `TODO.md` (what is
  done vs stretch — do not re-solve stretch items).
* `configs/highway.yaml` — the ONE config: `env`, `shaping`, `train`, `rppo`, `dqn`, `qrdqn`,
  `eval`, `select`, `rules`, `fsm`, `finetune`, `metadrive`, `cbf`, `vo`, `race`. No
  hyperparameter is ever hard-coded in a notebook; the notebook only loads and selects.
* `utils.py` — `load_config`, `set_global_seeds`, `drive_path(cfg, key, *parts)`,
  `curve_dir(cfg, tag)`, `save_mp4(frames, path, fps)`, `save_json`.
* The papers in `paper/` when you state a rule fact: `Formalization_of_Interstate_Traffic_Rules_in_Temporal_Logic.pdf`
  (RG1–RG4, RI1–RI4; cite the rule tag) and `XAI_NeSy.pptx` (System 1/2, Kautz taxonomy,
  shielding vs reward shaping — cite the slide range from README §2).

Then read only the modules that notebook actually uses (listed in its block below).

### REAL SIGNATURES YOU MUST CALL (do not rewrite them into the notebook)
Every quantity comes from `cfg = load_config("configs/highway.yaml")`.

```python
# envs/highway_factory.py
env = create_environment(cfg, render=False, seed=None, logic_reward=False)
scene = read_scene(env)                         # SI-unit ego + neighbours dict

# agents/baselines.py
model = train_rppo(cfg, path=None)              # also train_dqn, train_qrdqn
model = load_model(path, algo)                  # algo in {"rppo","dqn","qrdqn"}
pred  = as_predictor(model, algo)               # unifies recurrent vs feed-forward .predict
model = finetune_logic_reward(model, cfg, drive_dir=None)   # Part 2, warm-start

# eval/evaluate.py
res = evaluate(model, cfg, seeds=None, apply_shield=False, count_violations=False)
      # -> {"summary": {..., "rule_violation_rate": {...}}, "episodes": [...], "seeds": [...]}
best = select_nesy_method(metrics_by_name, cfg, baseline_key=None)   # Part 2 pick

# eval/plots.py   (use SPARINGLY — see house style)
plot_training_curves(cfg, tags=("rppo","dqn","qrdqn"))
plot_eval_comparison(metrics_by_name, cfg=None, save_as=None)
plot_violation_rates(metrics_by_name, cfg=None, save_as="violation_rates.png")

# eval/race.py   (Part 4)
out = race(model_a, model_b, cfg, seeds=None)                 # scorecard dict
record_race_video(model_a, model_b, cfg, path, seed=None)

# nesy/roadmap.py
preds = predicates(scene, cfg)
act, fsm = safety_shield(action, preds, fsm_state, cfg)
viol = rule_violations(preds, cfg)
rule_encoding_agreement(scene, cfg)             # "one rule, three encodings" check

# envs/metadrive_factory.py   (Part 3)
env = make_env_md(cfg, render=False, seed=None)
obs = read_kin_obs_md(env, cfg); scene = read_scene_md(env)
act, fsm = nesy_md_action(part2_model, env, cfg, fsm_state, shield=True)
```

Videos are recorded by the **standalone scripts** run as a subprocess, never inline
(pygame/panda3d must never crash the kernel):
`demo/demo.py --model <ckpt> --algo <a> [--shield] --out <mp4>` and
`demo/demo_md.py` (MetaDrive 3D, GPU; auto top-down fallback on CPU).

### HARD CONSTRAINTS (identical across the four)
* **Function-only `.py`, notebook is the only orchestrator.** If a fix or new behaviour is
  needed, put the logic in the relevant module (`envs/`, `agents/`, `eval/`, `nesy/`, `demo/`),
  add a one-line note in the notebook saying what changed and why. Never bury logic in a cell.
* **One config.** Deep-select from `configs/highway.yaml`; the fast-vs-real switch is the single
  `train.total_timesteps` (≈3k sanity / 30k real) — one number moves all algorithms.
* **`SMOKE` flag.** `SMOKE = True` → tiny budget + few seeds so the whole notebook runs
  end-to-end in minutes; `SMOKE = False` → the real run. The notebook must execute top-to-bottom
  under `SMOKE = True` with zero errors before handoff.
* **Idempotent / resumable.** Skip a run if its checkpoint/metrics already exist in Drive
  (compute is the Colab bottleneck).
* **Fixed seeds, shared eval seeds** (`cfg["eval_seeds"]`) across every algorithm and part, so
  every comparison is fair.
* **Keep the working bootstrap** — mount Drive → clone the private repo (GitHub token prompt) →
  `pip install -r requirements.txt` → xvfb/`MUJOCO_GL`-style render setup → `load_config`. Clean
  up its narration; don't break it. Warnings are fixed at the source — **no `filterwarnings`**.
* **End by saving the `.mp4` to Drive** under `.../nesy-highway-driving/videos/`, plus the
  checkpoint under `.../checkpoints/`. Print the exact Drive paths and confirm the files landed.
* **Plot a little.** Only the one or two figures that prove the notebook's claim; everything else
  is a printed mean±std table. Use the per-algorithm colours already in `eval/plots.py`.

### NOTEBOOK STYLE
Every code cell is preceded by a short markdown cell in plain prose (2–4 sentences): why the
cell exists and what to look for in its output. Assume RL basics, not this project. Any paper
claim is cited to a rule tag (RG1…RI4) or a lecture slide range. State the claim a figure tests
in its caption, and give a one-line takeaway under it.

### VERIFICATION BEFORE HANDOFF (do this, report the result)
1. Run the whole notebook with `SMOKE = True` end-to-end — zero errors, every figure renders,
   the `.mp4` is written to Drive and its path printed.
2. Confirm the checkpoint the next notebook expects is saved to Drive with the right filename.
3. Print a short "what to run next / known gaps" list tied to `TODO.md` (do not silently expand
   scope into stretch items).

---

## PER-NOTEBOOK BLOCKS

### Notebook 1 — `colab_1_baseline.ipynb`: pick the best algorithm, save it + video to Drive
Extra reads: `agents/baselines.py`, `eval/evaluate.py`, `eval/plots.py`, `demo/demo.py`,
`envs/highway_factory.py`.

**Outcome.** Train the three model-free baselines on the aggressive-overtaking `highway-v0`
task under one shared budget, evaluate them on the shared held-out seeds, and **pick the single
best algorithm**. Save the winner and register its demo video to Drive.

Do:
1. **Study the env** briefly — the discrete meta-actions, the `Kinematics` observation, one
   rendered frame, and a short random-policy statistics print (speed / overtakes / crash rate)
   so the learning problem is concrete. Keep it to text + at most one figure.
2. **Train** `train_rppo`, `train_dqn`, `train_qrdqn` on the identical env, seeds, and the ONE
   `train.total_timesteps`. Show the live progress lines (overtakes/ep, overtakes/100 steps,
   crash %). One compact 4-panel `plot_training_curves` is the only training figure.
3. **Evaluate** all three with `evaluate(...)` on `cfg["eval_seeds"]`; print a mean±std table
   (crash rate, on-road %, overtakes/ep + diagnostics, return, length). One `plot_eval_comparison`
   bar chart, no more.
4. **Pick the winner** by the config rule in `select` (lowest crash rate, tie → most overtakes,
   among models that actually learn to pass). Save `part1_best.zip` and write `part1_best.json`
   recording which algorithm won and its metrics — **Parts 2–4 read these**.
5. **Register the video to Drive.** Run `demo/demo.py --model part1_best.zip --algo <winner>` as
   a subprocess to produce a ≥30 s `part1_best.mp4`, save it under `.../videos/`, print the Drive
   path, and confirm it is listed there. Display it inline once.

Say plainly which algorithm won and why. Do not doctor the pick to match a preferred algorithm.

---

### Notebook 2 — `colab_2_nesy.ipynb`: apply NeSy, compare, save the best model
Extra reads: `nesy/roadmap.py`, `eval/evaluate.py` (`select_nesy_method`), `eval/plots.py`,
`agents/baselines.py` (`finetune_logic_reward`, `load_model`), `labs/lab3_fsm.py`, `demo/demo.py`.

**Outcome.** Load the Part-1 winner and make it obey the traffic rules the **two ways the
lecture contrasts** — a zero-retraining **safety shield** and a **logic-shaped-reward fine-tune**
— then compare four configs on the **independent** MTL monitor and **save the best model**.

Do:
1. Load `part1_best.zip` (read `part1_best.json` for the algo). Confirm it violates the rules
   heavily — that heavy-violation baseline is the raw material NeSy fixes.
2. **Step A — predicates.** Ground the SI scene into truth-valued predicates via
   `predicates(scene, cfg)`; explain each against its rule tag (RG1…RI2) and the paper.
3. **Step B — shield.** Wrap the frozen policy in the FSM `safety_shield(...)` (zero retraining):
   an unsafe manoeuvre is replaced by the safest legal fallback. This is the lecture's shielding.
4. **Step C — logic-reward fine-tune.** `finetune_logic_reward(model, cfg, drive_dir=...)` warm-
   started from the winner (`finetune.total_timesteps`). This is the lecture's reward shaping.
5. **Compare four configs** — *baseline*, *+shield*, *+reward*, *+shield+reward* — with
   `evaluate(..., apply_shield=?, count_violations=True)`, scored by the **independent**
   `rule_violations` monitor (never the reward/shield the agent optimised). One
   `plot_violation_rates` figure + one printed per-rule table.
6. **Pick + save the best method** with `select_nesy_method(metrics_by_name, cfg)` (fewest total
   violations among configs that keep overtaking and don't worsen crashes). Save the winning
   config's checkpoint as **`part2_nesy.zip`** to Drive.
7. **Register the video to Drive.** `demo/demo.py --shield` → `part2_nesy.mp4` under `.../videos/`;
   print the path, confirm it landed, display once.

State the best method honestly; if the shield alone wins, say so.

---

### Notebook 3 — `colab_3_metadrive.ipynb`: run the model in 3D, print live velocity on the video
Extra reads: `envs/metadrive_factory.py`, `demo/demo_md.py`, `nesy/roadmap.py`
(`rule_encoding_agreement`, `continuous_shield`), `labs/lab1_cmd_vel.py`, `labs/lab4_*`,
`labs/lab5_cbf.py`, `part1_best.json`.

**Outcome.** Port the pipeline to **MetaDrive** (continuous `(v, ω)` = `cmd_vel`, 3D physics)
using the **second-best** Part-1 algorithm (the point is that the *symbolic* layer, not the
policy, provides the safety). Apply that model through the Lab-1 bridge + FSM shield + CBF/VO,
and end with a **3D chase-camera video that prints the ego velocity in real time on the frame.**

Do:
1. **Runtime note up top.** MetaDrive needs Python ≤ 3.11 → the first cell installs the
   condacolab Python-3.10 path (one kernel restart, then Run all again), GPU runtime for 3D
   (auto top-down fallback on CPU). Keep the existing working setup.
2. Load the **second-best** algorithm from Part 1 (read `part1_best.json`; if RecurrentPPO won,
   use the runner-up per `select`).
3. **The bridge:** `read_scene_md` → `read_kin_obs_md` → model picks a manoeuvre →
   `safety_shield` → Lab-1 `manoeuvre_to_cmd_vel` → **CBF (Lab 5) + velocity obstacles (Lab 4)**
   → continuous `(v, ω)` → `make_env_md` step, via `nesy_md_action(...)`. Reuse the same
   `predicates`/rules unchanged. Run the `rule_encoding_agreement(scene, cfg)` check (no MetaDrive
   needed) to show the discrete shield mask and the CBF barrier agree on `gap < safe_distance`.
4. **The video with live velocity (the thing asked for).** Record a ≥30 s 3D chase-camera clip
   with `demo/demo_md.py`. **Add a real-time telemetry overlay** so each rendered frame prints the
   ego's current speed `v` (m/s) and yaw-rate `ω` (and optionally the FSM state / whether CBF is
   clamping). Put the overlay-drawing helper in `demo/demo_md.py` (function-only; e.g. draw the
   text onto the frame before `save_mp4`), not in the notebook. Save `part3_metadrive.mp4` to
   Drive, print the path, confirm it landed, display inline.
5. Keep plots to essentially zero here — this notebook is a **3D demonstration**, not a study.

If the 3D buffer is unavailable (CPU), fall back to the top-down recorder but keep the same
live-velocity overlay.

---

### Notebook 4 — `colab_4_race.ipynb`: race with target-vs-target collisions disabled
Extra reads: `eval/race.py`, `envs/highway_factory.py`, `nesy/roadmap.py`, `configs/highway.yaml`
(`race` block), `part1_best.json`, `part2_nesy.zip`.

**Outcome.** Put both agents — **A = the plain Part-1 baseline (no shield)** and **B = the
Part-2 NeSy agent (shield on)** — in the **same highway scene** and race them: each overtakes
the background traffic and tries to get ahead of the other. Score *who gets ahead* **and** *who
stays safe / rule-compliant under competitive pressure*, never the winner alone.

New required behaviour — **the two target cars must not crash into each other.** The two
controlled vehicles should be able to overlap / pass through one another without registering a
collision, while **still crashing normally against the background (field) traffic.** Implement
this as a config-gated env option, not a notebook hack:

1. Add `race.no_target_collisions: true` to the `race` block in `configs/highway.yaml` (default
   on, with a one-line comment).
2. In `envs/highway_factory.py` (or `eval/race.py`'s `make_race_env`, whichever owns the race
   env) implement the flag so that **collision handling between the two controlled vehicles is
   suppressed** — e.g. wrap/patch highway-env's per-pair collision check so a pair is only a
   crash when at least one vehicle is *not* controlled. Controlled-vs-background collisions must
   behave exactly as before. Keep it a pure function/wrapper; add a one-line note in the
   notebook explaining the change and *why* (so a car isn't eliminated by its rival rather than
   by its own bad driving — the race measures driving skill vs the field, not bumper-cars).
3. Verify the flag: a quick assertion cell that forces the two controlled cars together and
   confirms neither reports `crashed`, while a controlled car driven into background traffic
   still does.

Then:
4. Load A (`part1_best.zip`) and B (`part2_nesy.zip`); run `race(model_a, model_b, cfg, seeds)`
   over the `race.seeds` (start slots swap on alternate seeds to cancel positional bias).
5. **Scorecard**, printed as a table (minimal plotting): per-agent progress/distance,
   overtakes vs the field, crashes against the field, per-rule violation rate (independent
   monitor), and B's shield-intervention rate. Always pair progress with crash + violation
   metrics.
6. **Register the video to Drive.** `record_race_video(model_a, model_b, cfg, path, seed=...)`
   → `part4_race.mp4` under `.../videos/`; print the path, confirm it landed, display inline.

Honest caveat to state: a "be ahead" incentive rewards aggression, so the scorecard reports
safety alongside finishing position — and with target-vs-target collisions off, any remaining
crash is genuinely the agent's own fault against the field.
