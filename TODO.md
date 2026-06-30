# TODO — `nesy-highway-driving`

Status tracker for the plan in [`README.md`](README.md). Parts 1 & 2 are the current focus and are
fully functional; Parts 3 & 4 are scaffolded with explicit open items.

---

## ✅ Done

### Part 1 — baseline (highway-env)
- [x] `make_env` + `read_scene` + reward shaping/overtake-counter wrappers.
- [x] `train_ppo`, `train_dqn`, `load_model`. Part-1 best is PPO (the recommended baseline), picked directly in the notebook — no selection function.
- [x] **Training-curve logging** → `metrics/curves/<algo>/progress.csv` (CSV + TensorBoard) and
      `plot_training_curves` (PPO vs DQN).
- [x] `evaluate` (crash/on-road/overtakes/return/length) + `plot_eval_comparison` bar chart.
- [x] `record_video` / `record_random_video` save a **clean MP4** (H.264 + yuv420p, frames padded to
      a multiple of 16 — no imageio `macro_block_size` warning). → `part1_best.mp4`.

### Part 2 — NeSy + XAI (highway-env)
- [x] **Step A** predicates (`nesy/roadmap.py::predicates`) on the SI scene.
- [x] **Step B** FSM safety shield (`safety_shield` + `labs/lab3_fsm.py`), zero retraining.
- [x] **Step C** logic-shaped-reward fine-tune (`finetune_logic_reward` + `LogicRewardWrapper`).
- [x] Independent MTL monitor (`rule_violations`) → per-rule violation rates (not the agent's reward).
- [x] **Shield vs no-shield XAI comparison**: four configs (baseline / +shield / +reward /
      +shield+reward), `plot_violation_rates`, and explicit **best-method selection**
      (`eval.evaluate.select_nesy_method`). → `part2_nesy.mp4`.

### Infrastructure
- [x] **NumPy 2.x fix** — removed the `numpy<2` pin (root cause of the `numpy.dtype size changed` ABI
      error); `setup_colab.sh` enforces NumPy 2.x; each notebook self-heals + restarts once.
- [x] `rich` added to `requirements.txt` (needed by `progress_bar=True`).
- [x] Docs consolidated into `README.md` + this `TODO.md` (removed `project.md`,
      `ARCHITECTURE_*.md`, `IMPLEMENTATION_PLAN.md`, `nesy/ROADMAP.md`, `nesy/RULES.md`).

---

## ⏳ Remaining

### Part 3 — MetaDrive (planned)
- [ ] Confirm the installed-MetaDrive API in `envs/metadrive_factory.py`:
      - velocity-control hook for `_velocity_to_native` (currently a proportional best-effort map);
      - neighbour enumeration in `read_scene_md` (traffic-manager attribute names);
      - offscreen-RGB / lidar config keys.
- [ ] Intersection predicates (`nesy/roadmap.py::intersection_predicates`) — wire stop-line /
      traffic-light / priority once MetaDrive exposes them; add FSM states `STOP_SIGN_WAIT`, `YIELD`,
      `LIGHT_STOP` masks in `labs/lab3_fsm.py`.
- [ ] Replace the 1-D CBF projection (`labs/lab5_cbf.py`) with the full QP once the Lab-5 API is known.
- [ ] Run Part 3 under the Python-3.10 `condacolab` path (MetaDrive doesn't build on Colab's 3.12).

### Part 4 — race (planned)
- [ ] Per-agent overtake counter in the multi-agent env (`eval/race.py::_agent_outcome` returns 0).
- [ ] Optional MetaDrive MARL race variant (Tier 2) on the Part-3 velocity + CBF/VO stack.
- [ ] Lane-assignment swap across races to cancel positional bias.

### NeSy stretch (Stages D/E)
- [ ] **Differentiable logic** — fuzzy / Łukasiewicz predicates as a smooth gradient signal
      (`predicates` already returns floats where natural).
- [ ] **Symbolic distillation** — distil the policy into a small human-readable rule set over the
      predicates, audited by the independent monitor.
- [ ] `RI3` (U-turn: needs heading/reference path) and `RI4` (emergency lane: needs lane-type map) —
      currently stubbed `False`.

---

## 📝 Notes
- **Run order:** Part 1 → Part 2 (Part 2 loads Part 1's best checkpoint from Drive). Don't start a
  part before the previous `.mp4` + checkpoint exist on Drive.
- **Private repo:** notebooks clone via a GitHub token (`GITHUB_TOKEN`); push changes before re-running
  on Colab, since Colab runs the *cloned* `.py` files, not your local edits.
- **Don't confound the comparison:** keep Part-1 shaping light; always count violations with the
  independent monitor, never the reward the agent optimises.
- **Shield intervention rate** should be low after the Step-C fine-tune — report it; a high rate means
  the policy still proposes unsafe manoeuvres the shield must veto.
