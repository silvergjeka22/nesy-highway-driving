# TODO — `nesy-highway-driving`

Status tracker for the plan in [`README.md`](README.md). All three parts are implemented and
runnable end-to-end; the open items are stretch goals.

---

## ✅ Done

### Part 1 — baseline (highway-env)
- [x] `create_environment` + `read_scene` + overtake-counter / reward-shaping wrappers.
- [x] **Aggressive-overtaker reward**: `normalize_reward: false` (speed is the only earner),
      `overtake_bonus` per pass, one-off `collision_penalty`, 2 Hz decisions
      (`policy_frequency: 2` — at 1 Hz collision avoidance is unlearnable and the crash rate
      floors at ~100%), `reward_speed_range: [15, 30]` so braking to dodge still pays a little.
- [x] `train_rppo` / `train_dqn` / `train_qrdqn` (one shared `train.total_timesteps` budget —
      3k sanity / 30k real): live progress lines with
      **overtakes/ep, overtakes/100 steps and crash %**; the same numbers logged to
      `metrics/curves/<algo>/progress.csv` and plotted (4-panel training curves).
- [x] `evaluate` (crash / on-road / overtakes + diagnostics / return / length, live per-seed
      progress prints) + `plot_eval_comparison`.
- [x] Best-model pick (lowest crash rate, then most overtakes) → `part1_best.zip` +
      `part1_best.json` (Parts 2-3 read these).
- [x] `demo/demo.py`: standalone ≥30s MP4, offscreen, per-episode overtake prints, `[demo] OK/FAIL`.

### Part 2 — NeSy + XAI (highway-env)
- [x] **Step A** predicates (`nesy/roadmap.py::predicates`) on the SI scene.
- [x] **Step B** FSM safety shield (`safety_shield` + `labs/lab3_fsm.py`), zero retraining.
      RG3 is enforced actively: above the limit the shield forbids IDLE too, forcing deceleration.
- [x] **Step C** logic-shaped-reward fine-tune (`finetune_logic_reward`, 20k steps — converges).
- [x] Independent MTL monitor (`rule_violations`) counted on the **post-step** scene.
- [x] Four-config comparison (baseline / +shield / +reward / +shield+reward) over all six rules,
      `plot_violation_rates`, explicit `select_nesy_method` pick, `demo.py --shield` video.

### Part 3 — MetaDrive robotics lab
- [x] `make_env_md` (continuous `(v, ω)` action = `cmd_vel`), `read_scene_md` (same SI scene
      schema), `read_kin_obs_md` (reconstructed highway observation).
- [x] The Lab-1 bridge (`nesy_md_action`): discrete model → FSM shield → `manoeuvre_to_cmd_vel`
      → CBF (Lab 5) + velocity obstacles (Lab 4) → MetaDrive step; overtakes counted on MetaDrive.
- [x] **Showcase = the second-best Part-1 algorithm** (the safety comes from the symbolic layer).
- [x] "One rule, three encodings" agreement check (`rule_encoding_agreement`) — runs without
      MetaDrive.
- [x] **3D video fixed**: `demo/demo_md.py` uses an exact-size `RGBCamera` (the `main_camera`
      buffer follows the OS window and can come back short → shape crash) with a chase-view
      `perceive` and BGR→RGB conversion; GPU runtime for 3D, auto-fallback to top-down on CPU.
- [x] Setup: condacolab Python-3.10 primary path (one restart), GitHub-main build as the 3.12
      fallback; warning sources fixed by `pygame-ce` + `jupyter_client>=8.6.2` + removing `gym`.

### Infrastructure
- [x] **Warnings fixed at the source** everywhere; `utils.silence_warnings` deleted, no
      `filterwarnings('ignore')` in the repo. pygame's `pkg_resources` import is blocked *before*
      `import gymnasium` (gymnasium's plugin loader imports highway-env → pygame itself).
- [x] Notebooks share one setup pattern (mount → clone with token → pip install + xvfb);
      `bash/setup_colab.sh` deleted (nothing used it).
- [x] Dead code removed: `train_ppo_md`/`build_ppo_md` (fresh-PPO path), `filter_action_md`,
      `evaluate(action_filter=)`, `metadrive.ppo` config block.

---

## ⏳ Remaining (stretch)

- [ ] Intersection predicates (`nesy/roadmap.py::intersection_predicates`) — needs a MetaDrive
      intersection map + stop-line/light/priority state; FSM states `STOP_SIGN_WAIT`/`YIELD`/
      `LIGHT_STOP` masks are placeholders until then.
- [ ] Replace the 1-D CBF projection (`labs/lab5_cbf.py`) with the full QP.
- [ ] `RI3` (U-turn: needs a reference path) and `RI4` (emergency lane: needs a lane-type map).
- [ ] **Differentiable logic** — fuzzy / Łukasiewicz predicates as a smooth gradient signal.
- [ ] **Symbolic distillation** — distil the policy into a human-readable rule set over the
      predicates, audited by the independent monitor.

---

## 📝 Notes
- **Run order:** Part 1 → Part 2 → Part 3 (each loads the previous checkpoints from Drive).
- **Private repo:** notebooks clone via a GitHub token; push changes before re-running on Colab,
  since Colab runs the *cloned* `.py` files, not your local edits. The notebooks currently pull
  `BRANCH = "part3"` — switch to `main` after merging.
- **Don't confound the comparison:** always count violations with the independent monitor, never
  the reward/shield the agent optimises.
