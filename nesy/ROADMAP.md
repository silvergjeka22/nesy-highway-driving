# NeSy roadmap — `nesy/ROADMAP.md`

The five-stage plan that turns the formal rules (`RULES.md`) into behaviour. Each
stage is independently shippable and measured against the Part-1 baseline on the
same seeds, with violations counted by an **independent MTL monitor**
(`rule_violations`), never by the shield/reward the agent sees.

| Stage | What | Code | Notebook |
|---|---|---|---|
| **A. Predicates** | ground the scene into truth-valued predicates | `nesy/roadmap.py::predicates` | `colab_2_nesy` |
| **B. Safety shield** | FSM over manoeuvres enforces the hard constraints on the frozen policy (zero retraining) | `nesy/roadmap.py::safety_shield` + `labs/lab3_fsm.py` + `labs/lab4_velocity_obstacles.py` | `colab_2_nesy` |
| **C. Logic-shaped reward** | continue-train (warm start, low LR) with `- Σ λ_i · violation_i` | `nesy/roadmap.py::logic_penalty` + `envs/highway_factory.py::LogicRewardWrapper` + `agents/baselines.py::finetune_logic_reward` | `colab_2_nesy` |
| **D. Differentiable logic** *(stretch)* | fuzzy / Łukasiewicz predicates as a smooth gradient signal | TODO — `predicates` already returns floats where natural | — |
| **E. Distillation + monitor** | distil a human-readable rule set; audit every episode with the independent monitor | `rule_violations` (monitor done); distillation TODO | `colab_2_nesy` / `colab_3` |

**One rule, three encodings** (the project's headline): a hard rule is an MTL
formula (papers), a discrete **manoeuvre shield** (Part 2), and a **CBF /
velocity-obstacle** constraint on `(v, ω)` (Part 3, `labs/lab5_cbf.py` +
`labs/lab4_velocity_obstacles.py`). Showing the discrete shield and the CBF agree
on shared scenarios is the strongest result.

**Headline target:** equal-or-better overtaking with *provably* fewer rule
violations than the pure-neural baseline.
