# nesy-highway-driving

A **transparent, model-free autonomous-driving baseline** on `highway-env`, deliberately kept
simple so that **Neuro-Symbolic (NeSy)** reasoning — explicit, formally-specified driving rules —
can be layered on top and measured against it. The car learns to **overtake traffic while staying
safe**; the symbolic layer then makes that safety *provable* and *explainable* using traffic rules
formalised in temporal logic.

This is the clean baseline counterpart to the MACURA-Drone project: where that one demonstrates a
sophisticated model-based algorithm, this one is intentionally simple and interpretable, with one
clean story — **a driving baseline ready for Neuro-Symbolic reasoning.**

## What's in this repo

| Path | What it is |
|------|-----------|
| [`README.md`](README.md) | This overview — start here. |
| [`project.md`](project.md) | The original project brief (task, goal, algorithms, NeSy idea). |
| [`ARCHITECTURE_nesy-highway-driving.md`](ARCHITECTURE_nesy-highway-driving.md) | Code structure, function-only contract, Colab workflow, `highway-env`→MetaDrive migration. |
| [`paper/`](paper/) | The temporal-logic traffic-rule papers: open-access links, citations, and full extracted notes. |
| [`nesy/RULES.md`](nesy/RULES.md) | **The rule catalog** — each rule → a predicate over the observation → tagged constraint *or* heuristic. |
| [`nesy/ROADMAP.md`](nesy/ROADMAP.md) | The five-stage plan that turns rules into a shield, then reward shaping, then differentiable logic, then explanations. |

## The idea in one paragraph

Train a car to drive and overtake safely on a highway using **standard model-free RL** (PPO,
with DQN as a second baseline), over a **discrete tactical action space** — `LANE_LEFT, IDLE,
LANE_RIGHT, FASTER, SLOWER`. The low-level controller handles steering and throttle; the agent
only picks the manoeuvre. That choice is the hinge: symbolic driving rules are naturally written
over *manoeuvres* ("don't change left unless the left gap is safe"), so a later NeSy layer can
reason in the **same vocabulary** the policy acts in — masking, shaping, or explaining its
decisions cleanly.

## Why temporal-logic traffic rules (the papers)

Rather than invent ad-hoc safety heuristics, the symbolic knowledge is taken from a well-known
line of work from TU Munich / fortiss that formalises **real traffic law** (German *StVO*, the
Vienna Convention, and court rulings) into machine-interpretable **temporal logic**, with concrete
predicates and parameter values:

- **Interstate rules** — Maierhofer et al., *IEEE IV 2020* (the paper you cited, `arnumber
  9304549`): safe distance, no abrupt braking, speed limits, traffic flow, no stopping, no
  overtaking on the right, emergency lane, considering entering vehicles. → most directly usable
  for `highway-env`.
- **Intersection rules** — Maierhofer et al., *IEEE IV 2022* (`arnumber 9827153`): stop signs,
  traffic lights, right-before-left, priority, left-turn yielding. → reserved for the MetaDrive /
  robotics extension.
- **Highway / dual-carriageway rules** — Esterle et al., *arXiv 2007.00330* (supporting,
  open-access): the cleanest rule table for highways, whose relational labels (`behind`, `left`,
  `in-front`, `near`, `safe-distance`, `lane-change`) map almost one-to-one onto the `highway-env`
  observation. → the easiest place to write the first predicates.

Full notes, abstracts, citations and open-access links are in [`paper/`](paper/).
**Note:** the IEEE Xplore PDFs are paywalled, so the binaries could not be auto-downloaded; the
open-access mediatum/arXiv versions are linked and the complete rule content has been extracted
into the `paper/*.md` notes.

## How rules become behaviour: constraints vs heuristics

Exactly as the brief suggests — *some rules as constraints, some as heuristics* — each formal rule
is tagged in [`nesy/RULES.md`](nesy/RULES.md):

- **Hard constraints (safety shield).** Rules that bound *safety* and that humans almost never
  violate (safe distance, safe lane change, speed limit, no stopping, stay on road) override the
  policy: if a chosen manoeuvre would violate the rule, it's replaced with the safest legal
  fallback.
- **Soft heuristics (reward shaping).** Comfort/efficiency rules humans routinely trade off (no
  right-passing outside exceptions, real speed advantage when overtaking, don't impede flow, don't
  brake abruptly, don't accelerate while being overtaken) become reward penalties the policy
  *learns* from.

This split isn't arbitrary — it follows the papers' own evaluation, where safety rules show
near-100% human compliance while comfort/efficiency rules are violated 20–35% of the time.

## The plan (NeSy roadmap)

0. **Neural baseline** — PPO + DQN on `highway-env`, fixed seeds, recorded metrics.
1. **Predicates** — ground the observation into truth-valued predicates.
2. **Safety shield** — enforce the hard constraints on the frozen policy (zero retraining).
3. **Logic-shaped reward** — retrain with the heuristic rules as penalties.
4. **Differentiable logic** — soft predicates / t-norms trained end-to-end with the policy.
5. **Explanation** — distil and audit rule violations with an independent temporal-logic monitor.

**Headline result to aim for:** *the same or better overtaking, with provably fewer rule
violations than the pure-neural baseline.* Details in [`nesy/ROADMAP.md`](nesy/ROADMAP.md).

## Evaluation metrics

Crash rate · off-road / on-road % · overtakes per episode · return · episode length · **plus**
per-rule violation rate. PPO vs DQN on the same seeds and the same eval, so every comparison is
fair.

## Simulator: `highway-env` now, MetaDrive later

The baseline uses `highway-env` for its simple discrete actions and compact observation. The
structure is kept simulator-agnostic so a later move to **MetaDrive** (more realistic, better for
robotics) is additive: switch the action to **velocity (linear + angular)** so it maps directly to
a robot's `cmd_vel` topic, and enable the intersection rules from the 2022 paper. See the
architecture doc for the migration details.

## Status & next step

Documentation and the rule catalog are in place. The natural next step is **Stage 0–1**: stand up
the `highway-env` baseline (PPO/DQN) and implement the predicate library from
[`nesy/RULES.md`](nesy/RULES.md), per the function-only / Colab workflow in the architecture doc.
