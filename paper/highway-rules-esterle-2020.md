# P3 — Formalizing Traffic Rules for Machine Interpretability (Esterle et al., 2020)

**Citation:** K. Esterle, L. Gressenbuch, A. Knoll, "Formalizing Traffic Rules for Machine
Interpretability," *IEEE Connected and Automated Vehicles Symposium (CAVS)*, 2020.
arXiv:2007.00330 · Open PDF: https://arxiv.org/pdf/2007.00330

> Supporting paper (not one of the two you cited) but the most practical starting point for
> `highway-env`, because its relational labels map almost directly onto the kinematic observation.

## 1. What the paper does

Formalises behavioural rules for **dual carriageways** (highways) from the *StVO* / Vienna
Convention using **Linear Temporal Logic (LTL)**, with a deliberately small set of **partially
overlapping relational labels** between two agents. Rules are split into `premise ⟹ conclusion`;
exceptions fold into the premise. Evaluated with the Spot model checker on the INTERACTION dataset
inside the BARK simulator.

## 2. Relational labels (the cheap, observation-friendly vocabulary)

For ego `i` relative to another agent `j` (computed from bounding-box overlap, so they are
*partially overlapping*):

- `behind(ij)`, `in-front(ij)`, `left(ij)`, `right(ij)` — relative position.
- `near(ij)` — closer than `d_near`.
- `sd-front(i)` / `sd-rear(i)` — i keeps a safe distance to the preceding / following vehicle.
- `lane-change(i)` — i is crossing a lane boundary.
- `collide(i)`, `dense(i)`, `speed-adv(ij)` (i faster than j by ≥ `v_diff`), `acc(i)`
  (accelerates above `a_lim`).

Overtaking is treated as **passing** (behind → left → in-front sequence).

## 3. The codified rules (Table III)

| Rule | Premise ⟹ Conclusion (paraphrased) |
|------|-------------------------------------|
| **No right passing** | When not on a diverging/acceleration lane, not in dense traffic, not in a built-up non-motorway: do **not** pass on the right (no `behind → right → in-front` sequence). |
| **Safe lane change** | `lane-change(i) ⟹ sd-rear(i)` — when changing lanes, keep a safe distance to the new follower. |
| **Speed advantage for overtaking** | A passing sequence (`behind → left → in-front`) requires a real speed advantage until you are in front. |
| **Safe distance (preceding)** | Always keep a safe distance to the vehicle ahead (`G sd-front`). |
| **Being overtaken** | `right(ij) ∧ near(ij) ⟹ ¬acc(i)` — when being passed on your left, do not accelerate. |
| **Zipper merge** | At a lane end, alternate merging so the other lane's vehicle becomes your new predecessor. |

## 4. Parameter values (Section V)

`a_lim = 0.5 m/s²`, `v_diff = 10 km/h`, `r_dense = 20 m`, `N_dense = 8`,
`d_near = 5 m` (zipper) / `3 m` (being overtaken), reaction time `1 s`.
Human compliance on the dataset: ~1 lane change in 4 violates the safe rear distance during the
change; safe-distance-to-leader violated ~4–8% of the time.

## 5. Why this is the best on-ramp for the project

`highway-env` gives the ego the kinematic state of nearby vehicles, from which `behind/left/
in-front/right`, `near`, `lane-change`, and the safe-distance predicates are computable with a
few lines of geometry. The five highway rules ("safe distance", "safe lane change", "no right
passing", "speed advantage for overtaking", "being overtaken") are exactly the rules an
overtake-and-stay-safe agent needs — so the **first NeSy predicates and the first shield should
be built from this paper**, then enriched with P1's MTL formulations and parameter values.
