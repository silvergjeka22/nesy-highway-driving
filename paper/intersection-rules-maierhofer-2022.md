# P2 — Intersection Traffic Rules in Temporal Logic (Maierhofer et al., 2022)

**Citation:** S. Maierhofer, P. Moosbrugger, M. Althoff, "Formalization of Intersection
Traffic Rules in Temporal Logic," *IEEE IV*, 2022. DOI 10.1109/IV51971.2022.9827153 ·
Open PDF: https://mediatum.ub.tum.de/doc/1664592/uw2i3i5kwjh3w4ezek0qov5og.Maierhofer-2022-IV.pdf ·
IEEE: https://ieeexplore.ieee.org/document/9827153

> Extraction/condensation for engineering use. See PDF for exact formulas.

## 1. What the paper does

Extends the interstate work (P1) to **intersections**, which need far more rules because all
road-user types are present and right-of-way must be inferred. Covers three intersection types:
**signalized** (traffic lights), **traffic-sign-regulated**, and **unregulated**. Rules are in
**MTL** (now with `X` next and `S` since, and time intervals), evaluated with the Hydra runtime
monitor on the inD dataset (real intersections) plus SUMO/CommonRoad scenarios (>2,000 vehicles).
Road network is modelled with **lanelets** (CommonRoad format).

## 2. The formalised rules

| ID | Name | Plain meaning |
|----|------|---------------|
| **R-IN1** | Stop sign | At sign 206, come to a full standstill before the stop line for at least `t_slw` before entering. |
| **R-IN2** | Traffic light | Never cross on red. On yellow, do not cross if you can still stop comfortably (above accel threshold `a_pos`) before the intersection. Direction-specific (separate lights for left/straight/right). Green-arrow (sign 720) right turns are exempt. |
| **R-IN3** | Right before left | If you are to the *left* of another vehicle (by incoming lanelet) and your paths cross, only enter if you do not endanger it. Applies when priorities are equal and no lights regulate the intersection. |
| **R-IN4** | Priority | Do not enter if another vehicle with right-of-way (inferred from priority traffic signs) would be endangered. |
| **R-IN5** | Turning left | A left-turning vehicle without priority must not endanger oncoming straight/right traffic. |

The recurring conclusion is a single relational meta-predicate: **`not_endanger_intersection(ego, o)`**
— roughly "do not cause the other vehicle to brake and do not be in the conflict area when it
will be there (within `t_ib`), and don't enter for `t_ia` after it leaves."

## 3. Key building blocks

- **Conflict area:** `in_intersection_conflict_area(k,p)` — k is in p's lane but with a different
  driving direction and they approach from different incomings (i.e. their paths physically cross).
- **Priority:** `has_priority(k,p,dir_k,dir_p)` derived from a priority table over German traffic
  signs (priority road > no sign / sign 102 > yield 205 > stop 206 > green arrow), plus
  `same_priority`. `on_incoming_left_of(k,p)` encodes "right before left."
- **Lights:** `at_traffic_light(k,dir,color)`, `relevant_traffic_light(k)`,
  `braking_intersection_possible(k)` (can stop before the line without exceeding `a_pos`).
- **Stop line:** `stop_line_in_front(k)`, `passing_stop_line(k)`, `in_standstill(k)`.

## 4. Parameter values (Table VII)

| Param | Value | Meaning |
|-------|-------|---------|
| `t_ib` | 1.0 s | min time before the other vehicle enters the conflict area |
| `t_ia` | 0.5 s | wait after the other vehicle leaves |
| `t_slw` | 3.0 s | required stop duration at a stop sign |
| `d_sl` | 1.0 m | stop-line proximity |
| `d_br` | 15.0 m | "causes braking" distance |
| `a_br` | −1.0 m/s² | "causes braking" accel threshold |
| `a_pos` | −4.0 m/s² | comfortable stop threshold (yellow light) |
| `v_err` | 0.1 m/s | standstill tolerance |

## 5. Relevance to this project

`highway-env`'s default highway scenario has **no intersections**, so P2 is **not used in the
baseline**. It becomes relevant in the **MetaDrive / robotics** extension (intersections, traffic
lights, right-of-way), where the `not_endanger`, `has_priority`, and `at_traffic_light` predicates
become the symbolic layer. It is kept here as the forward-looking half of the rule library so the
NeSy design generalises beyond the highway. See `../nesy/ROADMAP.md` (MetaDrive migration).
