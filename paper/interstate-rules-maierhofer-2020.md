# P1 — Interstate Traffic Rules in Temporal Logic (Maierhofer et al., 2020)

**Citation:** S. Maierhofer, A.-K. Rettinger, E. C. Mayer, M. Althoff,
"Formalization of Interstate Traffic Rules in Temporal Logic," *IEEE IV*, 2020, pp. 752–759.
DOI 10.1109/IV47402.2020.9304549 · Open PDF:
https://mediatum.ub.tum.de/doc/1552345/321027549433.pdf · IEEE: https://ieeexplore.ieee.org/document/9304549

> These notes are an extraction/condensation of the paper for engineering use in this project.
> Equations are paraphrased; see the PDF for exact formulas.

## 1. What the paper does

Formalises traffic rules for **German interstates** by combining three legal sources — the
Road Traffic Regulation (*StVO*), the Vienna Convention on Road Traffic (VCoRT), and court
decisions — and expresses them in **Metric Temporal Logic (MTL)** over finite traces. Rules
are written from the **ego vehicle's** viewpoint and evaluated by a runtime monitor on recorded
trajectories (>2,500 vehicles from the highD dataset + CommonRoad). The pipeline is: (1) extract
rules from legal sources → (2) concretise in natural language → (3) extract predicates/functions
→ (4) write MTL formulas.

## 2. MTL operators used

`G` globally (always), `F` finally (eventually), `P` previously, `O` once (in the past), with
time intervals `[lb, ub]`. Boolean `¬, ∧, ∨, ⟹`. Each rule has the shape `G(premise ⟹ conclusion)`.

## 3. The formalised rules

### General rules

| ID | Name | Plain meaning |
|----|------|---------------|
| **R_G1** | Safe distance to preceding vehicle | Keep a safe distance to the leader in the same lane so you can stop even if it brakes hard. After someone cuts in, you have `t_c` seconds to re-establish the distance. |
| **R_G2** | No unnecessary (abrupt) braking | Do not brake abruptly without reason. Legitimate reasons: no leader and accel within threshold; obstacle ahead and accel difference within threshold; or recovering a violated safe distance. |
| **R_G3** | Maximum speed limit | Do not exceed (a) the posted lane speed limit, (b) the speed at which you could still stop within the field of view, (c) the max speed for your vehicle type, (d) the speed that lets you brake comfortably for an upcoming limit. |
| **R_G4** | Preserve traffic flow | Do not drive so slowly you impede flow (unless a slow leading vehicle forces it). |

### Interstate-specific rules

| ID | Name | Plain meaning |
|----|------|---------------|
| **R_I1** | No stopping | No standstill on carriageway/shoulder/ramps unless in congestion or the leader is standing. |
| **R_I2** | No overtaking on the right ("don't drive faster than left traffic") | Do not drive faster than a vehicle to your left. Exceptions: left vehicle is in a queue/slow traffic/congestion and you are only *slightly* faster; you are separated by a broad lane marking; you are on an access ramp and the left vehicle is on the main carriageway in free flow. |
| **R_I3** | No reversing / U-turns | Reversing and U-turns are prohibited. |
| **R_I4** | Emergency lane | In congestion/slow traffic, leftmost lane keeps left, all others keep right, to open a rescue corridor (use shoulder if road too narrow). |
| **R_I5** | Consider entering vehicles | Do not move into the rightmost main-carriageway lane in front of a vehicle that is about to merge from an access ramp. |

Evaluation result (highD): only ~63% of human drivers satisfy R_G1 (safe distance) and ~78%
satisfy R_G3 (speed), but R_I1/R_I3/R_I4 are essentially always satisfied — i.e. the
"hard-constraint" rules are rarely violated by humans, the "soft" comfort/efficiency rules
often are. This directly informs which rules to use as **shields** vs **heuristics** (see
`../nesy/RULES.md`).

## 4. Key predicates (the reusable vocabulary)

Position: `in_same_lane(k,p)`, `in_front_of(k,p)`, `left_of(k,p)`, `on_main_carriageway(k)`,
`main_carriageway_right_lane(k)`, `leftmost_lane(k)`, `rightmost_lane(k)`, `single_lane(k)`,
`drives_leftmost`, `drives_rightmost`, `right_of_broad_marking(k)`, `interstate_broad_enough(k)`.

Velocity: `keeps_lane_speed_limit(k)`, `keeps_fov_speed_limit`, `keeps_type_speed_limit`,
`keeps_braking_speed_limit`, `preserves_flow(k)`, `slow_leading_vehicle(k)`, `in_standstill(k)`,
`slightly_higher_speed(k,p)`, `drives_faster(k,p)`, `reverses(k)`.

Braking: **`keeps_safe_distance_prec(k,p)`** with safe distance
`d_safe(v_k,v_p) = v_p²/(2|a_min,p|) − v_k²/(2|a_min,k|) + v_k·t_d` (assumes the leader can
brake harder than the follower), `unnecessary_braking(k)`.

General: `in_congestion`, `in_slow_moving_traffic`, `in_vehicle_queue`, `makes_u_turn`,
`cut_in(k,p)`.

## 5. Concrete parameter values (Table II — use these as defaults)

| Param | Value | Meaning |
|-------|-------|---------|
| `t_d` | 0.3 s | follower reaction time (safe distance) |
| `t_c` | 3.0 s | time to recover safe distance after a cut-in |
| `a_min,ego` | −10.0 m/s² | ego max braking |
| `a_min,o` | −10.5 m/s² | other-vehicle max braking |
| `a_max,ego` | 5.0 m/s² | ego max accel |
| `a_abrupt` | −2.0 m/s² | "abrupt" braking threshold (R_G2) |
| `v_su` | 36.66 m/s | suggested speed when no limit (~132 km/h) |
| `v_so` | 5.55 m/s | "slightly higher speed" margin (~20 km/h) |
| `Δv_fl` | 15.0 m/s | "drives slowly / impedes flow" threshold |
| `v_err` | 0.01 m/s | standstill tolerance |
| `s_fov` | 200.0 m | field of view |
| `n_con` | 3 | min vehicles to count as congestion |

## 6. How this maps into the project

R_G1 (safe distance), R_G3 (speed limit), R_I2 (no right overtaking), R_I1 (no stopping) are
the rules that translate cleanly onto `highway-env`'s kinematic observation and discrete
manoeuvres. See `../nesy/RULES.md` for the predicate-by-predicate mapping and the
constraint-vs-heuristic assignment.
