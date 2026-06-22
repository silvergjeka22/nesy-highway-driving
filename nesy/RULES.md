# Rule catalog — `nesy/RULES.md`

Each formal traffic rule → a predicate over the observation → tagged **hard
constraint** (safety shield, Part 2 Step B) or **soft heuristic** (logic-shaped
reward, Part 2 Step C). Parameters live in `configs/highway.yaml` under `rules:`
and are taken from **Maierhofer et al., *Formalization of Interstate Traffic
Rules in Temporal Logic*, IEEE IV 2020** (Table II) unless noted as
simulator-adapted.

## Interstate rules (2020 paper) — used in Parts 1–2 (`highway-env`)

| Rule | Meaning | Predicate(s) in `roadmap.py` | Tag | Params (paper) |
|---|---|---|---|---|
| **RG1** | keep a safe distance to the leader; no lane change into an unsafe gap | `keeps_safe_distance`, `safe_gap_left/right`, `too_close` | **constraint** | `t_d=0.3`, `a_min_ego=-10.0`, `a_min_other=-10.5` |
| **RG2** | no unnecessary (abrupt) braking | `unnecessary_braking` | heuristic | `a_abrupt=-2.0` |
| **RG3** | obey the speed limit | `over_speed_limit`, `speed_below_min` | **constraint** (upper), heuristic (lower) | `v_max=30.0` (sim) |
| **RG4** | preserve traffic flow behind a slow leader | `impedes_flow` | heuristic | `delta_v_fl=15.0` |
| **RI1** | no stopping where forbidden | `in_standstill`, `must_not_stop` | **constraint** | `v_err=0.01` |
| **RI2** | no passing on the right (outside queue/slow/congestion) | `passing_on_right`, `faster_than_left` | heuristic | `v_con/v_smt/v_qv`, `slightly_higher_speed` |
| **RI3** | no U-turn / reversing | `makes_uturn` (TODO: needs heading; stubbed) | **constraint** | `delta_theta_uturn=1.57` |
| **RI4** | keep the emergency lane clear in congestion | TODO (needs lane-type map; stubbed) | **constraint** | `w_road_min=7.0` |

### Safe distance (RG1)
Following the paper's `keeps_safe_distance_prec`, the legal safe gap for the ego
following a leader is the RSS-style distance

```
d_safe = v_ego * t_d + v_ego^2 / (2|a_min_ego|) - v_lead^2 / (2|a_min_other|)
```

`keeps_safe_distance` ⇔ `(leader_x - ego_x - car_length) >= d_safe`. Lane-change
safety (`safe_gap_left/right`) applies the same test to the nearest vehicle
ahead **and** behind in the target lane; in Part 3 this is replaced by the
**velocity-obstacle** test (`labs/lab4_velocity_obstacles.py`).

## Intersection rules (2022 paper) — reserved for Part 3 (MetaDrive)

Stop signs, traffic lights, right-before-left, priority, left-turn yielding.
New FSM states (`STOP_SIGN_WAIT`, `YIELD`, `LIGHT_STOP`). Predicate grounding
needs MetaDrive's intersection/traffic-light API and is **stubbed with TODOs**
in `nesy/roadmap.py` / `envs/metadrive_factory.py` until that API is wired in.

## Constraint vs heuristic (why the split)
Safety rules humans almost never break (RG1, RG3-upper, RI1, stay-on-road) →
**hard shield** (zero-retraining, ~0 violations). Comfort/efficiency rules
humans trade off 20–35 % of the time (RG2, RG4, RI2) → **soft reward penalties**
the policy *learns*. This mirrors the papers' own compliance findings.
