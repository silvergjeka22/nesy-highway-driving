# Status — what is done

All three parts are finished and run end to end in Google Colab. See [`README.md`](README.md) for the
full explanation and the results. Short summary of the completed work:

## Part 1 — teach a car to drive (highway-env)
- Built the driving environment and an aggressive-overtaking reward (fast driving + passing scores,
  crashing is punished).
- Trained and compared three algorithms — PPO, DQN, QR-DQN — on equal terms.
- Evaluated them on the same test runs, with training curves and comparison plots.
- Picked the best driver (**QR-DQN**) and recorded a demo video.

## Part 2 — add the traffic rules (highway-env)
- Turned the scene into simple true/false facts (predicates) for six traffic rules.
- **Safety shield** (a state machine) that blocks unsafe moves — no re-training needed.
- **Reward shaping** that teaches the soft rules during a short re-training.
- An **independent checker** that counts every rule violation, separate from what the car is trained
  on.
- Compared four setups (neural only / + shield / + reward / + shield + reward), plotted the
  rule-breaking, and picked the best (**shield + reward**). Recorded a demo video.

## Part 3 — a new simulator (MetaDrive)
- A bridge that lets the trained driver control a 3D robot through the labs: move → shield → speed
  command (Lab 1) → safety filters (Control Barrier Function, velocity obstacles, LIDAR).
- Compared three setups — brain only / + shield / **MCTS + shield** — and recorded a video of each.
- The "one rule, three encodings" check: the same safe-distance rule as a logic fact, a blocked move,
  and a math filter — all agree.

## Not included (on purpose)
- The full-optimisation version of the safety filter (a simple version is used).
