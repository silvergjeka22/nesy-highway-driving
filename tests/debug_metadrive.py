"""Debug script for MetaDrive bridge — run on Colab after loading cfg and model.

Usage (in colab_3 notebook, after loading showcase model):
    exec(open("tests/debug_metadrive.py").read())
"""
import numpy as np
from envs.metadrive_factory import (make_env_md, read_scene_md, read_kin_obs_md,
                                    nesy_md_action, count_passes_md)

env = make_env_md(cfg, seed=cfg["seed"])
obs, info = env.reset(seed=cfg["seed"])
fsm = cfg["fsm"]["initial_state"]
bridge = {}
_, ahead = count_passes_md(env, set())
overtakes = 0

print("=== MetaDrive Bridge Debug ===")
md = cfg["metadrive"]
print(f"v_max={md['v_max']}  max_steer={md.get('max_steer',1.0)}  "
      f"obs_x_offset={md.get('obs_x_offset',0)}  obs_speed_scale={md['obs_speed_scale']}")
print(f"v_cruise={0.6*md['v_max']:.1f} m/s (warmup threshold)")
print()

for step in range(300):
    scene = read_scene_md(env)
    ego = scene["ego"]

    action_out, fsm = nesy_md_action(showcase, env, cfg, fsm, shield=True, bridge=bridge)
    obs, r, term, trunc, info = env.step(action_out)

    passed, ahead = count_passes_md(env, ahead)
    overtakes += passed

    lat = bridge.get("lane_change")
    if lat is not None:
        phase = f"LANE->{lat['y_target']:+.1f}"
    elif ego["v"] < 0.6 * md["v_max"]:
        phase = "WARMUP"
    else:
        phase = "MODEL"

    if step % 10 == 0 or passed or lat is not None or term or trunc:
        print(f"step {step:3d} | v={ego['v']:5.2f} y={ego['y']:+5.1f} lane={ego['lane']} "
              f"on_road={ego.get('on_road', True)} | [{phase:11s}] "
              f"out=[{action_out[0]:+.3f},{action_out[1]:+.3f}] | fsm={fsm:13s} | ot={overtakes}")

    if term or trunc:
        print(f"  DONE at step {step}: crash={info.get('crash', False)} "
              f"arrive={info.get('arrive_dest', False)} offroad={info.get('out_of_road', False)}")
        break

env.close()
print(f"\n=== END DEBUG: {overtakes} overtakes, final v={ego['v']:.2f}, "
      f"on_road={ego.get('on_road', True)} ===")
