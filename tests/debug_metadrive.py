"""Debug script for MetaDrive bridge — run on Colab after loading cfg and model.

Usage (in colab_3 notebook, after loading showcase model):
    exec(open("tests/debug_metadrive.py").read())
"""
import numpy as np
from envs.metadrive_factory import make_env_md, read_scene_md, read_kin_obs_md, nesy_md_action
from nesy.roadmap import ACTIONS

env = make_env_md(cfg, seed=cfg["seed"])
obs, info = env.reset(seed=cfg["seed"])
fsm = cfg["fsm"]["initial_state"]
bridge = {}

print("=== MetaDrive Bridge Debug ===")
md = cfg["metadrive"]
print(f"v_max={md['v_max']}  max_steer={md.get('max_steer',1.0)}  "
      f"obs_x_offset={md.get('obs_x_offset',0)}")
print(f"obs_speed_scale={md['obs_speed_scale']}  lc_cooldown={md.get('lc_cooldown',12)}")
print(f"v_cruise={0.6*md['v_max']:.1f} m/s  (warmup threshold)")
print()

for step in range(60):
    scene = read_scene_md(env)
    kin_obs = read_kin_obs_md(env, cfg)

    action_out, fsm = nesy_md_action(showcase, env, cfg, fsm, shield=True, bridge=bridge)
    obs, r, term, trunc, info = env.step(action_out)

    v_cur = scene["ego"]["v"]
    on_road = scene["ego"].get("on_road", True)
    n_others = len(scene.get("others", []))

    v_native = action_out[0] * md["v_max"]
    steer_native = action_out[1] * md.get("max_steer", 1.0)
    phase = "WARMUP" if v_cur < 0.6 * md["v_max"] else "MODEL"

    print(f"step {step:3d} | v={v_cur:5.2f} on_road={on_road} | "
          f"[{phase:6s}] out=[{action_out[0]:+.3f},{action_out[1]:+.3f}] "
          f"steer={steer_native:+.3f} | cd={bridge.get('lc_cd',0)}")

    if step < 3 or step % 10 == 0:
        print(f"  ego obs: {kin_obs[0]}")
        print(f"  ego scene: x={scene['ego']['x']:.1f} y={scene['ego']['y']:.1f} "
              f"vx={scene['ego']['vx']:.2f} vy={scene['ego']['vy']:.2f}")

    if term or trunc:
        print(f"  DONE at step {step}: crashed={info.get('crash', False)}")
        break

env.close()
print(f"\n=== END DEBUG (final v={v_cur:.2f}, on_road={on_road}) ===")
