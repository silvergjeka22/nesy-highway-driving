"""Debug script for MetaDrive bridge — run on Colab after loading cfg and model.

Usage (in colab_3 notebook, after loading showcase model):
    exec(open("tests/debug_metadrive.py").read())
"""
import numpy as np
from envs.metadrive_factory import make_env_md, read_scene_md, read_kin_obs_md, nesy_md_action
from nesy.roadmap import ACTIONS, predicates, continuous_shield
from labs.lab1_cmd_vel import manoeuvre_to_cmd_vel
from labs.lab5_cbf import cbf_filter

env = make_env_md(cfg, seed=cfg["seed"])
obs, info = env.reset(seed=cfg["seed"])
fsm = cfg["fsm"]["initial_state"]

print("=== MetaDrive Bridge Debug ===")
print(f"v_max={cfg['metadrive']['v_max']}  cbf.v_max={cfg['cbf']['v_max']}  cbf.v_min={cfg['cbf']['v_min']}")
print(f"obs_speed_scale={cfg['metadrive']['obs_speed_scale']}")
print()

for step in range(20):
    scene = read_scene_md(env)
    kin_obs = read_kin_obs_md(env, cfg)
    a, _ = showcase.predict(kin_obs, deterministic=True)
    manoeuvre = ACTIONS[int(a)]

    v_cmd, omega_cmd = manoeuvre_to_cmd_vel(manoeuvre, scene, cfg)
    (v_cbf, w_cbf), cbf_int = cbf_filter(v_cmd, omega_cmd, scene, cfg)
    (v_full, w_full), full_int = continuous_shield(v_cmd, omega_cmd, scene, cfg)

    action_out, fsm = nesy_md_action(showcase, env, cfg, fsm, shield=False)
    obs, r, term, trunc, info = env.step(action_out)

    v_cur = scene["ego"]["v"]
    on_road = scene["ego"].get("on_road", True)
    n_others = len(scene.get("others", []))

    print(f"step {step:3d} | v={v_cur:5.2f} on_road={on_road} others={n_others} | "
          f"action={int(a)} ({manoeuvre:11s}) | "
          f"v_cmd={v_cmd:5.2f} w={omega_cmd:+.2f} | "
          f"cbf->v={v_cbf:5.2f} int={cbf_int} | "
          f"full->v={v_full:5.2f} int={full_int} | "
          f"out=[{action_out[0]:+.3f},{action_out[1]:+.3f}]")

    if step == 0:
        print(f"  ego obs row: {kin_obs[0]}")
        print(f"  ego scene: x={scene['ego']['x']:.1f} y={scene['ego']['y']:.1f} "
              f"vx={scene['ego']['vx']:.2f} vy={scene['ego']['vy']:.2f}")

    if term or trunc:
        print(f"  DONE at step {step}: crashed={info.get('crash', False)}")
        break

env.close()
print("\n=== END DEBUG ===")
