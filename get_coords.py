import numpy as np
from robot_grasp_rag.scenario_3_long_horizon_mujoco import RobosuiteAdapterLongHorizon as RobosuiteAdapter
env = RobosuiteAdapter(env_name="PickPlace", robot="Panda")
env.reset(seed=42)
actors = env.unwrapped.scene.get_all_actors()
for a in actors:
    if any(n in a.name.lower() for n in ["milk", "bread", "cereal", "can"]):
        print(a.name, a.pose.p)
