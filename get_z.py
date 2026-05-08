import robosuite
env = robosuite.make("PickPlace", robots="Panda", has_renderer=False, use_camera_obs=False)
env.reset()
import numpy as np
for i in range(100):
   env.step(np.zeros(env.action_dim))
for name, act in env.unwrapped.model.mujoco_objects.items():
   print(name)
