from robot_grasp_rag.scenario_3_long_horizon_mujoco import RobosuiteAdapterLongHorizon
env = RobosuiteAdapterLongHorizon(robot="Panda", gripper_types="RethinkGripper")
print(env.env.robots[0].gripper.name)
