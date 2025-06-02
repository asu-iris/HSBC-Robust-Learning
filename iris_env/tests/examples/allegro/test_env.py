import time

from envs.allegro_env import AllegroEnv
from envs import rotations
import numpy as np


# this dexterous manipulation simulation is a quasi-dynamic simulation:
# dynamic model is about the "position advancement" and the control input (command) is
# the desired robot joint displacement.
# the newtonian dynamics (from torque/force to velocity to position) is hidden under the hood.

class env_param:
    def __init__(self):
        # ---------------------------------------------------------------------------------------------
        #      xml path, there are around 20 objects, specify any "env_allegro_****.xml" you want
        # ---------------------------------------------------------------------------------------------
        self.model_path_ = 'envs/xmls/env_allegro_bowl.xml'

        # system dimensions:
        self.n_robot_qpos_ = 16
        self.n_qpos_ = 23
        self.n_qvel_ = 22
        self.n_cmd_ = 16

        # ---------------------------------------------------------------------------------------------
        #      initial state and target state
        # ---------------------------------------------------------------------------------------------
        np.random.seed(100)

        self.init_robot_qpos_ = np.array([
            0.125, 1.13, 1.45, 1.24,
            -0.02, 0.445, 1.17, 1.5,
            -0.459, 1.54, 1.11, 1.23,
            0.638, 1.85, 1.5, 1.26
        ])

        # random init and target pose for object
        init_obj_xy = np.array([-0.03, -0.01]) + 0.005 * np.random.randn(2)
        init_obj_pos = np.hstack([init_obj_xy, 0.03])
        init_yaw_angle = np.pi * np.random.rand(1) - np.pi / 2
        init_obj_quat_rand = rotations.rpy_to_quaternion(np.hstack([init_yaw_angle, 0, 0]))
        self.init_obj_qpos_ = np.hstack((init_obj_pos, init_obj_quat_rand))

        self.target_p_ = np.array([-0.02, -0.01, 0.03])
        yaw_angle = init_yaw_angle + np.random.choice([np.pi / 2, -np.pi / 2])
        self.target_q_ = rotations.rpy_to_quaternion(np.hstack([yaw_angle, 0, 0]))


# -------------------------------
#        init envs
# -------------------------------
param = env_param()
env = AllegroEnv(param)

# -------------------------------
#        init cost function (you can customize inside)
# -------------------------------
path_cost_fn, final_cost_fn = env.init_cost_fns()
cost_params = np.concatenate([param.target_p_, param.target_q_])

print('init done')
print(env.data_.qpos, type(env.data_.qpos))
print(env.mjx_data.qpos, type(env.mjx_data.qpos), env.mjx_data.qpos.devices())

# -------------------------------
#        random policy
# -------------------------------
while True:
    # state is [object pose + all finger joint positions]
    curr_x = env.get_state()

    # control input is the desired joint displacement (i.e., delta q)
    curr_u = 0.01 * np.random.randn(env.n_cmd_)

    # step forward
    env.step(curr_u)

    # # evaluate cost function
    # print(path_cost_fn(curr_x, curr_u, cost_params))
    # print(final_cost_fn(curr_x, cost_params))
