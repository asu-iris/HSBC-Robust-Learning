from envs.trifinger_env import TriFingerEnv
from envs import rotations
import numpy as np


# this simulation is a quasi-dynamic simulation:
# dynamic model is about the "position advancement" and the control is
# the desired robot joint displacement.
# the true dynamics (from torque/force to velocity to position) is hidden under the hood.

class env_param:
    def __init__(self):
        # ---------------------------------------------------------------------------------------------
        #      xml path, there are around 20 objects, specify any "env_trifinger_****.xml" you want
        # ---------------------------------------------------------------------------------------------
        self.model_path_ = 'envs/xmls/env_trifinger_cube.xml'

        # ---------------------------------------------------------------------------------------------
        #      initial state and target state
        # ---------------------------------------------------------------------------------------------
        np.random.seed(100)

        # random initial pose for object
        init_obj_height = 0.03
        init_xy_rand = 0.1 * np.random.rand(2) - 0.05
        yaw_angle = - np.pi * np.random.rand(1) + np.pi / 2
        init_obj_quat_rand = rotations.rpy_to_quaternion(np.hstack([yaw_angle, 0, 0]))

        self.init_obj_qpos_ = np.hstack((init_xy_rand, init_obj_height, init_obj_quat_rand))
        self.init_robot_qpos_ = np.array([
            0.0, -0.7, -1.1,
            0.0, -0.7, -1.1,
            0.0, -0.7, -1.1
        ])

        # random target pose for object
        target_xy_rand = 0.1 * np.random.rand(2) - 0.05
        self.target_p_ = np.hstack([target_xy_rand, init_obj_height])
        yaw_angle = np.pi * np.random.rand(1) - np.pi / 2
        self.target_q_ = rotations.rpy_to_quaternion(np.hstack([yaw_angle, 0, 0]))


# -------------------------------
#        init envs
# -------------------------------
param = env_param()
env = TriFingerEnv(param)

# -------------------------------
#        init cost function (you can customize inside)
# -------------------------------
path_cost_fn, final_cost_fn = env.init_cost_fns()
cost_params = np.concatenate([param.target_p_, param.target_q_])

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

    # evaluate cost function
    print(path_cost_fn(curr_x, curr_u, cost_params))
    print(final_cost_fn(curr_x, cost_params))
